"""
enrich_events.py
================
Futbol maç tracking + kalibrasyon verilerinden kural tabanlı mikro-aksiyonlar,
baskı istatistikleri ve oyuncu odaklı istatistikler türetir.

Girdi:
  --tracks                tracks.csv (ya da tracks_with_jersey.csv)
                          Sütunlar: frame_id, track_id, cls_id, bbox, team_id, jersey_number
  --calibration           calibration_events.json
                          Pipeline çıktısı — mevcut possession/pass olaylarını içerir.
  --calibration_frames    calibration_frames.jsonl  (opsiyonel ama önerilir)
                          Pipeline tarafından --out_frames ile üretilir;
                          her satırda per-frame world_xy koordinatları bulunur.
  --fps                   Kare hızı (varsayılan: 25)
  --output                Çıktı JSON dosyası (varsayılan: enriched_action_events.json)

Çıktı:  enriched_action_events.json
        Zamana göre sıralı JSON olay listesi — LLM spiker için zenginleştirilmiş metadata.

Çalıştırma örneği:
  python enrich_events.py \\
      --tracks output/tracks_with_jersey_RUN123.csv \\
      --calibration output/calibration_events_RUN123.json \\
      --calibration_frames output/calibration_frames_RUN123.jsonl \\
      --fps 25 \\
      --output output/enriched_action_events_RUN123.json

Bağımlılıklar: numpy, pandas  (requirements.txt'te zaten mevcut)
"""

from __future__ import annotations

import argparse
import json
import math
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Saha sabitleri — SoccerNet / NBJW kalibrasyon koordinat sistemi
#   x: -52.5 m (sol kale çizgisi) … 0 (orta çizgi) … +52.5 m (sağ kale çizgisi)
#   y: -34.0 m (alt kenar)         … 0 (orta)        … +34.0 m (üst kenar)
# ---------------------------------------------------------------------------

FIELD_HALF_L: float = 52.5   # Uzunluk yarısı (m)
FIELD_HALF_W: float = 34.0   # Genişlik yarısı (m)

# Final third: orta çizgiden itibaren son üçte bir dilim — |x| > 52.5/3 ≈ 17.5 m
FINAL_THIRD_X: float = FIELD_HALF_L / 3.0   # ≈ 17.5 m

# Ceza sahası sınırları: kale çizgisinden 16.5 m iç, 40.32 m geniş
PENALTY_DEPTH: float = 16.5
PENALTY_HALF_W: float = 20.16   # 40.32 / 2

# Ceza sahası x eşiği (mutlak değer): |x| > 36.0 m
PENALTY_X_THRESH: float = FIELD_HALF_L - PENALTY_DEPTH   # 36.0 m

# Top sahipliği için maksimum oyuncu–top dünya mesafesi eşiği (metre)
POSSESSION_DIST_M: float = 2.5

# Hakem/kaleci rezerve track_id eşiği (pipeline.py ile uyumlu)
SPECIAL_TRACK_MIN: int = 800_000_000


# ---------------------------------------------------------------------------
# Geometri yardımcıları
# ---------------------------------------------------------------------------

def _is_special_track(track_id: int) -> bool:
    """Hakem/kaleci olarak rezerve edilmiş track_id'leri filtreler."""
    return int(track_id) >= SPECIAL_TRACK_MIN


# ---------------------------------------------------------------------------
# Veri yükleme
# ---------------------------------------------------------------------------

def _parse_bbox(val: Any) -> Tuple[float, float, float, float]:
    """
    Farklı bbox formatlarını ayrıştırır:
      - "[x1, y1, x2, y2]"  →  Python string (en yaygın)
      - "x1 y1 x2 y2"       →  boşluk ayrımlı
      - [x1, y1, x2, y2]    →  zaten liste/tuple
    """
    if isinstance(val, (list, tuple)):
        return tuple(float(v) for v in val[:4])
    s = str(val).strip().strip("[](){}")
    nums = re.findall(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?", s)
    if len(nums) >= 4:
        return tuple(float(n) for n in nums[:4])
    raise ValueError(f"bbox ayrıştırılamadı: {val!r}")


def load_tracks(csv_path: str) -> pd.DataFrame:
    """
    tracks.csv / tracks_with_jersey.csv yükler ve standartlaştırır.

    Kabul edilen sütunlar:
      Gerekli : frame_id, track_id, cls_id, bbox  (veya x1/y1/x2/y2 ayrı sütunlar)
      Opsiyonel: team_id, jersey_number
    """
    df = pd.read_csv(csv_path)
    df.columns = [c.strip() for c in df.columns]

    # --- bbox ayrıştırma ---
    if "bbox" in df.columns and df["bbox"].dtype == object:
        bbox_parsed = df["bbox"].apply(_parse_bbox)
        df["bx1"] = bbox_parsed.apply(lambda b: b[0])
        df["by1"] = bbox_parsed.apply(lambda b: b[1])
        df["bx2"] = bbox_parsed.apply(lambda b: b[2])
        df["by2"] = bbox_parsed.apply(lambda b: b[3])
    elif all(c in df.columns for c in ("x1", "y1", "x2", "y2")):
        df = df.rename(columns={"x1": "bx1", "y1": "by1", "x2": "bx2", "y2": "by2"})
    else:
        for c in ("bx1", "by1", "bx2", "by2"):
            if c not in df.columns:
                df[c] = 0.0

    # Ayak noktası: bbox alt-merkez (kamera projeksiyonu için tercih edilir)
    df["px"] = (df["bx1"] + df["bx2"]) / 2.0
    df["py"] = df["by2"].astype(float)

    # --- Tip zorlama ---
    df["frame_id"] = pd.to_numeric(df["frame_id"], errors="coerce").fillna(0).astype(int)
    df["track_id"] = pd.to_numeric(df["track_id"], errors="coerce").fillna(-1).astype(int)
    df["cls_id"]   = pd.to_numeric(df.get("cls_id", 0), errors="coerce").fillna(0).astype(int)

    # Opsiyonel sütunlar — yoksa -1 ile doldur
    if "team_id" not in df.columns:
        df["team_id"] = -1
    else:
        df["team_id"] = pd.to_numeric(df["team_id"], errors="coerce").fillna(-1).astype(int)

    if "jersey_number" not in df.columns:
        df["jersey_number"] = -1
    else:
        df["jersey_number"] = pd.to_numeric(df["jersey_number"], errors="coerce").fillna(-1).astype(int)

    return df.sort_values("frame_id").reset_index(drop=True)


def load_calibration_frames_jsonl(jsonl_path: str) -> pd.DataFrame:
    """
    calibration_frames.jsonl yükler.

    Her satır formatı:
      {"frame_idx": N, "t": T, "data": {"players": [...], "ball": {"world_xy": [x, y]}}}

    Oyuncu formatı:
      {"track_id": N, "team_id": N, "world_xy": [x, y]}

    Döndürür: DataFrame — frame_idx, t, ball_x, ball_y, players (list[dict])
    """
    records: List[Dict[str, Any]] = []

    with open(jsonl_path, "r", encoding="utf-8") as f:
        for raw_line in f:
            line = raw_line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except Exception:
                continue

            frame_idx = int(rec.get("frame_idx", rec.get("frame_id", 0)))
            t_sec     = float(rec.get("t", 0.0))
            data      = rec.get("data") or {}

            # Top dünya koordinatı — kalibrasyon yoksa NaN
            ball   = data.get("ball") or {}
            bxy    = ball.get("world_xy")
            ball_x = float(bxy[0]) if bxy else float("nan")
            ball_y = float(bxy[1]) if bxy else float("nan")

            records.append({
                "frame_idx": frame_idx,
                "t":         t_sec,
                "ball_x":    ball_x,
                "ball_y":    ball_y,
                "players":   data.get("players") or [],  # per-row Python list
            })

    if not records:
        return pd.DataFrame(columns=["frame_idx", "t", "ball_x", "ball_y", "players"])

    return pd.DataFrame(records).sort_values("frame_idx").reset_index(drop=True)


def load_calibration_events(json_path: str) -> List[Dict[str, Any]]:
    """calibration_events.json içindeki 'events' listesini yükler."""
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if isinstance(data, list):
        return data
    ev = data.get("events")
    return list(ev) if isinstance(ev, list) else []


# ---------------------------------------------------------------------------
# Top sahipliği (possession) zaman serisi — iki ayrı yol
# ---------------------------------------------------------------------------

def build_possession_from_calib_frames(
    calib_frames: pd.DataFrame,
    tracks_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    calibration_frames.jsonl world_xy koordinatlarını kullanarak
    her frame için topa sahip oyuncuyu belirler.  (Tercih edilen yol)

    Algoritma:
    1. Her frame'deki 'players' listesindeki world_xy ile top world_xy arasındaki
       Öklid mesafesini hesapla.
    2. POSSESSION_DIST_M içindeki en yakın oyuncuyu seç
       (özel track_id'ler — hakem/kaleci — hariç tutulur).
    3. team_id ve jersey_number bilgisini tracks_df'ten tamamla.

    Döndürür: DataFrame — frame_idx, t, ball_x, ball_y,
                           possessing_track_id, possessing_team, jersey_number
    """
    # (frame_id, track_id) → (team_id, jersey_number) hızlı eşlemesi
    meta_lookup: Dict[Tuple[int, int], Tuple[int, int]] = {}
    for _, row in tracks_df[["frame_id", "track_id", "team_id", "jersey_number"]].iterrows():
        key = (int(row["frame_id"]), int(row["track_id"]))
        meta_lookup[key] = (int(row["team_id"]), int(row["jersey_number"]))

    rows: List[Dict[str, Any]] = []

    for _, row in calib_frames.iterrows():
        fidx = int(row["frame_idx"])
        t    = float(row["t"])
        bx   = float(row["ball_x"])
        by   = float(row["ball_y"])

        poss_tid = poss_team = poss_jersey = float("nan")

        if not (math.isnan(bx) or math.isnan(by)):
            ball_pos  = np.array([bx, by], dtype=np.float64)
            best_dist = POSSESSION_DIST_M
            best_tid = best_team = best_jrs = -1

            for player in (row["players"] or []):
                if not isinstance(player, dict):
                    continue
                tid = int(player.get("track_id", -1))
                if tid < 0 or _is_special_track(tid):
                    continue
                wxy = player.get("world_xy")
                if wxy is None:
                    continue

                ppos = np.array([float(wxy[0]), float(wxy[1])], dtype=np.float64)
                dist = float(np.linalg.norm(ppos - ball_pos))

                if dist < best_dist:
                    best_dist = dist
                    best_tid  = tid
                    # tracks_df'ten kimlik bilgisini tamamla
                    tm, jrs   = meta_lookup.get((fidx, tid), (-1, -1))
                    # Kalibrasyon frame'in kendi team_id'si de kabul edilir
                    best_team = tm if tm >= 0 else int(player.get("team_id", -1))
                    best_jrs  = jrs

            if best_tid >= 0:
                poss_tid    = float(best_tid)
                poss_team   = float(best_team)
                poss_jersey = float(best_jrs)

        rows.append({
            "frame_idx":           fidx,
            "t":                   t,
            "ball_x":              bx,
            "ball_y":              by,
            "possessing_track_id": poss_tid,
            "possessing_team":     poss_team,
            "jersey_number":       poss_jersey,
        })

    return pd.DataFrame(rows).sort_values("frame_idx").reset_index(drop=True)


def build_possession_from_tracks_only(
    tracks_df: pd.DataFrame,
    fps: float,
) -> pd.DataFrame:
    """
    Fallback: calibration_frames.jsonl yoksa tracks.csv bbox piksel
    koordinatlarından sahiplik türetir.

    Top cls_id otomatik belirlenir (en küçük medyan bbox alanı = top).
    Piksel koordinatları SoccerNet koordinat sistemine normalize edilir.
    """
    df = tracks_df.copy()

    # Top sınıfı tespiti: en küçük medyan bbox alanı
    df["_area"]  = (df["bx2"] - df["bx1"]) * (df["by2"] - df["by1"])
    ball_cls     = int(df.groupby("cls_id")["_area"].median().idxmin())

    # Piksel → normalize saha koordinatı (±52.5 x, ±34 y)
    max_px = float(df["px"].max()) or 1.0
    max_py = float(df["py"].max()) or 1.0
    sx = (FIELD_HALF_L * 2.0) / max_px
    sy = (FIELD_HALF_W * 2.0) / max_py

    ball_df = df[df["cls_id"] == ball_cls].copy()
    ball_df["ball_x"] = ball_df["px"] * sx - FIELD_HALF_L
    ball_df["ball_y"] = ball_df["py"] * sy - FIELD_HALF_W
    ball_df = (
        ball_df[["frame_id", "ball_x", "ball_y"]]
        .rename(columns={"frame_id": "frame_idx"})
    )

    player_df = df[
        (df["cls_id"] != ball_cls) &
        (df["track_id"] >= 0) &
        ~(df["track_id"].apply(_is_special_track))
    ].copy()
    player_df["field_x"] = player_df["px"] * sx - FIELD_HALF_L
    player_df["field_y"] = player_df["py"] * sy - FIELD_HALF_W

    merged = player_df.merge(ball_df, left_on="frame_id", right_on="frame_idx", how="inner")
    if merged.empty:
        return pd.DataFrame(columns=[
            "frame_idx", "t", "ball_x", "ball_y",
            "possessing_track_id", "possessing_team", "jersey_number",
        ])

    # Her frame için en yakın oyuncu (vektörize idxmin)
    merged["dist"] = np.hypot(
        merged["field_x"] - merged["ball_x"],
        merged["field_y"] - merged["ball_y"],
    )
    poss = merged.loc[merged.groupby("frame_id")["dist"].idxmin()].copy()

    # Çok uzak → sahipsiz
    far = poss["dist"] > POSSESSION_DIST_M
    for col in ("track_id", "team_id", "jersey_number"):
        poss.loc[far, col] = float("nan")

    poss["t"] = poss["frame_id"] / fps
    poss = poss.rename(columns={
        "frame_id": "frame_idx",
        "track_id": "possessing_track_id",
        "team_id":  "possessing_team",
    })

    return (
        poss[["frame_idx", "t", "ball_x", "ball_y",
              "possessing_track_id", "possessing_team", "jersey_number"]]
        .reset_index(drop=True)
    )


# ---------------------------------------------------------------------------
# Modül 1 — Pas, Pas Serisi, Top Çalma (Araya Girme)
# ---------------------------------------------------------------------------

def detect_passes_and_interceptions(
    possession: pd.DataFrame,
    pass_series_threshold: int = 5,
    min_frames_stable: int = 5,
) -> List[Dict[str, Any]]:
    """
    Sahiplik değişimlerinden pas, pas serisi ve top çalma olayları türetir.

    Algoritma (vektörize):
    1. track_id değişimlerini shift() ile tespit et → cumsum() ile blok etiketle.
    2. Titreşimleri elemek için blok büyüklük filtresi (< min_frames_stable → gözardı).
    3. Stabil blok temsilcilerini karşılaştır:
       - Aynı takım + farklı oyuncu  → pass
       - Farklı takım               → interception
    4. Ardışık pas sayacı → pass_series_threshold aşılınca pass_series olayı.
    """
    events: List[Dict[str, Any]] = []

    pos = possession.dropna(subset=["possessing_track_id", "possessing_team"]).copy()
    if pos.empty:
        return events

    pos["possessing_track_id"] = pos["possessing_track_id"].astype(int)
    pos["possessing_team"]     = pos["possessing_team"].astype(int)
    pos["jersey_number"]       = pos["jersey_number"].fillna(-1).astype(int)

    # Sahiplik bloklarını etiketle: her track_id değişimi yeni blok başlatır
    track_chg   = pos["possessing_track_id"] != pos["possessing_track_id"].shift(1)
    pos["blk"]  = track_chg.cumsum()

    # Vektörize blok boyutu (groupby transform ile per-row)
    pos["blk_size"] = pos.groupby("blk")["blk"].transform("count")

    # Yalnızca stabil bloklar
    stable = pos[pos["blk_size"] >= min_frames_stable]
    if stable.empty:
        return events

    # Her bloktan ilk satırı al (blok temsilcisi)
    reps = stable.groupby("blk").first().reset_index()

    # Önceki blokla karşılaştırma için shift(1)
    reps["prev_track"] = reps["possessing_track_id"].shift(1)
    reps["prev_team"]  = reps["possessing_team"].shift(1)
    reps = reps.dropna(subset=["prev_track", "prev_team"])
    reps["prev_track"] = reps["prev_track"].astype(int)
    reps["prev_team"]  = reps["prev_team"].astype(int)

    same_team_streak: int = 0
    streak_team: Optional[int] = None

    for _, row in reps.iterrows():
        t         = float(row["t"])
        curr_tid  = int(row["possessing_track_id"])
        curr_team = int(row["possessing_team"])
        prev_team = int(row["prev_team"])
        prev_tid  = int(row["prev_track"])
        jersey    = int(row["jersey_number"])

        if curr_team == prev_team and curr_tid != prev_tid:
            # ---- Başarılı pas (aynı takım, farklı oyuncu) ----
            events.append({
                "timestamp_sec": round(t, 2),
                "event_type":    "pass",
                "team_id":       curr_team,
                "jersey_number": jersey,
                "description":   f"Takım {curr_team} içinde pas aktarımı (alıcı #{jersey})",
            })

            # Ardışık pas sayacı güncelle
            if streak_team == curr_team:
                same_team_streak += 1
            else:
                same_team_streak = 1
                streak_team      = curr_team

            if same_team_streak >= pass_series_threshold:
                events.append({
                    "timestamp_sec": round(t, 2),
                    "event_type":    "pass_series",
                    "team_id":       curr_team,
                    "jersey_number": -1,
                    "description": (
                        f"Takım {curr_team} {same_team_streak} ardışık başarılı pas — "
                        "güçlü oyun kurma pozisyonu"
                    ),
                })
                same_team_streak = 0  # Seriyi sıfırla; yeniden sayım başlar

        elif curr_team != prev_team:
            # ---- Top çalma / araya girme (farklı takım) ----
            events.append({
                "timestamp_sec": round(t, 2),
                "event_type":    "interception",
                "team_id":       curr_team,
                "jersey_number": jersey,
                "description":   f"Takım {curr_team} topu kaptı — #{jersey} araya girdi",
            })
            same_team_streak = 0
            streak_team      = None

    return events


# ---------------------------------------------------------------------------
# Modül 2 — Top Sürme (Dribbling)
# ---------------------------------------------------------------------------

def detect_dribbling(
    possession: pd.DataFrame,
    fps: float,
    min_duration_sec: float = 3.0,
    min_distance_m: float = 5.0,
    cooldown_sec: float = 10.0,
) -> List[Dict[str, Any]]:
    """
    Bir oyuncu sahiplikte kalırken belirli sürede yeterli mesafe kat ederse
    'dribbling' olayı üretir.

    Algoritma (vektörize):
    1. Sahiplik bloklarını cumsum() ile etiketle.
    2. Her blok içinde top koordinatlarının adım mesafelerini diff() + hypot() ile hesapla.
    3. rolling(window_size).sum() ile pencere içi kümülatif mesafe.
    4. Eşiği ilk aşan anda cooldown kontrollü olay üret.
    """
    events: List[Dict[str, Any]] = []

    pos = possession.dropna(subset=["possessing_track_id", "ball_x", "ball_y"]).copy()
    if pos.empty:
        return events

    # Sahiplik bloklarını etiketle
    track_chg  = pos["possessing_track_id"] != pos["possessing_track_id"].shift(1)
    pos["blk"] = track_chg.cumsum()

    window_size = max(1, int(min_duration_sec * fps))
    cooldown: Dict[int, float] = {}   # track_id → son olay zamanı

    for (_, tid_raw), grp in pos.groupby(["blk", "possessing_track_id"]):
        tid = int(tid_raw)

        if len(grp) < window_size:
            continue   # Blok yeterince uzun değil

        grp = grp.sort_values("t")

        # Adım adım mesafe (vektörel diff → hypot)
        dx   = grp["ball_x"].diff().fillna(0.0)
        dy   = grp["ball_y"].diff().fillna(0.0)
        step = np.hypot(dx.values, dy.values)

        # Kayan pencere kümülatif mesafesi
        roll = (
            pd.Series(step, index=grp.index)
            .rolling(window=window_size, min_periods=window_size)
            .sum()
        )

        # Eşiği ilk aşan an
        exceeds = roll[roll >= min_distance_m]
        if exceeds.empty:
            continue

        t = float(grp.at[exceeds.index[0], "t"])

        # Cooldown: aynı oyuncu cooldown_sec içinde tekrar tetiklemesin
        if cooldown.get(tid, -9999.0) + cooldown_sec > t:
            continue
        cooldown[tid] = t

        jersey = int(grp["jersey_number"].fillna(-1).iloc[0])
        team   = int(grp["possessing_team"].fillna(-1).iloc[0])

        events.append({
            "timestamp_sec": round(t, 2),
            "event_type":    "dribbling",
            "team_id":       team,
            "jersey_number": jersey,
            "description": (
                f"Takım {team}'in {jersey} numaralı oyuncusu topla hızla ilerliyor "
                f"({min_duration_sec:.0f}s içinde ≥{min_distance_m:.0f}m)"
            ),
        })

    return events


# ---------------------------------------------------------------------------
# Modül 3 — Tehlikeli Bölge Girişi
# ---------------------------------------------------------------------------

def detect_zone_entries(
    possession: pd.DataFrame,
    calib_frames: Optional[pd.DataFrame] = None,
    cooldown_sec: float = 5.0,
    min_opponents_in_zone: int = 1,
    opponent_zone_margin_m: float = 5.0,
) -> List[Dict[str, Any]]:
    """
    Topun tehlikeli bölgelere girişini tespit eder:
      - final_third_right / final_third_left : Saha üçte biri
      - penalty_right / penalty_left         : Ceza sahası

    Filtreler:
    1. Sahiplik bilgisi zorunlu — sahipsiz top girisi olay üretmez.
    2. calib_frames mevcutsa rakip oyuncu varlığı kontrolü:
       Bölgeye giriş sırasında rakip takımdan en az ``min_opponents_in_zone``
       oyuncu bölge sınırından ``opponent_zone_margin_m`` metre içinde olmalıdır.
       Rakip yoksa bölge girişi savunmasız sayılır ve olay filtrelenir.

    Algoritma (vektörize):
    1. Boolean maskeleri (np.ndarray) ile her frame'in bölge durumunu belirle.
    2. Önceki frame maskeyle karşılaştır (prev_mask = concat([False], mask[:-1])).
    3. mask & ~prev_mask → 0→1 geçişleri (bölgeye giriş anları).
    4. Cooldown ile sık tetiklenmeyi önle.
    """
    events: List[Dict[str, Any]] = []

    pos = possession.dropna(subset=["ball_x", "ball_y"]).copy().reset_index(drop=True)
    if pos.empty:
        return events

    bx = pos["ball_x"].values.astype(np.float64)
    by = pos["ball_y"].values.astype(np.float64)
    t  = pos["t"].values.astype(np.float64)

    # --- Kalibrasyon frame zaman indeksi (hızlı arama için) ---
    # Yapı: {frame_idx: players_list}
    calib_players_by_t: Optional[Dict[float, List[Dict[str, Any]]]] = None
    if calib_frames is not None and not calib_frames.empty:
        # her frame'in yaklaşık zamanını float key ile sakla
        calib_players_by_t = {}
        for _, cfrow in calib_frames.iterrows():
            calib_players_by_t[float(cfrow["t"])] = cfrow["players"] or []
        _calib_times = np.array(sorted(calib_players_by_t.keys()), dtype=np.float64)
    else:
        _calib_times = np.array([], dtype=np.float64)

    def _opponents_in_zone(
        ts: float,
        attacker_team: int,
        zone_x_min: float,
        zone_x_max: float,
        zone_y_min: float,
        zone_y_max: float,
    ) -> int:
        """Belirtilen bölge içinde ya da yakınında rakip oyuncu sayısını döndürür."""
        if calib_players_by_t is None or len(_calib_times) == 0:
            # Kalibrasyon verisi yok — filtrelenemez, olay kabul edilir
            return min_opponents_in_zone

        # En yakın kalibrasyon frame’ini bul
        idx = int(np.searchsorted(_calib_times, ts))
        if idx >= len(_calib_times):
            idx = len(_calib_times) - 1
        if idx > 0 and abs(_calib_times[idx - 1] - ts) < abs(_calib_times[idx] - ts):
            idx -= 1
        players = calib_players_by_t.get(float(_calib_times[idx]), [])

        count = 0
        for p in players:
            if not isinstance(p, dict):
                continue
            team = int(p.get("team_id", -1))
            if team < 0 or team == attacker_team:
                continue  # same team or unknown
            wxy = p.get("world_xy")
            if wxy is None:
                continue
            px_val, py_val = float(wxy[0]), float(wxy[1])
            # Bölge sınırından margin kadar genişletilmiş kutu içinde mi?
            if (
                zone_x_min - opponent_zone_margin_m <= px_val <= zone_x_max + opponent_zone_margin_m
                and zone_y_min - opponent_zone_margin_m <= py_val <= zone_y_max + opponent_zone_margin_m
            ):
                count += 1
        return count

    # Bölge tanımları: (mask, desc, x_min, x_max, y_min, y_max)
    # Rakip kontrol kutuları bölge kutusu — margin fonksiyon içinde ekleniyor
    FIELD_W = FIELD_HALF_W
    zone_defs: Dict[str, Tuple[np.ndarray, str, float, float, float, float]] = {
        "final_third_right": (
            bx > FINAL_THIRD_X,
            "Sağ taraf final third — tehlikeli bölge girişi",
            FINAL_THIRD_X, FIELD_HALF_L, -FIELD_W, FIELD_W,
        ),
        "final_third_left": (
            bx < -FINAL_THIRD_X,
            "Sol taraf final third — tehlikeli bölge girişi",
            -FIELD_HALF_L, -FINAL_THIRD_X, -FIELD_W, FIELD_W,
        ),
        "penalty_right": (
            (bx > PENALTY_X_THRESH) & (np.abs(by) < PENALTY_HALF_W),
            "Sağ ceza sahasına giriş! Tehlikeli atak",
            PENALTY_X_THRESH, FIELD_HALF_L, -PENALTY_HALF_W, PENALTY_HALF_W,
        ),
        "penalty_left": (
            (bx < -PENALTY_X_THRESH) & (np.abs(by) < PENALTY_HALF_W),
            "Sol ceza sahasına giriş! Tehlikeli atak",
            -FIELD_HALF_L, -PENALTY_X_THRESH, -PENALTY_HALF_W, PENALTY_HALF_W,
        ),
    }

    last_t: Dict[str, float] = {}

    for zone_name, (mask, desc, zx0, zx1, zy0, zy1) in zone_defs.items():
        # 0→1 geçişi: önceki frame dışarıda, bu frame içeride
        prev_mask  = np.concatenate([[False], mask[:-1]])
        entry_mask = mask & ~prev_mask

        for i in np.where(entry_mask)[0]:
            ts = float(t[i])
            if ts - last_t.get(zone_name, -9999.0) < cooldown_sec:
                continue

            # Sahiplik bilgisi zorunlu
            poss_team   = pos.at[i, "possessing_team"]
            poss_jersey = pos.at[i, "jersey_number"]
            if pd.isna(poss_team) or int(poss_team) < 0:
                continue  # sahipsiz top — tehdit sayılmaz

            attacker_team = int(poss_team)

            # Rakip varlığı kontrolü
            n_opp = _opponents_in_zone(ts, attacker_team, zx0, zx1, zy0, zy1)
            if n_opp < min_opponents_in_zone:
                continue  # bölgede rakip yok — savunmasız, tehdit değil

            last_t[zone_name] = ts

            events.append({
                "timestamp_sec": round(ts, 2),
                "event_type":    "zone_entry",
                "zone":          zone_name,
                "team_id":       attacker_team,
                "jersey_number": int(poss_jersey) if pd.notna(poss_jersey) else -1,
                "opponents_in_zone": n_opp,
                "description":   (
                    f"{desc} "
                    f"(x={bx[i]:.1f}m, y={by[i]:.1f}m, "
                    f"rakip sayısı={n_opp})"
                ),
            })

    return events


# ---------------------------------------------------------------------------
# Modül 4 — Baskı İstatistikleri (Rolling Window)
# ---------------------------------------------------------------------------

def compute_pressure_events(
    possession: pd.DataFrame,
    fps: float,
    calib_frames: Optional[pd.DataFrame] = None,
    window_sec: float = 40.0,
    dominance_threshold: float = 0.70,
    cooldown_sec: float = 20.0,
    min_pressers: int = 2,
    press_radius_m: float = 6.0,
) -> List[Dict[str, Any]]:
    """
    Baskı olayları: iki çıkarım yöntemi, calib_frames varlığına göre seçilir.

    **calib_frames mevcut (tercih edilen):**
      Her frame için topa sahip takımın etrafındaki rakip oyuncu sayısını hesapla.
      ``press_radius_m`` metre içinde en az ``min_pressers`` rakip varsa
      o frame "baskı altında" sayılır.
      ``window_sec`` içinde bu oran ``dominance_threshold``'u aştığında
      cooldown kontrollü baskı olayı üretilir.

    **calib_frames yok (fallback):**
      40 saniyelik kaydırma penceresiyle topun hangi yarıda kaldığını ölçer.
      Bu yöntem baskıyı değil alan üstenligi ölçer; olay etiketinde belirtilir.
    """
    events: List[Dict[str, Any]] = []

    pos = possession.dropna(subset=["ball_x"]).copy().reset_index(drop=True)
    if pos.empty:
        return events

    window_frames = max(1, int(window_sec * fps))
    t_arr  = pos["t"].values.astype(np.float64)

    # ==================================================================
    # YOL A: calib_frames mevcut — gerçek yakın baskı tespiti
    # ==================================================================
    if calib_frames is not None and not calib_frames.empty:
        cf_times   = calib_frames["t"].values.astype(np.float64)
        cf_players: List[List[Dict[str, Any]]] = list(calib_frames["players"])

        press_arr      = np.zeros(len(pos), dtype=np.float32)
        press_team_arr = np.full(len(pos), -1, dtype=np.int32)

        for i in range(len(pos)):
            poss_team = pos.at[i, "possessing_team"]
            if pd.isna(poss_team) or int(poss_team) < 0:
                continue
            attacker_team = int(poss_team)

            bx_val = float(pos.at[i, "ball_x"])
            by_col = "ball_y" if "ball_y" in pos.columns else None
            by_val = float(pos.at[i, by_col]) if by_col else float("nan")
            if math.isnan(bx_val):
                continue

            cf_idx = int(np.searchsorted(cf_times, t_arr[i]))
            if cf_idx >= len(cf_times):
                cf_idx = len(cf_times) - 1
            if cf_idx > 0 and abs(cf_times[cf_idx - 1] - t_arr[i]) < abs(cf_times[cf_idx] - t_arr[i]):
                cf_idx -= 1

            ball_pos = np.array([bx_val, by_val if not math.isnan(by_val) else 0.0], dtype=np.float64)
            n_pressers = 0
            opp_team = -1
            for p in (cf_players[cf_idx] or []):
                if not isinstance(p, dict):
                    continue
                team = int(p.get("team_id", -1))
                if team < 0 or team == attacker_team:
                    continue
                wxy = p.get("world_xy")
                if wxy is None:
                    continue
                dist = float(np.linalg.norm(
                    np.array([float(wxy[0]), float(wxy[1])], dtype=np.float64) - ball_pos
                ))
                if dist <= press_radius_m:
                    n_pressers += 1
                    opp_team = team

            if n_pressers >= min_pressers:
                press_arr[i] = 1.0
                press_team_arr[i] = opp_team

        press_ratio = (
            pd.Series(press_arr.astype(float))
            .rolling(window=window_frames, min_periods=window_frames // 2)
            .mean()
            .values
        )

        last_t: float = -9999.0
        for i in range(len(pos)):
            if math.isnan(press_ratio[i]):
                continue
            ts = float(t_arr[i])
            if press_ratio[i] < dominance_threshold or ts - last_t < cooldown_sec:
                continue
            last_t = ts

            win_start   = max(0, i - window_frames + 1)
            win_teams   = press_team_arr[win_start: i + 1]
            valid_teams = win_teams[win_teams >= 0]
            if len(valid_teams) > 0:
                pressing_team = int(np.bincount(valid_teams).argmax())
            else:
                pressing_team = -1

            events.append({
                "timestamp_sec": round(ts, 2),
                "event_type":    "pressure",
                "team_id":       pressing_team,
                "jersey_number": -1,
                "press_ratio":   round(float(press_ratio[i]), 2),
                "description": (
                    f"Takım {pressing_team} yoğun baskı uyguluyor — "
                    f"son {window_sec:.0f}s içinde top taşıyıcının etrafında "
                    f"çoklu rakip (oran: %{press_ratio[i] * 100:.0f})"
                ),
            })

        return events

    # ==================================================================
    # YOL B: calib_frames yok — alan üstenligi (fallback, düşük öncelik)
    # ==================================================================
    pos["in_right"]    = (pos["ball_x"] > 0.0).astype(float)
    pos["right_ratio"] = (
        pos["in_right"]
        .rolling(window=window_frames, min_periods=window_frames // 2)
        .mean()
    )

    rr_arr = pos["right_ratio"].values

    last_right_t: float = -9999.0
    last_left_t:  float = -9999.0

    for i in range(len(pos)):
        if math.isnan(rr_arr[i]):
            continue
        ts = float(t_arr[i])
        rr = float(rr_arr[i])

        if rr >= dominance_threshold and ts - last_right_t >= cooldown_sec:
            events.append({
                "timestamp_sec": round(ts, 2),
                "event_type":    "territory_dominance_right",
                "team_id":       -1,
                "jersey_number": -1,
                "description": (
                    f"Son {window_sec:.0f} saniyedir top sağ yarı sahada "
                    f"(oran: %{rr * 100:.0f}) — sol takım alan üstenlendi"
                ),
            })
            last_right_t = ts

        elif rr <= (1.0 - dominance_threshold) and ts - last_left_t >= cooldown_sec:
            events.append({
                "timestamp_sec": round(ts, 2),
                "event_type":    "territory_dominance_left",
                "team_id":       -1,
                "jersey_number": -1,
                "description": (
                    f"Son {window_sec:.0f} saniyedir top sol yarı sahada "
                    f"(oran: %{(1 - rr) * 100:.0f}) — sağ takım alan üstenlendi"
                ),
            })
            last_left_t = ts

    return events

def compute_player_activity_events(
    possession: pd.DataFrame,
    period_sec: float = 120.0,
    touch_threshold: int = 3,
) -> List[Dict[str, Any]]:
    """
    Belirli periyotlarda (varsayılan: 2 dakika) en az touch_threshold kez
    topa dokunan oyuncular için 'active_player' olayı üretir.

    Dokunuş tanımı: sahiplik bloğunun başlangıç frame'i
    (aynı oyuncu için ardışık frame'ler tek dokunuş sayılır).

    Algoritma (vektörize):
    1. cumsum() ile sahiplik blok başlangıçlarını (dokunuşları) işaretle.
    2. period_idx = t // period_sec ile periyot dilimine at.
    3. groupby([period_idx, track_id]).size() → dokunuş sayısı.
    4. Eşiği aşan oyuncular için olay üret.
    """
    events: List[Dict[str, Any]] = []

    pos = possession.dropna(subset=["possessing_track_id", "possessing_team"]).copy()
    if pos.empty:
        return events

    pos["possessing_track_id"] = pos["possessing_track_id"].astype(int)
    pos["possessing_team"]     = pos["possessing_team"].astype(int)
    pos["jersey_number"]       = pos["jersey_number"].fillna(-1).astype(int)

    # Sahiplik blok başlangıçları = dokunuşlar
    track_chg       = pos["possessing_track_id"] != pos["possessing_track_id"].shift(1)
    pos["touch"]    = track_chg.astype(int)
    pos["period_idx"] = (pos["t"] // period_sec).astype(int)

    # Dokunuş satırları — ardışık aynı-oyuncu frame'leri tek kez sayılır
    touch_rows = pos[pos["touch"] == 1]

    # Periyot × oyuncu dokunuş sayısı (vektörize groupby)
    touch_counts = (
        touch_rows
        .groupby(["period_idx", "possessing_track_id", "possessing_team", "jersey_number"])
        .size()
        .reset_index(name="touch_count")
    )

    active = touch_counts[touch_counts["touch_count"] >= touch_threshold]

    for _, row in active.iterrows():
        team   = int(row["possessing_team"])
        jersey = int(row["jersey_number"])
        tid    = int(row["possessing_track_id"])
        pidx   = int(row["period_idx"])

        # Periyottaki ilk dokunuş zamanı
        first_t = touch_rows[
            (touch_rows["period_idx"]          == pidx) &
            (touch_rows["possessing_track_id"] == tid)
        ]
        ts = float(first_t["t"].iloc[0]) if not first_t.empty else float(pidx * period_sec)

        events.append({
            "timestamp_sec": round(ts, 2),
            "event_type":    "active_player",
            "team_id":       team,
            "jersey_number": jersey,
            "description": (
                f"Takım {team}'de {jersey} numaralı oyuncu çok aktif "
                f"({int(row['touch_count'])} dokunuş, son {period_sec / 60:.0f} dakikada)"
            ),
        })

    return events


# ---------------------------------------------------------------------------
# Kalibrasyon olaylarını ortak çıktı formatına dönüştür
# ---------------------------------------------------------------------------

def _normalize_calib_event(ev: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """
    calibration_events.json içindeki ham olayları çıktı JSON şemasına çevirir.
    Desteklenen tipler: possession_start, pass, possession_change.
    Diğerleri 'calibration_<type>' öneki ile aktarılır.
    """
    ev_type = str(ev.get("type") or ev.get("event_type") or "unknown")
    t       = float(ev.get("t", 0.0))
    team_id = int(ev.get("team_id", -1))
    jersey  = int(ev.get("jersey_number", -1))

    if ev_type == "possession_start":
        return {
            "timestamp_sec": round(t, 2),
            "event_type":    "possession_start",
            "team_id":       team_id,
            "jersey_number": jersey,
            "description":   (
                f"Takım {team_id} topa sahip "
                f"(track #{ev.get('player_track_id', '?')})"
            ),
        }

    if ev_type == "pass":
        dist = float(ev.get("distance_m", 0.0))
        return {
            "timestamp_sec": round(t, 2),
            "event_type":    "pass",
            "team_id":       team_id,
            "jersey_number": jersey,
            "distance_m":    round(dist, 1),
            "description":   f"Takım {team_id} pas — mesafe: {dist:.1f}m",
        }

    if ev_type == "possession_change":
        from_team = int(ev.get("from_team_id", -1))
        return {
            "timestamp_sec": round(t, 2),
            "event_type":    "interception",
            "team_id":       team_id,
            "from_team_id":  from_team,
            "jersey_number": jersey,
            "description":   (
                f"Sahiplik değişimi — Takım {from_team} → Takım {team_id}"
            ),
        }

    # Bilinmeyen kalibrasyon olayı — olduğu gibi aktar
    return {
        "timestamp_sec": round(t, 2),
        "event_type":    f"calibration_{ev_type}",
        "team_id":       team_id,
        "jersey_number": jersey,
        "description":   json.dumps(ev, ensure_ascii=False),
    }


# ---------------------------------------------------------------------------
# Ana giriş noktası
# ---------------------------------------------------------------------------

def main() -> None:
    global POSSESSION_DIST_M

    parser = argparse.ArgumentParser(
        description=(
            "Futbol tracking + kalibrasyon verisinden kural tabanlı mikro-aksiyonlar, "
            "baskı istatistikleri ve oyuncu aktivite olayları türetir."
        )
    )
    parser.add_argument(
        "--tracks", required=True,
        help="tracks.csv veya tracks_with_jersey.csv yolu",
    )
    parser.add_argument(
        "--calibration", required=True,
        help="calibration_events.json yolu",
    )
    parser.add_argument(
        "--calibration_frames", default=None,
        help="calibration_frames.jsonl yolu (opsiyonel — world_xy için önerilir)",
    )
    parser.add_argument(
        "--output", default="enriched_action_events.json",
        help="Çıktı JSON dosyası (varsayılan: enriched_action_events.json)",
    )
    parser.add_argument(
        "--fps", type=float, default=25.0,
        help="Kare hızı (varsayılan: 25)",
    )
    parser.add_argument(
        "--possession_threshold", type=float, default=POSSESSION_DIST_M,
        help=f"Top sahipliği mesafe eşiği — metre (varsayılan: {POSSESSION_DIST_M})",
    )
    parser.add_argument(
        "--pass_series_n", type=int, default=5,
        help="Pas serisi eşiği — ardışık başarılı pas sayısı (varsayılan: 5)",
    )
    parser.add_argument(
        "--dribble_sec", type=float, default=3.0,
        help="Dribbling minimum süre — saniye (varsayılan: 3.0)",
    )
    parser.add_argument(
        "--dribble_m", type=float, default=5.0,
        help="Dribbling minimum mesafe — metre (varsayılan: 5.0)",
    )
    parser.add_argument(
        "--pressure_window_sec", type=float, default=40.0,
        help="Baskı istatistiği kayan pencere süresi — saniye (varsayılan: 40)",
    )
    parser.add_argument(
        "--pressure_threshold", type=float, default=0.70,
        help="Baskı baskınlık oranı eşiği 0–1 (varsayılan: 0.70)",
    )
    parser.add_argument(
        "--activity_period_sec", type=float, default=120.0,
        help="Oyuncu aktivite periyodu — saniye (varsayılan: 120)",
    )
    parser.add_argument(
        "--activity_touches", type=int, default=3,
        help="Aktif oyuncu dokunuş eşiği (varsayılan: 3)",
    )
    parser.add_argument(
        "--no_calib_events", action="store_true",
        help="calibration_events.json olaylarını çıktıya ekleme",
    )
    args = parser.parse_args()

    # Modül düzeyi eşiği güncelle
    POSSESSION_DIST_M = args.possession_threshold

    # ------------------------------------------------------------------
    # [1/5] Veri yükleme
    # ------------------------------------------------------------------
    print(f"[1/5] Tracks yükleniyor: {args.tracks}")
    tracks_df = load_tracks(args.tracks)
    print(
        f"      {len(tracks_df):,} satır | "
        f"{tracks_df['track_id'].nunique()} benzersiz track | "
        f"{tracks_df['frame_id'].nunique()} frame"
    )

    print(f"[2/5] Kalibrasyon olayları yükleniyor: {args.calibration}")
    calib_events = load_calibration_events(args.calibration)
    print(f"      {len(calib_events)} kalibrasyon olayı")

    # ------------------------------------------------------------------
    # [3/5] Possession (top sahipliği) zaman serisi
    # ------------------------------------------------------------------
    print("[3/5] Possession (top sahipliği) zaman serisi oluşturuluyor...")

    calib_frames_path = args.calibration_frames
    if calib_frames_path and Path(calib_frames_path).exists():
        print("      Mod: calibration_frames.jsonl (world_xy — tercih edilen)")
        calib_frames = load_calibration_frames_jsonl(calib_frames_path)
        print(f"      {len(calib_frames):,} kalibrasyon frame'i yüklendi.")
        possession = build_possession_from_calib_frames(
            calib_frames=calib_frames,
            tracks_df=tracks_df,
        )
    else:
        calib_frames = None
        print(
            "      Mod: tracks.csv piksel koordinatları (fallback — "
            "calibration_frames.jsonl sağlanmadı veya bulunamadı)"
        )
        possession = build_possession_from_tracks_only(tracks_df, fps=args.fps)

    valid_frames = possession["possessing_track_id"].notna().sum()
    total_frames = len(possession)
    pct = valid_frames / max(1, total_frames) * 100
    print(f"      {total_frames:,} frame | {valid_frames:,} sahipli (%{pct:.1f})")

    # ------------------------------------------------------------------
    # [4/5] Olay tespiti
    # ------------------------------------------------------------------
    print("[4/5] Mikro-aksiyonlar ve istatistikler hesaplanıyor...")

    all_events: List[Dict[str, Any]] = []

    ev_pass = detect_passes_and_interceptions(
        possession,
        pass_series_threshold=args.pass_series_n,
    )
    print(f"      Pas / Araya girme : {len(ev_pass):>4} olay")
    all_events.extend(ev_pass)

    ev_drib = detect_dribbling(
        possession,
        fps=args.fps,
        min_duration_sec=args.dribble_sec,
        min_distance_m=args.dribble_m,
    )
    print(f"      Dribbling         : {len(ev_drib):>4} olay")
    all_events.extend(ev_drib)

    ev_zone = detect_zone_entries(
        possession,
        calib_frames=calib_frames,
    )
    print(f"      Bölge girişi      : {len(ev_zone):>4} olay")
    all_events.extend(ev_zone)

    ev_pres = compute_pressure_events(
        possession,
        fps=args.fps,
        calib_frames=calib_frames if 'calib_frames' in dir() else None,
        window_sec=args.pressure_window_sec,
        dominance_threshold=args.pressure_threshold,
    )
    print(f"      Baskı istatistiği : {len(ev_pres):>4} olay")
    all_events.extend(ev_pres)

    ev_act = compute_player_activity_events(
        possession,
        period_sec=args.activity_period_sec,
        touch_threshold=args.activity_touches,
    )
    print(f"      Aktif oyuncu      : {len(ev_act):>4} olay")
    all_events.extend(ev_act)

    # Kalibrasyon olaylarını ortak formata dönüştürerek ekle
    if not args.no_calib_events:
        calib_normalized = [_normalize_calib_event(ce) for ce in calib_events]
        calib_normalized = [e for e in calib_normalized if e is not None]
        all_events.extend(calib_normalized)
        print(f"      Kalibrasyon olayı : {len(calib_normalized):>4} olay (eklendi)")

    # Zamana göre sırala
    all_events.sort(key=lambda e: float(e.get("timestamp_sec", 0.0)))

    print(f"      {'─' * 34}")
    print(f"      Toplam            : {len(all_events):>4} olay")

    # ------------------------------------------------------------------
    # [5/5] JSON çıktısı
    # ------------------------------------------------------------------
    print(f"[5/5] JSON yazılıyor: {args.output}")
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(all_events, f, ensure_ascii=False, indent=2)

    print(f"      Tamamlandı → {out_path.resolve()}")


if __name__ == "__main__":
    main()
