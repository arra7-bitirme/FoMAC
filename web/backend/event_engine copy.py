"""
event_engine.py
===============
Gerçek Zamanlı Yapay Zeka Futbol Spikeri — Kural Tabanlı Olay Motoru
=====================================================================

Mimari:
  Adım 1 → Koordinat Dönüşümü  : JSONL kalibrasyon verisiyle homografi hesapla,
                                   piksel koordinatını saha (dünya) koordinatına çevir.
  Adım 2 → Saha Bölgeleme       : Kuşbakışı koordinatını mantıksal bölgelere ayır.
  Adım 3 → Olay Tespiti         : Top sahipliği, depar, orta, pas, tehlikeli bölge.
  Adım 4 → Öncelik Sistemi      : Her olay 1-3 arası öncelik alır (düşük→kritik).
  Adım 5 → Debounce & Birleştirme: Aynı türdeki tekrar olayları bastır.
  Adım 6 → Jersey Hafızası      : Track ID bağımsız, konum tabanlı forma no. belleği.
  Adım 7 → Agresif Depar Filtresi: Sadece topa yakın oyuncuların deparlarını logla.

Bağımlılıklar:
  pip install pandas numpy opencv-python
"""

import json
import math
import os
import random
from collections import Counter
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import pandas as pd


# ── Öncelik Sabitleri ──────────────────────────────────────────────────────────
PRIORITY_LOW    = 1   # Sakin an: top serbest, pas yapılıyor
PRIORITY_MID    = 2   # Heyecanlı: depar, bölge geçişi, top kapma
PRIORITY_HIGH   = 3   # Kritik: ceza sahası, şut açısı, gol tehdidi

# Forma numarası geçersizlik eşiği (track sisteminin ürettiği yapay ID'ler bu aralıkta)
JERSEY_MAX_VALID = 99   # 0–99 arası gerçek forma numarası

# Debounce: aynı (event_type, track_id) çifti için minimum bekleme
DEBOUNCE_FRAMES: Dict[str, int] = {
    "possession_change": 15,   # top el değiştirme — hızlı paslaşmada spam'i önler
    "zone_change":       10,   # bölge değişimi
    "sprint":           150,   # depar — 3 saniye @ 50 fps (agresif bastırma)
    "loose_ball":        25,   # top serbest
    "danger":             5,   # tehlike — mümkün olduğunca sık bildir
    "pass":              10,   # pas tespiti
    "cross":             10,   # orta tespiti
}

# ── Depar Filtresi Sabitleri ─────────────────────────────────────────────────
# Topa en fazla bu mesafedeki oyuncunun deparı loglanır.
SPRINT_MAX_BALL_DIST_M  = 20.0   # metre; toptan uzak sahte depar'ları kes
# Aynı takımdan en az bu süre geçmeden tekrar depar logu yok.
SPRINT_TEAM_COOLDOWN_FRAMES = 150  # 3 saniye @ 50 fps

# Pas alıcısı sprint istisnası: bu pencerede pas gelmiş ve oyuncu topa bu kadar yakınsa
PASS_RECEIVER_DIST_M    = 8.0    # metre

# ── Takım Kimliği Kilitleme ──────────────────────────────────────────────
# Kaç ardışık pencerede aynı takım → kilitle  (15 frame/w × 50fps = 0.3 s/w → 3w = 0.9 s)
TEAM_LOCK_WINDOWS    = 3
# Kaç ardışık pencerede farklı takım → kilidi kaldır  (13w = ~3.9 s güvenlik payı)
TEAM_UNLOCK_WINDOWS  = 13

# ── Pas / Orta Sabitleri ──────────────────────────────────────────────────────
# Sahiplik değişiminde topun en az bu kadar hareket etmesi pas sayılması için gereklidir.
PASS_MIN_BALL_DIST_M = 1.5    # metre (kalibrasyon gürültüsünü geç)
# Bu uzaklığı aşan pas → "uzun pas" / havadan; altında kalan → "kısa / yerden"
LONG_PASS_DIST_M        = 20.0   # metre
# Kanat tanımı: y ekseninde bu değerin ü stünde olan x konumları kanat sayılır
WING_Y_THRESHOLD        = 18.0   # metre (saha ortasından yanına)


# ─── Sabit Değerler (FIFA Standart Saha: 105 m × 68 m) ───────────────────────

PITCH_LENGTH = 105.0     # metre ; x ekseni: [-52.5, +52.5]
PITCH_WIDTH  = 68.0      # metre ; y ekseni: [-34.0, +34.0]

HALF_LEN = PITCH_LENGTH / 2   # 52.5 m
HALF_WID = PITCH_WIDTH  / 2   # 34.0 m

# Ceza sahası sınırları (FIFA)
PENALTY_DEPTH  = 16.5    # gol çizgisinden içeriye
PENALTY_HALF_W = 20.16   # saha orta çizgisinden yana

# ── Eşik Değerleri ─────────────────────────────────────────────────────────────
POSSESSION_DIST_M      = 2.5    # metre → bu mesafe altındaysa oyuncu topu elinde tutar
SPRINT_THRESHOLD_MPS   = 7.0    # m/s   → bu hız üstündeyse depar sayılır
SPRINT_MAX_MPS         = 12.5   # m/s   → Bolt zirvesi ≈ 10.4 m/s; piksel gürültüsü kapatma
SPRINT_COOLDOWN_FRAMES = 30     # aynı oyuncu için depar bildiriminin tekrar süresi (frame)

# Ceza sahasına girişi "tehlike" olarak değerlendirmek için toplu olma zorunluluğu
DANGER_REQUIRES_BALL = True

# Kalibrasyon için minimum kontrol noktası sayısı
MIN_HOMOGRAPHY_POINTS  = 4

# ── CSV Sınıf Kimlikleri (cls_id) ──────────────────────────────────────────────
CLS_PLAYER  = 0
CLS_BALL    = 1
CLS_REFEREE = 2

# ── Takım Adları ───────────────────────────────────────────────────────────────
TEAM_NAMES: Dict[int, str] = {
    0:  "Takım A",
    1:  "Takım B",
    2:  "Hakem",
    -1: "Bilinmiyor",
}

# ── Takım Hücum Yönleri ────────────────────────────────────────────────────────
# +1 = sağa hücum (+X),  -1 = sola hücum (-X)
# Takım A sağ kaleye hücum eder; Takım B sol kaleye hücum eder.
TEAM_ATTACK_DIR: Dict[int, int] = {
    0: +1,   # Takım A → sağ (+X)
    1: -1,   # Takım B → sol (-X)
}


# ─── Olay Veri Modeli ─────────────────────────────────────────────────────────

@dataclass
class Event:
    """
    Tek bir olayı temsil eden zengin veri nesnesi.

    Alanlar:
        frame_id  : Olayın gerçekleştiği frame numarası.
        priority  : 1 (düşük) → 2 (orta) → 3 (yüksek/kritik).
        event_type: İç sınıflandırma etiketi (debounce anahtarı olarak da kullanılır).
        message   : LLM'e iletilecek Türkçe metin.
        track_id  : İlgili oyuncunun track ID'si (yoksa None).
        team_id   : İlgili takım (yoksa -1).
    """
    frame_id:   int
    priority:   int
    event_type: str
    message:    str
    track_id:   Optional[int] = None
    team_id:    int           = -1

    def __str__(self) -> str:
        """LLM'e gönderilmeye hazır tek satır çıktı."""
        return (
            f"[Öncelik: {self.priority}] "
            f"[Frame: {self.frame_id:>6d}] "
            f"{self.message}"
        )


# ─── Bölge Tanımı ─────────────────────────────────────────────────────────────

class PitchZone(Enum):
    """Sahayı mantıksal bloklara ayıran bölge tanımları."""
    UNKNOWN          = "Bilinmiyor"
    OWN_PENALTY      = "Kendi Ceza Sahası"
    OWN_HALF         = "Kendi Yarı Sahası"
    CENTER_CIRCLE    = "Orta Saha Dairesi"
    OPPONENT_HALF    = "Rakip Yarı Sahası"
    OPPONENT_PENALTY = "Rakip Ceza Sahası"


# ─── Veri Modelleri ────────────────────────────────────────────────────────────

@dataclass
class PlayerState:
    """Tek bir nesne (oyuncu / hakem) için anlık durum."""
    track_id:     int
    team_id:      int
    cls_id:       int
    pixel_bbox:   Tuple[int, int, int, int]       # x1, y1, x2, y2
    world_xy:     Optional[Tuple[float, float]]   # saha koordinatı (m)
    zone:         PitchZone   = PitchZone.UNKNOWN
    speed_mps:    float       = 0.0
    has_ball:     bool        = False
    is_sprinting: bool        = False
    jersey_number: Optional[int] = None          # OCR'dan gelen forma numarası (doğrulanmış)
    vx_mps:       float          = 0.0           # İşaretli X-hızı (m/s); + = sağ, − = sol


@dataclass
class FrameState:
    """Bir frame'in tam anlık tablosu."""
    frame_id:        int
    players:         Dict[int, PlayerState]       = field(default_factory=dict)
    ball_world_xy:   Optional[Tuple[float, float]] = None
    ball_pixel_bbox: Optional[Tuple]              = None
    possessor_id:    Optional[int]                = None
    possessor_team:  int                          = -1


# ══════════════════════════════════════════════════════════════════════════════
# Ana Sınıf: EventEngine
# ══════════════════════════════════════════════════════════════════════════════

class EventEngine:
    """
    Kural tabanlı futbol olay motoru.

    Kullanım:
        engine = EventEngine("tracking.csv", "calibration.jsonl")
        events = engine.run()           # tüm veriyi işle
        # —ya da—
        events = engine.process_frame(1540)   # tek frame
    """

    def __init__(
        self,
        tracking_csv_path: str,
        calibration_jsonl_path: str,
        fps: float = 50.0,
        window_size: int = 30,
        mapping_json_path: Optional[str] = None,
        event_log_path: str = "events_metadata.jsonl",
    ):
        self.fps = fps
        # Kaç frame = 1 pencere (tumbling window).  15 frame @ 50 fps = 0,3 sn.
        self.window_size = window_size

        # ── Veri Yükleme ──────────────────────────────────────────────────────
        print("[EventEngine] Takip verisi yükleniyor …")
        self.tracking_df = pd.read_csv(tracking_csv_path)
        self._preprocess_tracking()

        print("[EventEngine] Kalibrasyon verisi yükleniyor …")
        self.calib_index: Dict[int, dict] = self._load_calibration(calibration_jsonl_path)
        # Sıralı key listesi (binary search için)
        self._calib_keys: List[int] = sorted(self.calib_index.keys())

        # ── Homografi Önbelleği ──────────────────────────────────────────────
        self._H_cache: Dict[int, np.ndarray] = {}      # frame_id → 3×3 H
        self._last_valid_H: Optional[np.ndarray] = None

        # ── Durum Makinesi ───────────────────────────────────────────────────
        self.prev_state:  Optional[FrameState] = None
        self.event_log:   List[Event]          = []   # artık Event nesneleri tutuluyor

        # Sprint throttle: {track_id → kalan cooldown frame sayısı}
        self._sprint_cooldowns: Dict[int, int] = {}

        # ── Pencere (Tumbling Window) Tamponu ───────────────────────────────
        # Dolduğunda _flush_window() tetiklenir; süzülmüş olay üretilir.
        self._frame_buffer: List["FrameState"] = []

        # ── Debounce Tablosu ─────────────────────────────────────────────────
        # Anahtar: (event_type, track_id veya -1) → son tetiklenme frame'i
        # Aynı olay tipinin minimum DEBOUNCE_FRAMES'den önce tekrar üretilmesini engeller.
        self._last_event_frame: Dict[Tuple[str, int], int] = {}

        # ── Takım Bazı Depar Cooldown ───────────────────────────────────────────
        # Aynı takımdan kaç frame geçtikten sonra yeni depar logu yapılabilir.
        # Anahtar: team_id (0 ya da 1) → son depar log'unun frame'i
        self._team_sprint_last_frame: Dict[int, int] = {}

        # ── Jersey Hafızası (Konum Tabanlı, Track ID Bağımsız) ────────────────
        # Anahtar: (team_id, jersey_number) → en son görülen world_xy konumu
        # Bu yapı, track ID değişse bile forma no.'nu oyuncunun fiziksel
        # konumuna bağlayarak makinç yeniden tanımlamayı sağlar.
        self._jersey_memory: Dict[Tuple[int, int], Tuple[float, float]] = {}
        # Ters yön: track_id → doğrulanmış jersey_number (hızlı lookup)
        self._track_to_jersey: Dict[int, int] = {}

        # ── Top Durum Tarihi (pas/orta tespiti için) ──────────────────────
        # Önceki frame'deki top dünya koordinatı ve hangi takımda olduğu.
        self._prev_ball_xy: Optional[Tuple[float, float]] = None
        self._prev_ball_possessor_team: int = -1

        # ── Takım Kimliği Kilitleme ────────────────────────────────────────────
        # track_id → (en son tutarlı takım_id, tutarlı pencere sayısı)
        self._track_team_history: Dict[int, Tuple[int, int]] = {}
        # track_id → kilitlenmiş takım_id
        self._track_team_locked: Dict[int, int] = {}
        # track_id → kilit kırma için art arda farklı pencere sayısı
        self._track_team_unlock_counter: Dict[int, int] = {}

        # ── Dosya Kayıt Yolu ─────────────────────────────────────────────────
        self._event_log_path = event_log_path

        # ── Takım / Oyuncu Haritalama ─────────────────────────────────────────
        # Varsayılan: hakem ve bilinmiyor; takımlar JSON yüklenince dolar.
        self._team_names: Dict[int, str] = {2: "Hakem", -1: "Bilinmiyor"}
        self._player_names: Dict[int, Dict[int, str]] = {}   # team_id → {jersey_no → isim}
        if mapping_json_path:
            self._load_mapping(mapping_json_path)
        else:
            # JSON verilmezse Takım A / Takım B varsayılana dön
            self._team_names[0] = "Takım A"
            self._team_names[1] = "Takım B"

    # ══════════════════════════════════════════════════════════════════════════
    # ADIM 0 — Veri Ön İşleme
    # ══════════════════════════════════════════════════════════════════════════

    def _preprocess_tracking(self) -> None:
        """
        CSV'yi temizle:
          • Bounding box merkez ve ayak noktasını hesapla.
          • Ardışık frame'ler arası piksel farkından hız tahmini yap.
        """
        df = self.tracking_df

        # Bounding box merkezi
        df["cx"] = (df["x1"] + df["x2"]) / 2.0
        df["cy"] = (df["y1"] + df["y2"]) / 2.0

        # Ayak noktası (bbox alt-orta): homografi dönüşümü için daha isabetli nokta
        df["foot_x"] = df["cx"]
        df["foot_y"] = df["y2"].astype(float)

        # Hız: ardışık frame'lerdeki merkez piksel farkı / kare (frame)
        df.sort_values(["track_id", "frame_id"], inplace=True, ignore_index=True)
        df["dx_px_per_frame"] = df.groupby("track_id")["cx"].diff().fillna(0.0)
        df["_dy"] = df.groupby("track_id")["cy"].diff().fillna(0.0)
        df["speed_px_per_frame"] = np.hypot(df["dx_px_per_frame"], df["_dy"])
        df.drop(columns=["_dy"], inplace=True)

        # Relink olan frame'lerde hız geçersizdir: trackID aniden atladı.
        # O frame için hızı sıfırla; böylece sahte depar bildirimi oluşmaz.
        if "relinked" in df.columns:
            df.loc[df["relinked"] == 1, "speed_px_per_frame"] = 0.0
            df.loc[df["relinked"] == 1, "dx_px_per_frame"] = 0.0

        # jersey_number sütunu yoksa boş oluştur
        if "jersey_number" not in df.columns:
            df["jersey_number"] = None

        self.tracking_df = df

    def _load_calibration(self, path: str) -> Dict[int, dict]:
        """
        JSONL kalibrasyon dosyasını satır satır oku.
        calibration_ok == True olan satırları {frame_idx: kayıt} sözlüğüne ekle.
        """
        index: Dict[int, dict] = {}
        with open(path, "r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                record = json.loads(line)
                if record.get("calibration_ok", False):
                    index[record["frame_idx"]] = record
        print(f"[EventEngine] {len(index)} kalibre edilmiş frame yüklendi.")
        return index

    def _load_mapping(self, path: str) -> None:
        """
        Takım/oyuncu haritalama JSON dosyasını yükle.

        Beklenen format (galjuv_mapping.json):
            {
              "rosters": {
                "galatasaray": [{"name": "Osimhen", "number": 45}, ...],
                "juventus":    [{"name": "Conceicao", "number": 7}, ...]
              }
            }
        Rosters'ın ilk anahtarı → team_id=0, ikincisi → team_id=1.
        """
        with open(path, "r", encoding="utf-8") as fh:
            data = json.load(fh)

        rosters = data.get("rosters", {})
        for idx, (key, players) in enumerate(rosters.items()):
            if idx >= 2:
                break
            # Anahtar (ör. "galatasaray") → gösterim ismine dönüştür
            display_name = key.title()   # "galatasaray" → "Galatasaray"
            self._team_names[idx] = display_name
            # Forma numarası → oyuncu adı sözlüğü
            self._player_names[idx] = {
                int(p["number"]): p["name"] for p in players
            }

        team0 = self._team_names.get(0, "?")
        team1 = self._team_names.get(1, "?")
        print(f"[EventEngine] Kadro yüklendi: {team0} (Takım A) vs {team1} (Takım B)")

    # ══════════════════════════════════════════════════════════════════════════
    # ADIM 1 — Koordinat Dönüşümü (Piksel → Saha Düzlemi)
    # ══════════════════════════════════════════════════════════════════════════

    def _compute_homography_for_frame(self, frame_id: int) -> Optional[np.ndarray]:
        """
        Bir frame için piksel→saha homografi matrisini hesapla.

        Yöntem:
        -------
        JSONL'de her oyuncu için hem piksel bounding box (bbox_xyxy) hem de
        saha koordinatı (world_xy) mevcuttur.
        Oyuncunun ayak noktası (bbox alt-ortası) ile world_xy çiftlerini
        kontrol noktası olarak kullanıp cv2.findHomography çağrısıyla 3×3 H
        matrisi elde edilir.

        Önbellek (cache):
        -----------------
        Hesaplanan matris self._H_cache[frame_id]'e yazılır.
        Geçerli bir H bulunamazsa en son geçerli H döner.
        """
        # Önbelleği kontrol et
        if frame_id in self._H_cache:
            return self._H_cache[frame_id]

        # En yakın kalibre edilmiş frame'i bul
        calib = self._nearest_calibration(frame_id)
        if calib is None:
            return self._last_valid_H

        pixel_pts: List[List[float]] = []
        world_pts:  List[List[float]] = []

        # Her oyuncu için kontrol noktası çifti oluştur
        for player in calib["data"].get("players", []):
            bbox = player["bbox_xyxy"]
            wx, wy = player["world_xy"]

            foot_x = (bbox[0] + bbox[2]) / 2.0   # bbox alt-orta: x
            foot_y = float(bbox[3])               # bbox alt-orta: y

            pixel_pts.append([foot_x, foot_y])
            world_pts.append([wx, wy])

        # Top merkezi de ek nokta olarak eklenebilir
        ball = calib["data"].get("ball")
        if ball:
            bbox = ball["bbox_xyxy"]
            bx = (bbox[0] + bbox[2]) / 2.0
            by = (bbox[1] + bbox[3]) / 2.0
            pixel_pts.append([bx, by])
            world_pts.append(list(ball["world_xy"]))

        if len(pixel_pts) < MIN_HOMOGRAPHY_POINTS:
            # Yetersiz kontrol noktası → önceki geçerli H kullan
            return self._last_valid_H

        src = np.array(pixel_pts, dtype=np.float32)
        dst = np.array(world_pts,  dtype=np.float32)

        # RANSAC ile sağlam homografi tahmini
        H, mask = cv2.findHomography(src, dst, cv2.RANSAC, ransacReprojThreshold=1.0)

        if H is not None:
            inlier_count = int(mask.sum()) if mask is not None else 0
            if inlier_count >= MIN_HOMOGRAPHY_POINTS:
                self._H_cache[frame_id] = H
                self._last_valid_H = H

        return H

    def pixel_to_world(
        self,
        pixel_x: float,
        pixel_y: float,
        H: np.ndarray,
    ) -> Tuple[float, float]:
        """
        Tek bir piksel noktasını saha koordinatına dönüştür.

        Formül:  [wx', wy', w]^T = H × [px, py, 1]^T
                 wx = wx'/w,  wy = wy'/w
        OpenCV'deki cv2.perspectiveTransform bunu tek satırda yapar.
        """
        pt = np.array([[[pixel_x, pixel_y]]], dtype=np.float32)  # (1,1,2)
        result = cv2.perspectiveTransform(pt, H)                  # (1,1,2)
        wx = float(result[0, 0, 0])
        wy = float(result[0, 0, 1])
        return wx, wy

    def _dynamic_px_to_m_scale(self, H: np.ndarray, cx: float = 640.0, cy: float = 360.0) -> float:
        """
        Görüntünün merkezinde 1 piksellik yatay hareketin sahada kaç metre
        olduğunu hesapla. Piksel hızından m/s'ye dönüşüm için kullanılır.
        """
        p1 = self.pixel_to_world(cx,       cy, H)
        p2 = self.pixel_to_world(cx + 1.0, cy, H)
        scale = math.hypot(p2[0] - p1[0], p2[1] - p1[1])
        return max(scale, 1e-4)   # sıfır bölmekten korun

    # ══════════════════════════════════════════════════════════════════════════
    # ADIM 2 — Saha Bölgeleme
    # ══════════════════════════════════════════════════════════════════════════

    @staticmethod
    def get_pitch_zone(world_xy: Tuple[float, float]) -> PitchZone:
        """
        Saha koordinatına (wx, wy) göre oyuncunun bulunduğu bölgeyi döner.

        Koordinat sistemi (FIFA standardı, metre, orijin = saha merkezi):
          x ekseni : −52.5 (sol gol çizgisi) → +52.5 (sağ gol çizgisi)
          y ekseni : −34.0 (alt yan çizgi)   → +34.0 (üst yan çizgi)

        Varsayım: sol (x<0) = kendi yarı sahası, sağ (x>0) = rakip yarı sahası.
        Pratikte bunu takım yönüne göre çevirebilirsiniz.
        """
        wx, wy = world_xy

        # Saha sınırı dışı kontrol (2 m tolerans)
        if abs(wx) > HALF_LEN + 2.0 or abs(wy) > HALF_WID + 2.0:
            return PitchZone.UNKNOWN

        left_box_x  = -(HALF_LEN - PENALTY_DEPTH)   # ≈ −36.0
        right_box_x =  (HALF_LEN - PENALTY_DEPTH)   # ≈ +36.0

        # Kendi ceza sahası (sol)
        if wx < left_box_x and abs(wy) < PENALTY_HALF_W:
            return PitchZone.OWN_PENALTY

        # Rakip ceza sahası (sağ)
        if wx > right_box_x and abs(wy) < PENALTY_HALF_W:
            return PitchZone.OPPONENT_PENALTY

        # Orta daire (yarıçap ≈ 9.15 m)
        if abs(wx) < 9.15 and abs(wy) < 9.15:
            return PitchZone.CENTER_CIRCLE

        # Yarı sahalar
        if wx < 0:
            return PitchZone.OWN_HALF
        return PitchZone.OPPONENT_HALF

    @staticmethod
    def get_zone_for_team(world_xy: Tuple[float, float], team_id: int) -> PitchZone:
        """
        Takım yönüne duyarlı bölge tespiti.

        Takım A (+X'e hücum eder):  sol yarı = kendi, sağ yarı = rakip.
        Takım B (−X'e hücum eder):  sağ yarı = kendi, sol yarı = rakip.
        Hakem / bilinmeyen: statik get_pitch_zone davranışına düşer (Team A gibi).
        """
        wx, wy = world_xy

        if abs(wx) > HALF_LEN + 2.0 or abs(wy) > HALF_WID + 2.0:
            return PitchZone.UNKNOWN

        left_box_x  = -(HALF_LEN - PENALTY_DEPTH)   # ≈ −36.0
        right_box_x =  (HALF_LEN - PENALTY_DEPTH)   # ≈ +36.0

        # Orta daire her takım için aynı
        if abs(wx) < 9.15 and abs(wy) < 9.15:
            return PitchZone.CENTER_CIRCLE

        attack_dir = TEAM_ATTACK_DIR.get(team_id, +1)

        if attack_dir == +1:
            # Takım A: sol = kendi, sağ = rakip
            if wx < left_box_x and abs(wy) < PENALTY_HALF_W:
                return PitchZone.OWN_PENALTY
            if wx > right_box_x and abs(wy) < PENALTY_HALF_W:
                return PitchZone.OPPONENT_PENALTY
            return PitchZone.OWN_HALF if wx < 0 else PitchZone.OPPONENT_HALF
        else:
            # Takım B: sağ = kendi, sol = rakip
            if wx > right_box_x and abs(wy) < PENALTY_HALF_W:
                return PitchZone.OWN_PENALTY
            if wx < left_box_x and abs(wy) < PENALTY_HALF_W:
                return PitchZone.OPPONENT_PENALTY
            return PitchZone.OWN_HALF if wx > 0 else PitchZone.OPPONENT_HALF

    # ══════════════════════════════════════════════════════════════════════════
    # ADIM 3 — Olay Tespiti (Heuristikler)
    # ══════════════════════════════════════════════════════════════════════════

    def detect_possession(
        self,
        players: Dict[int, PlayerState],
        ball_world_xy: Optional[Tuple[float, float]],
    ) -> Optional[int]:
        """
        Top sahipliğini Öklid mesafesiyle tespit et.

        Topun saha koordinatına en yakın oyuncu, mesafesi POSSESSION_DIST_M'den
        azsa topu elinde tutar. Hiçbiri bu eşiğin altında değilse None döner.

        Returns: Topu elinde tutan oyuncunun track_id'si veya None.
        """
        if ball_world_xy is None:
            return None

        best_id:   Optional[int] = None
        best_dist: float         = POSSESSION_DIST_M  # eşik başlangıç değeri

        for tid, ps in players.items():
            # Hakem ve bilinmeyen nesneler sahiplik hesabına dahil edilmez
            if ps.cls_id == CLS_REFEREE or ps.team_id not in (0, 1) or ps.world_xy is None:
                continue
            dist = math.hypot(
                ps.world_xy[0] - ball_world_xy[0],
                ps.world_xy[1] - ball_world_xy[1],
            )
            if dist < best_dist:
                best_dist = dist
                best_id   = tid

        return best_id

    def detect_sprint(self, track_id: int, speed_mps: float) -> bool:
        """
        Oyuncunun depar atıp atmadığını tespit et.

        Cooldown mekanizması:
          Bir oyuncu için depar tespit edilince SPRINT_COOLDOWN_FRAMES frame
          boyunca aynı oyuncu için tekrar bildirim üretilmez. Bu sayede
          LLM'e art arda aynı mesaj gönderilmez.

        Returns: True → depar atıyor (ve bildirim üretilmeli).
        """
        remaining = self._sprint_cooldowns.get(track_id, 0)
        if remaining > 0:
            self._sprint_cooldowns[track_id] = remaining - 1
            return False   # cooldown devam ediyor, sessiz kal

        # Gerçekçi olmayan hızları (relink gürültüsü) filtrele
        if speed_mps >= SPRINT_THRESHOLD_MPS and speed_mps <= SPRINT_MAX_MPS:
            self._sprint_cooldowns[track_id] = SPRINT_COOLDOWN_FRAMES
            return True

        return False

    # ══════════════════════════════════════════════════════════════════════════
    # ADIM 4 — Durum Makinesi ve Olay Üretimi
    # ══════════════════════════════════════════════════════════════════════════

    def _build_frame_state(self, frame_id: int) -> FrameState:
        """
        CSV + JSONL verilerini birleştirerek o frame'in tam FrameState'ini oluştur.

        Öncelik sırası (world_xy için):
          1. JSONL'deki doğrudan world_xy (kalibrasyon doğruланmış)
          2. Homografi ile piksel→saha dönüşümü
          3. Veri yoksa None
        """
        state = FrameState(frame_id=frame_id)

        # Homografi matrisini al (ya da önbellekten getir)
        H = self._compute_homography_for_frame(frame_id)

        # Dinamik px→m ölçeği (hız dönüşümü için)
        scale_px_to_m = self._dynamic_px_to_m_scale(H) if H is not None else 0.07

        # ── JSONL'den world_xy sözlüğü ──────────────────────────────────────
        calib_world: Dict[int, Tuple[float, float]] = {}
        ball_calib_xy: Optional[Tuple[float, float]] = None

        calib = self._nearest_calibration(frame_id)
        if calib:
            for p in calib["data"].get("players", []):
                calib_world[p["track_id"]] = tuple(p["world_xy"])
            ball_data = calib["data"].get("ball")
            if ball_data:
                ball_calib_xy = tuple(ball_data["world_xy"])

        # ── Bu frame'e ait CSV satırları ────────────────────────────────────
        frame_rows = self.tracking_df[self.tracking_df["frame_id"] == frame_id]
        if frame_rows.empty:
            return state

        # ── Top konumunu belirle ─────────────────────────────────────────────
        ball_rows = frame_rows[frame_rows["cls_id"] == CLS_BALL]
        if not ball_rows.empty:
            br = ball_rows.iloc[0]
            state.ball_pixel_bbox = (int(br["x1"]), int(br["y1"]),
                                     int(br["x2"]), int(br["y2"]))
            if H is not None:
                bx = float((br["x1"] + br["x2"]) / 2)
                by = float((br["y1"] + br["y2"]) / 2)
                state.ball_world_xy = self.pixel_to_world(bx, by, H)
            elif ball_calib_xy:
                state.ball_world_xy = ball_calib_xy
        elif ball_calib_xy:
            # CSV'de top yok ama JSONL'de var → kalibrasyon değerini kullan
            state.ball_world_xy = ball_calib_xy

        # ── Oyuncuları ve hakemi işle ────────────────────────────────────────
        other_rows = frame_rows[frame_rows["cls_id"].isin([CLS_PLAYER, CLS_REFEREE])]

        for _, row in other_rows.iterrows():
            tid    = int(row["track_id"])
            team   = int(row["team_id"])
            cls    = int(row["cls_id"])
            spf    = float(row.get("speed_px_per_frame", 0.0))

            # World koordinatı: 1) JSONL  2) Homografi  3) None
            if tid in calib_world:
                wxy = calib_world[tid]
            elif H is not None:
                foot_x = float((row["x1"] + row["x2"]) / 2)
                foot_y = float(row["y2"])
                wxy = self.pixel_to_world(foot_x, foot_y, H)
            else:
                wxy = None

            # m/s hız tahmini
            speed_mps = spf * scale_px_to_m * self.fps
            dx_spf    = float(row.get("dx_px_per_frame", 0.0))
            vx_mps    = dx_spf * scale_px_to_m * self.fps   # işaretli X hızı

            # ── Forma Numarası Fallback Mantığı ─────────────────────────────
            # CSV'deki jersey_number: -1, NaN, boş veya 1000+ ise geçersiz say.
            raw_jersey = row.get("jersey_number", None)
            jersey_num: Optional[int] = None
            try:
                jn = int(float(raw_jersey))   # float arası gelebilir (NaN için except)
                if 0 <= jn <= JERSEY_MAX_VALID:
                    jersey_num = jn
                # Negatif (-1) veya aşırı büyük (900000001) → None → fallback
            except (TypeError, ValueError):
                jersey_num = None

            ps = PlayerState(
                track_id      = tid,
                team_id       = team,
                cls_id        = cls,
                pixel_bbox    = (int(row["x1"]), int(row["y1"]),
                                 int(row["x2"]), int(row["y2"])),
                world_xy      = wxy,
                zone          = self.get_zone_for_team(wxy, team) if wxy else PitchZone.UNKNOWN,
                speed_mps     = speed_mps,
                vx_mps        = vx_mps,
                is_sprinting  = SPRINT_THRESHOLD_MPS <= speed_mps <= SPRINT_MAX_MPS,
                jersey_number = jersey_num,
            )
            state.players[tid] = ps

        # ── Top sahipliği ────────────────────────────────────────────────────
        poss_id = self.detect_possession(state.players, state.ball_world_xy)
        state.possessor_id = poss_id
        if poss_id and poss_id in state.players:
            state.players[poss_id].has_ball = True
            state.possessor_team = state.players[poss_id].team_id
        # ── Jersey hafızasını gücelle (tüm oyuncular için) ─────────────────
        for ps in state.players.values():
            self._update_jersey_memory(ps)
        return state

    # ══════════════════════════════════════════════════════════════════════════
    # Yardımcı: Forma Numarası Etiket Üretimi (Hafızalı)
    # ══════════════════════════════════════════════════════════════════════════

    def _update_jersey_memory(self, ps: PlayerState) -> None:
        """
        Jersey hafızasını gücelle.

        1. Oyuncunun bu frame'de geçerli bir forma numarası varsa:
           a) (team_id, jersey_no) → world_xy eşleştirmesini kaydet.
           b) track_id → jersey_no ters eşleşmesini kaydet.
        2. Forma numarası yoksa ama world_xy biliniyorsa:
           Hafızadaki tüm noktaları tara; aynı takımdan en yakın kayda
           bak. 3 metre içindeyse o forma numarasını bu oyuncuya ata.
        """
        MEMORY_MATCH_DIST_M = 3.0   # metre; konum eşleştirme toleransı

        if ps.world_xy is None or ps.team_id not in (0, 1):
            return

        if ps.jersey_number is not None:
            # Bilinen forma numarasını konuma bağla
            key = (ps.team_id, ps.jersey_number)
            self._jersey_memory[key] = ps.world_xy
            self._track_to_jersey[ps.track_id] = ps.jersey_number
            return

        # Forma numarası bilinmiyor → konumla eşleştirmeye çalış
        best_no:   Optional[int]   = None
        best_dist: float           = MEMORY_MATCH_DIST_M

        for (team, jersey_no), mem_xy in self._jersey_memory.items():
            if team != ps.team_id:
                continue
            dist = math.hypot(
                ps.world_xy[0] - mem_xy[0],
                ps.world_xy[1] - mem_xy[1],
            )
            if dist < best_dist:
                best_dist = dist
                best_no   = jersey_no

        if best_no is not None:
            # Hafızadan forma numarasını kopyala ve konumu güncelle
            ps.jersey_number = best_no
            self._track_to_jersey[ps.track_id] = best_no
            self._jersey_memory[(ps.team_id, best_no)] = ps.world_xy

    def _player_label(self, ps: "PlayerState") -> str:
        """
        Forma numarası fallback mantığı (hafıza destekli, gerçek isim destekli).

        Önce hafızadaki ters eşleşmeye bak; bulursa o numarayı kullan.
        Numara varsa JSON haritalamada oyuncu ismini ara.
        Her iki kaynakta da bilgi yoksa takım + "takımından bir oyuncu" yaz.
        """
        team_name = self._team_names.get(ps.team_id, f"Takım {ps.team_id}")
        # Önce ters eşleşme (makinenin bu frame'den önce öğrenmiş olduğu)
        jno = self._track_to_jersey.get(ps.track_id, ps.jersey_number)
        if jno is not None:
            player_name = self._player_names.get(ps.team_id, {}).get(jno)
            if player_name:
                return f"{team_name}'dan {player_name}"
            return f"{team_name} #{jno}"
        return f"{team_name} takımından bir oyuncu"

    # ══════════════════════════════════════════════════════════════════════════
    # Yardımcı: Hız → Betimsel Metin
    # ══════════════════════════════════════════════════════════════════════════

    @staticmethod
    def _speed_description(speed_mps: float) -> str:
        """
        Sayısal hız değerini (m/s) LLM için betimsel Türkçe eyleme çevirir.
        Sayısal değer çıktıda yer almaz.
        """
        if speed_mps < 4.0:
            return random.choice([
                "yavaş adımlarla ilerliyor",
                "sakin bir şekilde top sürüyor",
            ])
        elif speed_mps <= 8.0:
            return random.choice([
                "hızlıca hareketleniyor",
                "topla hızlandı",
            ])
        else:
            return random.choice([
                "müthiş bir depar atıyor",
                "adeta rüzgar gibi esiyor",
                "fırtına gibi ceza sahasına dalıyor",
            ])

    # ══════════════════════════════════════════════════════════════════════════
    # Yardımcı: Debounce Kontrolü
    # ══════════════════════════════════════════════════════════════════════════

    def _debounce_ok(self, event_type: str, track_id: int, frame_id: int) -> bool:
        """
        Verilen (event_type, track_id) çifti için debounce kontrolü yapar.

        Returns:
            True  → olay üretilebilir (cooldown bitti ya da hiç başlamadı).
            False → çok erken, aynı olayı bastır.
        """
        key = (event_type, track_id)
        min_gap = DEBOUNCE_FRAMES.get(event_type, 15)
        last = self._last_event_frame.get(key, -999)
        if (frame_id - last) < min_gap:
            return False
        self._last_event_frame[key] = frame_id
        return True

    def _team_sprint_ok(self, team_id: int, frame_id: int) -> bool:
        """
        Aynı takımdan kaç frame geçtiğine bak; yeterince süre geçtiyse True dön.
        Takım bazında depar spam'ını tamamen kesmek için kullanılır.
        """
        last = self._team_sprint_last_frame.get(team_id, -9999)
        if (frame_id - last) < SPRINT_TEAM_COOLDOWN_FRAMES:
            return False
        self._team_sprint_last_frame[team_id] = frame_id
        return True

    def _resolve_team_id(self, track_id: int, voted_team: int) -> int:
        """
        Takım Kimliği Kilitleme (Team Identity Locking).

        Algoritma:
        ─────────
        • Bir track_id için arka arkaya TEAM_LOCK_WINDOWS pencerede
          aynı takım görünürse o takım "kilitlenir".
        • Kilit sonrası anlık farklı-takım algılamalarını reddet (flicker).
        • Ancak TEAM_UNLOCK_WINDOWS pencere boyunca kesintisiz farklı takım
          görünüyorsa kilit güncellenir (~3.9 saniyelik güvenlik payı).

        Args:
            track_id   : Oyuncunun track ID'si.
            voted_team : Bu pencere için majority vote ile belirlenen takım.
        Returns:
            Gerçek (ve gerekirse kilitli/düzeltilmiş) takım ID'si.
        """
        locked = self._track_team_locked.get(track_id)
        hist_team, hist_count = self._track_team_history.get(track_id, (voted_team, 0))

        if locked is not None:
            if voted_team == locked:
                # Kilitli takımla uyumlu → kilit sağlam, unlock sayacını sıfırla
                self._track_team_unlock_counter[track_id] = 0
                self._track_team_history[track_id] = (locked, hist_count + 1)
                return locked
            else:
                # Kilitli takımla uyumsuz → unlock sayacını artır
                uc = self._track_team_unlock_counter.get(track_id, 0) + 1
                self._track_team_unlock_counter[track_id] = uc
                if uc >= TEAM_UNLOCK_WINDOWS:
                    # Yeterince uzun süre farklı takım → kilidi güncelle
                    self._track_team_locked[track_id] = voted_team
                    self._track_team_history[track_id] = (voted_team, 1)
                    self._track_team_unlock_counter[track_id] = 0
                    return voted_team
                # Anlık flicker → kilitli takımı koru
                return locked
        else:
            # Henüz kilitlenmemiş → pencere sayacını ilerlet
            if hist_team == voted_team:
                new_count = hist_count + 1
                self._track_team_history[track_id] = (voted_team, new_count)
                if new_count >= TEAM_LOCK_WINDOWS:
                    self._track_team_locked[track_id] = voted_team
                    self._track_team_unlock_counter[track_id] = 0
            else:
                # Farklı takım → tarihi sıfırla, kilitlemek için yeniden say
                self._track_team_history[track_id] = (voted_team, 1)
            return voted_team

    # ══════════════════════════════════════════════════════════════════════════
    # Yardımcı: Pas / Orta Tespiti
    # ══════════════════════════════════════════════════════════════════════════

    def _detect_pass_or_cross(
        self,
        frame_id:         int,
        curr:             "FrameState",
        prev_possessor_ps: Optional["PlayerState"],
    ) -> Optional[Event]:
        """
        Pas ya da Orta (cross) olayı üret.

        Tetiklenme koşulları:
        ─────────────────────
        1. Önceki frame'de bir oyuncu topun sahibiydi.
        2. Bu frame'de top o oyuncunun elinden çıkmış (poss değişti ya da lost).
        3. Topun hareketi (prev_ball_xy → curr_ball_xy) PASS_MIN_BALL_SPEED_MPS'den
           daha hızlı — bu sayede duragan top değişimlerini geçeriz.

        Top mesafesi + hedef bölgesi:
        ──────────────────────────────
        • Orta (cross) → pasın Y-koordinatı WING_Y_THRESHOLD üstteyse (kanattan)
                          VE heden bölge OPPONENT_PENALTY ise.
          (Kendi ceza sahasına kesen orta hiçbir zaman orta olarak loglanmaz.)
        • Uzun pas → mesafe LONG_PASS_DIST_M üstteyse
        • Kısa pas → géri kalan tüm durumlar
        """
        if prev_possessor_ps is None:
            return None
        if self._prev_ball_xy is None or curr.ball_world_xy is None:
            return None

        # Topun hareketi
        bx0, by0 = self._prev_ball_xy
        bx1, by1 = curr.ball_world_xy
        ball_dist = math.hypot(bx1 - bx0, by1 - by0)   # metre

        if ball_dist < PASS_MIN_BALL_DIST_M:
            return None   # çok küçük hareket → kalibrasyon gürültüsü, pas değil

        label    = self._player_label(prev_possessor_ps)
        from_xy  = prev_possessor_ps.world_xy or (bx0, by0)
        from_zone = prev_possessor_ps.zone

        # Hedef bölge: top şu an nerede? (possessorun takımına göre)
        to_zone = self.get_zone_for_team(curr.ball_world_xy, prev_possessor_ps.team_id)

        # ── Orta (Cross) tespiti ──────────────────────────────────────────
        # KESİN KOŞUL: kanat pozisyonu + hedef OPPONENT_PENALTY
        is_wing = abs(from_xy[1]) >= WING_Y_THRESHOLD
        is_cross = (
            is_wing
            and to_zone == PitchZone.OPPONENT_PENALTY
            and from_zone != PitchZone.OWN_PENALTY   # kendi cezasından atılan orta olamaz
        )

        if is_cross and self._debounce_ok("cross", prev_possessor_ps.track_id, frame_id):
            wing_side = "sol" if from_xy[1] > 0 else "sağ"
            pass_type = "havadan" if ball_dist > LONG_PASS_DIST_M else "tehlikeli"
            msg = (
                f"{label} {wing_side} kanattan rakip ceza sahasına doğru "
                f"{pass_type} bir orta kesti!"
            )
            return Event(
                frame_id=frame_id, priority=PRIORITY_HIGH,
                event_type="cross", message=msg,
                track_id=prev_possessor_ps.track_id,
                team_id=prev_possessor_ps.team_id,
            )

        # ── Pas tespiti ──────────────────────────────────────────────────
        if not self._debounce_ok("pass", prev_possessor_ps.track_id, frame_id):
            return None

        dist_m_int = int(round(ball_dist))
        if ball_dist >= LONG_PASS_DIST_M:
            pass_desc = f"{dist_m_int} metrelik uzun bir pas"
            priority  = PRIORITY_MID
        else:
            pass_desc = f"{dist_m_int} metrelik kısa bir pas"
            priority  = PRIORITY_LOW

        msg = f"{label}'dan {pass_desc}."
        return Event(
            frame_id=frame_id, priority=priority,
            event_type="pass", message=msg,
            track_id=prev_possessor_ps.track_id,
            team_id=prev_possessor_ps.team_id,
        )
    # ══════════════════════════════════════════════════════════════════════════

    def _generate_events(
        self,
        frame_id: int,
        curr: FrameState,
        prev: Optional[FrameState],
    ) -> List[Event]:
        """
        Durum makinesi: mevcut (curr) ile önceki (prev) FrameState'i karşılaştır.

        Üretilen olaylar ve öncelikleri
        ────────────────────────────────
        P1 (Düşük)  : Top serbest, kısa pas, bölge geçişi (kendi sahasında)
        P2 (Orta)   : Depar (topa yakın), uzun pas, rakip yarısına geçiş
        P3 (Kritik) : Orta, rakip ceza sahasına giriş, gol tehdidi
        """
        events: List[Event] = []

        prev_poss = prev.possessor_id if prev else None
        curr_poss = curr.possessor_id
        possession_changed = (curr_poss != prev_poss)

        # Önceki possessörün PlayerState'ini çıkar (pas/orta tespiti için)
        prev_possessor_ps: Optional[PlayerState] = None
        if prev_poss is not None:
            # Önce curr'de ara (top kaybedildiyse ama oyuncu hala alanda)
            prev_possessor_ps = curr.players.get(prev_poss) or \
                                (prev.players.get(prev_poss) if prev else None)

        # ── 0. Pas / Orta Tespiti ────────────────────────────────────────
        # Sahiplik değişti ve önceki sahiple ilgili bilgi varsa pas/orta bak.
        pass_event: Optional[Event] = None   # bu pencerede tespit edilen pas/orta
        if possession_changed and prev_poss is not None:
            pass_event = self._detect_pass_or_cross(frame_id, curr, prev_possessor_ps)
            if pass_event:
                events.append(pass_event)

        # ── 1. Top El Değiştirme ─────────────────────────────────────────────
        if possession_changed:
            if curr_poss is not None and curr_poss in curr.players:
                ps        = curr.players[curr_poss]
                label     = self._player_label(ps)
                zone_name = ps.zone.value

                # Topu alan oyuncu aynı anda depar atıyor mu? → Olayları BİRLEŞTİR
                if ps.is_sprinting and self._debounce_ok("possession_change", curr_poss, frame_id):
                    self._debounce_ok("sprint", curr_poss, frame_id)
                    priority = PRIORITY_MID
                    if ps.zone == PitchZone.OPPONENT_PENALTY:
                        priority = PRIORITY_HIGH
                    msg = (
                        f"{label} topu kapıp hemen atağa kalktı — "
                        f"{self._speed_description(ps.speed_mps)}! {zone_name}."
                    )
                    events.append(Event(
                        frame_id=frame_id, priority=priority,
                        event_type="possession_change", message=msg,
                        track_id=curr_poss, team_id=ps.team_id,
                    ))

                elif self._debounce_ok("possession_change", curr_poss, frame_id):
                    if ps.zone == PitchZone.OPPONENT_PENALTY:
                        priority = PRIORITY_HIGH
                    elif ps.zone in (PitchZone.OPPONENT_HALF, PitchZone.CENTER_CIRCLE):
                        priority = PRIORITY_MID
                    else:
                        priority = PRIORITY_LOW
                    msg = (
                        f"Top {label}'da. "
                        f"{zone_name} bölgesinde pas açısı arıyor."
                    )
                    events.append(Event(
                        frame_id=frame_id, priority=priority,
                        event_type="possession_change", message=msg,
                        track_id=curr_poss, team_id=ps.team_id,
                    ))

            elif curr_poss is None and prev_poss is not None:
                if self._debounce_ok("loose_ball", -1, frame_id):
                    events.append(Event(
                        frame_id=frame_id, priority=PRIORITY_LOW,
                        event_type="loose_ball",
                        message="Top sahipsiz kaldı, iki takım da topu kapmaya çalışıyor.",
                        track_id=None, team_id=-1,
                    ))

        # ── 2. Toplu Oyuncunun Bölge Değişimi ───────────────────────────────
        if curr_poss and curr_poss in curr.players and not possession_changed:
            curr_zone = curr.players[curr_poss].zone

            if prev and prev_poss == curr_poss and curr_poss in prev.players:
                prev_zone = prev.players[curr_poss].zone

                if curr_zone != prev_zone:
                    ps    = curr.players[curr_poss]
                    label = self._player_label(ps)

                    if self._debounce_ok("zone_change", curr_poss, frame_id):
                        if curr_zone == PitchZone.OPPONENT_PENALTY:
                            priority = PRIORITY_HIGH
                        elif curr_zone in (PitchZone.OPPONENT_HALF, PitchZone.CENTER_CIRCLE):
                            priority = PRIORITY_MID
                        else:
                            priority = PRIORITY_LOW

                        msg = (
                            f"{label} topla '{curr_zone.value}' bölgesine ilerledi — "
                            f"{self._speed_description(ps.speed_mps)}."
                        )
                        events.append(Event(
                            frame_id=frame_id, priority=priority,
                            event_type="zone_change", message=msg,
                            track_id=curr_poss, team_id=ps.team_id,
                        ))

                    # ── 3. Rakip Ceza Sahasına Giriş — Tehlike Uyarısı ──────
                    if curr_zone == PitchZone.OPPONENT_PENALTY:
                        if (not DANGER_REQUIRES_BALL or ps.has_ball) and \
                           self._debounce_ok("danger", curr_poss, frame_id):
                            msg = (
                                f"{label} ceza sahasına "
                                f"{self._speed_description(ps.speed_mps)}! Şut açısı arıyor!"
                            )
                            events.append(Event(
                                frame_id=frame_id, priority=PRIORITY_HIGH,
                                event_type="danger", message=msg,
                                track_id=curr_poss, team_id=ps.team_id,
                            ))

        # ── 4. Agresif Depar Filtresi ────────────────────────────────────────
        #
        # Kural:
        #   a) Oyuncu topa SAHİP ise: her zaman loglanabilir (hücum deparsı).
        #   b) Oyuncu topa sahip değilse:
        #      • Topa mesafesi SPRINT_MAX_BALL_DIST_M'den az olmalı (pres/kurtarma).
        #      • Aynı TAKİMDAN son SPRINT_TEAM_COOLDOWN_FRAMES frame içinde
        #        baska bir depar logu atılmamış olmalı.
        #
        for tid, ps in curr.players.items():
            if ps.cls_id == CLS_REFEREE or ps.team_id not in (0, 1):
                continue
            if not ps.is_sprinting:
                continue

            # Yalnızca ileri yönlü depar loglanır (geri giden depar yok sayılır)
            if TEAM_ATTACK_DIR.get(ps.team_id, +1) * ps.vx_mps <= 0:
                continue

            # Sahiplik değişiminde birleştirilmiş oyuncu zaten işlendi
            if possession_changed and tid == curr_poss:
                continue

            ball_xy = curr.ball_world_xy

            if ps.has_ball:
                # Hücum deparı: topa sahip oyuncu → takım cooldown'u tüket
                if not self._team_sprint_ok(ps.team_id, frame_id):
                    continue
            else:
                # KATİ KURAL: Topa sahip olmayan oyuncunun depar logu üretilmez.
                # Tek istisna: Bu pencerede pas tespit edildiyse VE
                # bu oyuncu topun şu anki konumuna PASS_RECEIVER_DIST_M'den yakınsa.
                if pass_event is None:
                    continue   # Bu pencerede pas yok → kesinlikle log üretme
                if ball_xy is None or ps.world_xy is None:
                    continue
                recv_dist = math.hypot(
                    ps.world_xy[0] - ball_xy[0],
                    ps.world_xy[1] - ball_xy[1],
                )
                if recv_dist > PASS_RECEIVER_DIST_M:
                    continue   # Toptan uzak → pas alıcısı değil, yok say
                if not self._team_sprint_ok(ps.team_id, frame_id):
                    continue

            # Bireysel debounce da geçmeli
            if not self._debounce_ok("sprint", tid, frame_id):
                continue

            label     = self._player_label(ps)
            ball_note = " (topla)" if ps.has_ball else " (pas alacak)"

            if ps.zone == PitchZone.OPPONENT_PENALTY:
                priority = PRIORITY_HIGH
            elif ps.zone in (PitchZone.OPPONENT_HALF, PitchZone.CENTER_CIRCLE):
                priority = PRIORITY_MID
            else:
                priority = PRIORITY_LOW

            msg = (
                f"{label} {self._speed_description(ps.speed_mps)}{ball_note}! — {ps.zone.value}."
            )
            events.append(Event(
                frame_id=frame_id, priority=priority,
                event_type="sprint", message=msg,
                track_id=tid, team_id=ps.team_id,
            ))

        # ── Top durumunu bir sonraki frame için sakla ───────────────────────
        self._prev_ball_xy = curr.ball_world_xy
        self._prev_ball_possessor_team = curr.possessor_team

        return events

    # ══════════════════════════════════════════════════════════════════════════
    # Pencere Toplama (Tumbling Window Aggregation)
    # ══════════════════════════════════════════════════════════════════════════

    def _aggregate_window(self, window: List[FrameState]) -> FrameState:
        """
        Penceredeki FrameState listesini istatistiksel olarak süz.

        Süzme Kuralları
        ───────────────
        • Sürekli veriler (konum, hız)           → pencere içi ORTALAMA (mean).
        • Kategorik veriler (team_id, cls_id,
          possessor_id)                           → ÇOK TEKRAR EDEN (majority vote).
        • Forma numarası                          → pencerede herhangi bir geçerli
                                                    okuma varsa o değer; ortalama/mod
                                                    uygulanmaz. Jersey hafızasına işlenir.
        • Oyuncu pencerenin ≥ 1/3'ünde görünmüyorsa jitter/flicker → atılır.
        """
        rep_frame_id = window[-1].frame_id   # zaman damgası için son frame

        # ── Top konumu: ortalama ──────────────────────────────────────────────
        ball_xys = [fs.ball_world_xy for fs in window if fs.ball_world_xy is not None]
        if ball_xys:
            ball_xy: Optional[Tuple[float, float]] = (
                float(np.mean([b[0] for b in ball_xys])),
                float(np.mean([b[1] for b in ball_xys])),
            )
        else:
            ball_xy = None

        # ── Possessor: majority vote ──────────────────────────────────────────
        poss_ids = [fs.possessor_id for fs in window if fs.possessor_id is not None]
        possessor_id: Optional[int] = (
            Counter(poss_ids).most_common(1)[0][0] if poss_ids else None
        )

        # ── Oyuncu başına toplama ─────────────────────────────────────────────
        min_appearances = max(1, len(window) // 3)   # en az 1/3 pencerede görünmeli

        all_track_ids: set = set()
        for fs in window:
            all_track_ids.update(fs.players.keys())

        players: Dict[int, PlayerState] = {}
        for tid in all_track_ids:
            frames_with = [fs for fs in window if tid in fs.players]
            if len(frames_with) < min_appearances:
                continue   # çok az görünüm → jitter / flicker → at

            ps_list = [fs.players[tid] for fs in frames_with]

            # Kategorik: majority vote
            team_id: int = Counter(ps.team_id for ps in ps_list).most_common(1)[0][0]
            cls_id:  int = Counter(ps.cls_id  for ps in ps_list).most_common(1)[0][0]
            # Takım kimliği kilitleme: anlık flicker'ı reddet
            if team_id in (0, 1):
                team_id = self._resolve_team_id(tid, team_id)

            # Sürekli: ortalama (sadece geçerli değerler)
            wxy_vals = [ps.world_xy for ps in ps_list if ps.world_xy is not None]
            world_xy: Optional[Tuple[float, float]] = (
                (float(np.mean([w[0] for w in wxy_vals])),
                 float(np.mean([w[1] for w in wxy_vals])))
                if wxy_vals else None
            )
            speed_mps: float = float(np.mean([ps.speed_mps for ps in ps_list]))

            # Forma numarası: penceredeki ilk geçerli okuma (ortalama/mod değil)
            jersey_number: Optional[int] = None
            for ps in ps_list:
                if ps.jersey_number is not None:
                    jersey_number = ps.jersey_number
                    break
            # Hafızadan tamamla (bu track daha önce tanındıysa)
            if jersey_number is None:
                jersey_number = self._track_to_jersey.get(tid)

            vx_mps        = float(np.mean([ps.vx_mps for ps in ps_list]))
            zone          = self.get_zone_for_team(world_xy, team_id) if world_xy else PitchZone.UNKNOWN
            is_sprinting  = SPRINT_THRESHOLD_MPS <= speed_mps <= SPRINT_MAX_MPS

            agg_ps = PlayerState(
                track_id      = tid,
                team_id       = team_id,
                cls_id        = cls_id,
                pixel_bbox    = ps_list[-1].pixel_bbox,   # en son frame'den
                world_xy      = world_xy,
                zone          = zone,
                speed_mps     = speed_mps,
                vx_mps        = vx_mps,
                has_ball      = (tid == possessor_id),
                is_sprinting  = is_sprinting,
                jersey_number = jersey_number,
            )
            players[tid] = agg_ps

            # Jersey hafızasını güncelle (pencerede gerçek okuma varsa)
            if jersey_number is not None and world_xy is not None and team_id in (0, 1):
                self._jersey_memory[(team_id, jersey_number)] = world_xy
                self._track_to_jersey[tid] = jersey_number

        # Possessor has_ball bayrağını garantile
        possessor_team = -1
        if possessor_id and possessor_id in players:
            players[possessor_id].has_ball = True
            possessor_team = players[possessor_id].team_id

        return FrameState(
            frame_id        = rep_frame_id,
            players         = players,
            ball_world_xy   = ball_xy,
            ball_pixel_bbox = window[-1].ball_pixel_bbox,
            possessor_id    = possessor_id,
            possessor_team  = possessor_team,
        )

    def _flush_window(self) -> List[Event]:
        """
        Tamponu boşalt: süz → olay üret → en yüksek önceliği tut → log'a ekle.

        Pencerede birden fazla olay tetiklenirse YALNIZCA en yüksek öncelikli
        olay(lar) loglanır; alt öncelikler yutulur.
        """
        if not self._frame_buffer:
            return []

        smoothed = self._aggregate_window(self._frame_buffer)
        self._frame_buffer.clear()

        # prev_state önceki pencerenin süzülmüş hali; _generate_events içinde
        # _prev_ball_xy de pencere sonunda güncellenir.
        all_events = self._generate_events(smoothed.frame_id, smoothed, self.prev_state)
        self.prev_state = smoothed   # SONRA güncelle

        if not all_events:
            return []

        # Sadece en yüksek öncelikli olay(ları) tut
        max_priority = max(e.priority for e in all_events)
        best_events  = [e for e in all_events if e.priority == max_priority]

        self.event_log.extend(best_events)
        for ev in best_events:
            self._write_event_log(ev)
        return best_events

    def _write_event_log(self, event: "Event") -> None:
        """
        Olayı events_metadata.jsonl dosyasına ek (append) modunda yazar.
        Her satır bağımsız bir JSON nesnesidir — Qwen3 veri havuzu formatı.
        """
        record = {
            "priority":   event.priority,
            "frame":      event.frame_id,
            "event_type": event.event_type,
            "event_text": event.message,
        }
        with open(self._event_log_path, "a", encoding="utf-8") as fh:
            fh.write(json.dumps(record, ensure_ascii=False) + "\n")

    # ══════════════════════════════════════════════════════════════════════════
    # Genel API: Tek Frame / Tüm Video
    # ══════════════════════════════════════════════════════════════════════════

    def process_frame(self, frame_id: int) -> List[Event]:
        """
        Frame'i tampona ekle; pencere dolduğunda süzülmüş olay üret.

        Pencere dolana kadar boş liste döner. Pencere tamamlandığında
        o penceredeki en yüksek öncelikli olay(lar) döner.

        LLM entegrasyonu:
            for event in engine.process_frame(1540):
                llm_client.chat(str(event))   # her zaman ≥ Pmin
        """
        frame_state = self._build_frame_state(frame_id)
        self._frame_buffer.append(frame_state)

        if len(self._frame_buffer) >= self.window_size:
            return self._flush_window()
        return []

    def run(
        self,
        start_frame:    Optional[int] = None,
        end_frame:      Optional[int] = None,
        min_priority:   int           = 1,
        verbose:        bool          = True,
    ) -> List[Event]:
        """
        Tüm frame'leri sırayla işle; tüm üretilen Event nesnelerini döndür.

        Args:
            start_frame  : İlk frame numarası (None = CSV'deki en küçük frame).
            end_frame    : Son frame numarası  (None = CSV'deki en büyük frame).
            min_priority : Sadece bu öncelik ve üstü olayları terminale bas / döndür.
                           Örn. min_priority=2 → düşük öncelikli loglar filtrelenir.
            verbose      : True ise terminale de basılır.

        Returns:
            Event nesnelerinin listesi.
        """
        all_frames: List[int] = sorted(self.tracking_df["frame_id"].unique().tolist())

        if start_frame is not None:
            all_frames = [f for f in all_frames if f >= start_frame]
        if end_frame is not None:
            all_frames = [f for f in all_frames if f <= end_frame]

        if not all_frames:
            print("[EventEngine] İşlenecek frame bulunamadı.")
            return []

        print(
            f"[EventEngine] {len(all_frames)} frame işlenecek "
            f"({all_frames[0]} → {all_frames[-1]}) …"
        )

        for fid in all_frames:
            events = self.process_frame(fid)
            if verbose:
                for ev in events:
                    if ev.priority >= min_priority:
                        print(str(ev))

        # Kalan yarım pencereyi de değerlendir (son N frame silindiğinde kaybolmasın)
        if self._frame_buffer:
            remaining = self._flush_window()
            if verbose:
                for ev in remaining:
                    if ev.priority >= min_priority:
                        print(str(ev))

        filtered = [e for e in self.event_log if e.priority >= min_priority]
        print(
            f"[EventEngine] Bitti. Toplam {len(self.event_log)} olay üretildi "
            f"({len(filtered)} tanesi öncelik ≥ {min_priority})."
        )
        return filtered

    def get_llm_prompt(
        self,
        min_priority: int = 1,
        max_events:   int = 50,
    ) -> str:
        """
        Olay log'unu LLM'e gönderilmeye hazır tek bir metin olarak döndür.

        Args:
            min_priority : Sadece bu öncelik ve üstü olayları dahil et.
            max_events   : LLM bağlam penceresini aşmamak için son N olayı al.

        Örnek kullanım:
            # Tüm kritik+orta olayları Qwen3'e gönder
            prompt = engine.get_llm_prompt(min_priority=2)
            response = qwen_client.chat(prompt)
        """
        if not self.event_log:
            return "Henüz kayıt edilmiş olay yok."

        filtered = [e for e in self.event_log if e.priority >= min_priority]
        # En son max_events olayı al (bağlam taşmasını önle)
        recent = filtered[-max_events:]

        header = (
            "Aşağıda bir futbol maçının gerçek zamanlı olay günlüğü bulunmaktadır. "
            "Sen deneyimli bir futbol spikerisin. "
            "Bu olayları akıcı ve heyecanlı bir şekilde, Türkçe olarak yorumla:\n\n"
        )
        body = "\n".join(str(e) for e in recent)
        return header + body

    # ══════════════════════════════════════════════════════════════════════════
    # Yardımcı Metotlar
    # ══════════════════════════════════════════════════════════════════════════

    def _nearest_calibration(self, frame_id: int) -> Optional[dict]:
        """
        Kalibrasyon frame indeksinde, verilen frame_id'ye en yakın kaydı döndür.
        İkili arama (O log n) kullanır.
        """
        if not self._calib_keys:
            return None
        pos = self._bisect_nearest(self._calib_keys, frame_id)
        return self.calib_index[self._calib_keys[pos]]

    @staticmethod
    def _bisect_nearest(sorted_list: List[int], target: int) -> int:
        """Sıralı listede target değerine en yakın elemanın indeksini döner."""
        lo, hi = 0, len(sorted_list) - 1
        while lo < hi:
            mid = (lo + hi) // 2
            if sorted_list[mid] < target:
                lo = mid + 1
            else:
                hi = mid
        # lo ve lo-1 komşularından yakın olanı seç
        if lo > 0 and abs(sorted_list[lo - 1] - target) <= abs(sorted_list[lo] - target):
            return lo - 1
        return lo


# ─── Giriş Noktası ────────────────────────────────────────────────────────────

if __name__ == "__main__":
    BASE_DIR    = os.path.dirname(os.path.abspath(__file__))
    TRACKING    = os.path.join(BASE_DIR, "tracking_20260418204640_00053_merged.csv")
    CALIBRATION = os.path.join(BASE_DIR, "calibration_frames_20260418204058_58334.jsonl")

    # Motor oluştur
    engine = EventEngine(
        tracking_csv_path      = TRACKING,
        calibration_jsonl_path = CALIBRATION,
        fps                    = 50.0,
    )

    # Tüm veriyi işle — sadece orta ve yüksek öncelikli olayları terminale bas
    events = engine.run(verbose=True, min_priority=2)

    # ── Öncelik dağılımı özeti ──────────────────────────────────────────────
    print("\n" + "═" * 60)
    print("OLAY ÖZETİ (Tüm Öncelikler):")
    print("═" * 60)
    from collections import Counter
    dist = Counter(e.priority for e in engine.event_log)
    for p in sorted(dist):
        label = {1: "Düşük ", 2: "Orta  ", 3: "Kritik"}.get(p, str(p))
        print(f"  Öncelik {p} ({label}): {dist[p]:>4d} olay")

    # ── LLM'e göndermek için hazır metin (son 30 kritik/orta olay) ─────────
    print("\n" + "═" * 60)
    print("LLM PROMPT ÖRNEĞİ (min_priority=2, son 30 olay):")
    print("═" * 60)
    print(engine.get_llm_prompt(min_priority=2, max_events=30))
