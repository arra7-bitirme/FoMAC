"""
llm_lab.py — LLM Commentary Optimization Lab

Pipeline'ı yeniden çalıştırmadan mevcut JSON'lar üzerinde
farklı prompt stratejilerini test et, referansla karşılaştır.

Kullanım:
  python llm_lab.py                          # tüm item'lar, varsayılan variant
  python llm_lab.py --item goal              # sadece gol event'i
  python llm_lab.py --item 3                 # 3. item (0-indexed)
  python llm_lab.py --item all --variant B   # tüm item'lar, variant B
  python llm_lab.py --variant all            # tüm variant'ları karşılaştır (sadece gol)
  python llm_lab.py --dry                    # LLM'e göndermeden prompt'u göster
  python llm_lab.py --save                   # sonuçları lab_results/ altına kaydet

Variant'lar:
  A  — Mevcut pipeline davranışı (baseline)
  B  — Zenginleştirilmiş: takım adları çözüldü, oyuncu isimleri eklendi
  C  — B + daha derin top yörüngesi + event_engine temizliği + narratif akış
"""

import argparse
import json
import os
import re
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# ── Bağımlılıklar ─────────────────────────────────────────────────────────────
try:
    import httpx
except ImportError:
    httpx = None

sys.path.insert(0, str(Path(__file__).parent))
from pipeline import (
    _build_commentary_item_prompt,
    _build_match_context,
    _strip_think_blocks,
    _extract_commentary_text_best_effort,
)

# ── Sabitler ──────────────────────────────────────────────────────────────────
UPLOADS_DIR = Path(__file__).parent / "uploads"
REF_FILE    = Path(__file__).parent / "input_files" / "galjuvgol1.txt"
LLM_URL     = "http://localhost:8001"
LLM_MODEL   = "nvidia/Qwen3-8B-NVFP4"
MAX_TOKENS  = 1500

# ─────────────────────────────────────────────────────────────────────────────
# 1. DOSYA BULMA
# ─────────────────────────────────────────────────────────────────────────────

def _find_latest(pattern: str) -> Optional[Path]:
    files = sorted(UPLOADS_DIR.glob(pattern))
    return files[-1] if files else None


def load_data() -> Dict[str, Any]:
    ci   = _find_latest("commentary_input_*.json")
    jf   = _find_latest("tracking_*.jersey.json")
    mf   = _find_latest("galjuv_mapping_*.json")

    if not ci:
        sys.exit("ERROR: commentary_input_*.json bulunamadı")

    with open(ci, encoding="utf-8") as f:
        commentary_input = json.load(f)

    jersey_by_track: Dict[int, Dict] = {}
    if jf:
        with open(jf, encoding="utf-8") as f:
            jersey_by_track = {j["track_id"]: j for j in json.load(f)}

    roster_raw: Optional[str] = None
    if mf:
        with open(mf, encoding="utf-8") as f:
            roster_raw = f.read()

    ref_text = REF_FILE.read_text(encoding="utf-8") if REF_FILE.exists() else ""

    print(f"[DATA] commentary_input : {ci.name}")
    print(f"[DATA] jersey           : {jf.name if jf else 'YOK'}")
    print(f"[DATA] roster/mapping   : {mf.name if mf else 'YOK'}")
    print(f"[DATA] referans metin   : {'OK' if ref_text else 'YOK'}")
    print()

    return {
        "commentary_input": commentary_input,
        "jersey_by_track": jersey_by_track,
        "roster_raw": roster_raw,
        "ref_text": ref_text,
    }


# ─────────────────────────────────────────────────────────────────────────────
# 2. ZENGİNLEŞTİRME KATMANI
# ─────────────────────────────────────────────────────────────────────────────

def _resolve_team_name(team_id: Any, team_names: Dict[str, str]) -> str:
    if team_id is None:
        return "bilinmiyor"
    return team_names.get(str(team_id), f"Takım {team_id}")


def _resolve_player(
    track_id: int,
    jersey_by_track: Dict[int, Dict],
    roster_lookup: Dict[str, str],
    team_names: Dict[str, str],
) -> str:
    """track_id → 'Victor Osimhen (Galatasaray #45)' gibi bir açıklama döndür."""
    info = jersey_by_track.get(track_id, {})
    jnum = str(info.get("jersey_number") or "-1")
    pname = info.get("player_name") or roster_lookup.get(jnum)
    team_id = info.get("team_id")
    team = _resolve_team_name(team_id, team_names) if team_id is not None else None

    parts = []
    if pname:
        parts.append(pname)
    if jnum != "-1":
        parts.append(f"#{jnum}")
    if team:
        parts.append(f"({team})")
    return " ".join(parts) if parts else f"track_{track_id}"


def _fix_engine_text(text: str, team_names: Dict[str, str]) -> str:
    """event_engine'deki 'Takım A' / 'Takım B' ifadelerini gerçek isimle değiştir."""
    t0 = team_names.get("0", "Takım 0")
    t1 = team_names.get("1", "Takım 1")
    text = re.sub(r"Tak[iı]m\s+A\b", t0, text, flags=re.IGNORECASE)
    text = re.sub(r"Tak[iı]m\s+B\b", t1, text, flags=re.IGNORECASE)
    return text


def _describe_ball_trajectory(frame_samples: List[Dict]) -> str:
    """frame_samples'tan daha zengin bir top yörüngesi açıklaması üret."""
    ball_frames = [s for s in frame_samples if s.get("ball")]
    if not ball_frames:
        return ""

    positions = [(s["ball"]["world_xy"][0], s["ball"]["world_xy"][1]) for s in ball_frames]
    timecodes  = [s.get("timecode", "") for s in ball_frames]
    in_box     = sum(1 for s in ball_frames if s["ball"].get("penalty_area_proximity"))

    # Yön hesapla
    sx, sy = positions[0]
    ex, ey = positions[-1]
    dx, dy = ex - sx, ey - sy

    parts = []

    # Ceza sahası
    if in_box >= 2:
        parts.append("top ceza sahasına girdi")
    elif in_box == 1:
        parts.append("top kısa süreliğine ceza sahası yakınına ulaştı")

    # Yatay hareket (kanat)
    if abs(dx) >= 5:
        if dx > 0:
            parts.append(f"sağa ({round(abs(dx), 0):.0f}m) kaydı")
        else:
            parts.append(f"sola ({round(abs(dx), 0):.0f}m) kaydı")

    # Dikey hareket (ileri/geri)
    if dy >= 8:
        parts.append(f"rakip kaleye doğru hızla ilerledi ({round(dy,0):.0f}m)")
    elif dy <= -8:
        parts.append(f"kendi kalesine doğru geri döndü ({round(abs(dy),0):.0f}m)")
    elif abs(dy) < 3 and abs(dx) < 3:
        parts.append("top dar bir alanda tutuldu")

    # Baskı trendi
    pressures = [s["ball"].get("nearby_pressure_count", 0) for s in ball_frames]
    if pressures:
        avg_p = sum(pressures) / len(pressures)
        max_p = max(pressures)
        if max_p >= 5:
            parts.append(f"pik baskı: {max_p} rakip etrafında")
        elif avg_p >= 3:
            parts.append(f"ortalama {avg_p:.1f} oyuncu yakın")

    if not parts:
        return ""

    return "Top: " + "; ".join(parts) + f" [{timecodes[0]}→{timecodes[-1]}]"


def enrich_item(
    item: Dict,
    jersey_by_track: Dict[int, Dict],
    roster_lookup: Dict[str, str],
    team_names: Dict[str, str],
) -> Dict:
    """
    Mevcut item'ı zenginleştir:
    - focus_players'a player_name ve takım adı ekle
    - nearest_players'a player_name ekle
    - event_engine_context'teki 'Takım A/B' → gerçek isim
    - ball_trajectory daha zengin hesapla
    """
    import copy
    item = copy.deepcopy(item)

    # ── focus_players zenginleştir ─────────────────────────────────────────
    ms = item.get("match_state") or {}
    ss = ms.get("state_summary") or {}
    for fp in ss.get("focus_players") or []:
        tid = fp.get("track_id")
        if tid:
            info = jersey_by_track.get(int(tid), {})
            jnum = str(info.get("jersey_number") or "-1")
            pname = info.get("player_name") or roster_lookup.get(jnum)
            if pname:
                fp["player_name"] = pname
            if fp.get("team_id") is None or int(fp.get("team_id", -1)) == -1:
                if info.get("team_id") is not None:
                    fp["team_id"] = info["team_id"]
            # takım adı ekle
            team_id = fp.get("team_id")
            if team_id is not None and int(team_id) != -1:
                fp["team_name"] = _resolve_team_name(team_id, team_names)

    # ── nearest_players zenginleştir ──────────────────────────────────────
    for fs in ms.get("frame_samples") or []:
        ball = fs.get("ball")
        if not ball:
            continue
        for np_ in ball.get("nearest_players") or []:
            tid = np_.get("track_id")
            if tid:
                info = jersey_by_track.get(int(tid), {})
                jnum = str(info.get("jersey_number") or "-1")
                pname = info.get("player_name") or roster_lookup.get(jnum)
                if pname:
                    np_["player_name"] = pname
                if np_.get("team_id") is None or int(np_.get("team_id", -1)) == -1:
                    if info.get("team_id") is not None:
                        np_["team_id"] = info["team_id"]
                team_id = np_.get("team_id")
                if team_id is not None and int(team_id) != -1:
                    np_["team_name"] = _resolve_team_name(team_id, team_names)

    # ── event_engine_context: Takım A/B → gerçek isim + jersey → isim ────
    engine_ctx = item.get("event_engine_context") or []
    for ec in engine_ctx:
        raw_text = ec.get("text", "")
        raw_text = _fix_engine_text(raw_text, team_names)
        # "#N" → isim varsa sadece isim, yoksa "bir oyuncu" (pipeline ile aynı davranış)
        def _replace_jersey(m):
            num = m.group(1)
            name = roster_lookup.get(num)
            return name if name else "bir oyuncu"
        raw_text = re.sub(r"#(\d+)", _replace_jersey, raw_text)
        ec["text"] = raw_text
        # player_name field: track_id → jersey OCR → roster lookup
        tid = ec.get("track_id")
        if tid and not ec.get("player_name"):
            info = jersey_by_track.get(int(tid), {})
            jnum = str(info.get("jersey_number") or "-1")
            pname = info.get("player_name") or roster_lookup.get(jnum)
            if pname:
                ec["player_name"] = pname
        # team_id field: track_id'den doldur (jersey_by_track'ta her zaman var)
        if ec.get("team_id") is None and tid:
            info = jersey_by_track.get(int(tid), {})
            ec_tid_team = info.get("team_id")
            if ec_tid_team is not None:
                ec["team_id"] = ec_tid_team

    # ── ball_trajectory override ───────────────────────────────────────────
    frame_samples = ms.get("frame_samples") or []
    traj = _describe_ball_trajectory(frame_samples)
    if traj:
        item.setdefault("_lab_enriched", {})["ball_trajectory"] = traj

    return item


# ─────────────────────────────────────────────────────────────────────────────
# 3. PROMPT VARIANT'LARI
# ─────────────────────────────────────────────────────────────────────────────

def build_prompt_variant_a(item: Dict, recent: List[str], match_ctx: Optional[Dict]) -> str:
    """A — Mevcut pipeline davranışı (baseline, zenginleştirme yok)."""
    return _build_commentary_item_prompt(item, recent, match_context=match_ctx)


def build_prompt_variant_b(
    item: Dict,
    recent: List[str],
    match_ctx: Optional[Dict],
    jersey_by_track: Dict,
    roster_lookup: Dict,
    team_names: Dict,
) -> str:
    """B — Zenginleştirilmiş veri: takım adları, oyuncu isimleri, engine text fix."""
    enriched = enrich_item(item, jersey_by_track, roster_lookup, team_names)
    return _build_commentary_item_prompt(enriched, recent, match_context=match_ctx)


def build_prompt_variant_c(
    item: Dict,
    recent: List[str],
    match_ctx: Optional[Dict],
    jersey_by_track: Dict,
    roster_lookup: Dict,
    team_names: Dict,
    all_items: List[Dict],
    item_idx: int,
) -> str:
    """C — B + oyuncu/takım bilgisi net özet başta + top yörüngesi + önceki olaylar."""
    enriched = enrich_item(item, jersey_by_track, roster_lookup, team_names)

    # ── Key players: preamble'a temiz isimler ─────────────────────────────
    preamble_lines: List[str] = []

    # Event engine'den en yüksek öncelikli → oyuncu adı çıkar (# olmadan)
    engine_ctx = enriched.get("event_engine_context") or []
    top_events = sorted(engine_ctx, key=lambda x: -x.get("priority", 0))[:2]
    seen_players: set = set()
    for ec in top_events:
        pname = ec.get("player_name") or ""
        # Event engine text'inden # ve semboller kaldırılmış versiyonu al
        clean_text = re.sub(r"#\d+\s*", "", ec.get("text", "")).strip()
        # "—" ve tire de kaldır
        clean_text = re.sub(r"\s*[—–-]+\s*", " ", clean_text).strip()
        if pname and pname not in seen_players:
            preamble_lines.append(f"- Sahadaki oyuncu: {pname} ({clean_text[:80]})")
            seen_players.add(pname)
        elif clean_text:
            # text'te roster'dan bilinen isimler varsa onları da topla
            for rname in roster_lookup.values():
                if rname and rname in clean_text and rname not in seen_players:
                    seen_players.add(rname)
            preamble_lines.append(f"- {clean_text[:100]}")

    # Top yörüngesi (# olmadan)
    traj = enriched.get("_lab_enriched", {}).get("ball_trajectory")
    if traj:
        clean_traj = re.sub(r"#\d+\s*", "", traj)
        preamble_lines.append(f"- {clean_traj}")

    # Önceki olaylar (max 3)
    if item_idx > 0:
        prev_items = all_items[max(0, item_idx - 3):item_idx]
        for pi in prev_items:
            ptc  = pi.get("event_timecode") or pi.get("timecode", "")
            plbl = pi.get("event_label", "")
            preamble_lines.append(f"- Onceki olay {ptc}: {plbl}")

    # Pipeline prompt'u al (zenginleştirilmiş item)
    base_prompt = _build_commentary_item_prompt(enriched, recent, match_context=match_ctx)

    # Eğer oyuncu adları tespit edildiyse → kurallara açık izin ekle
    if seen_players:
        names_str = ", ".join(sorted(seen_players))
        permission = (
            f"\nNOT: Şu oyuncu adlarını yorumunda KULLANABİLİRSİN (bunlar gerçek veri): {names_str}\n"
        )
        # Kurallar bölümünün hemen öncesine ekle
        base_prompt = base_prompt.replace(
            "- Forma numarası veya oyuncu ismi UYDURMA;",
            f"{permission}- Forma numarası veya oyuncu ismi UYDURMA;",
        )

    if preamble_lines:
        preamble = (
            "SAHADA KIM VAR (Bu isimleri ve bilgileri yorumunda kullan):\n"
            + "\n".join(preamble_lines)
            + "\n"
        )
        return preamble + "\n" + base_prompt
    return base_prompt


# ─────────────────────────────────────────────────────────────────────────────
# 4. REFERANS METİN EŞLEŞTİRME
# ─────────────────────────────────────────────────────────────────────────────

def parse_reference(ref_text: str) -> List[Tuple[float, float, str]]:
    """
    galjuvgol1.txt'i parse et.
    Returns: [(start_sec, end_sec, text), ...]
    """
    segments = []
    pattern = re.compile(r"\((\d+):(\d+)\s*-\s*(\d+):(\d+)\)\s*(.*?)(?=\(\d+:\d+|\Z)", re.DOTALL)
    for m in pattern.finditer(ref_text):
        s_min, s_sec, e_min, e_sec = int(m.group(1)), int(m.group(2)), int(m.group(3)), int(m.group(4))
        text = m.group(5).strip()
        if text:
            segments.append((s_min * 60 + s_sec, e_min * 60 + e_sec, text))
    return segments


def find_reference_snippet(event_t: float, ref_segments: List[Tuple]) -> Optional[str]:
    """Event zamanına en yakın referans segmentini bul (±10s tolerans)."""
    best = None
    best_dist = float("inf")
    for start, end, text in ref_segments:
        # Segment içinde mi?
        if start <= event_t <= end:
            return text
        # En yakın segment
        dist = min(abs(event_t - start), abs(event_t - end))
        if dist < best_dist and dist <= 10.0:
            best_dist = dist
            best = text
    return best


# ─────────────────────────────────────────────────────────────────────────────
# 5. LLM ÇAĞRISI
# ─────────────────────────────────────────────────────────────────────────────

SYSTEM_PROMPT = (
    "Sen deneyimli bir Türkçe canlı futbol spikerisin. "
    "Görevin: sana verilen maç verilerini (olay tipi, sahada top konumu, baskı seviyesi, "
    "forma numaraları, oyuncu isimleri, önceki olaylar) analiz edip heyecanlı, akıcı, özgün "
    "Türkçe bir spiker cümlesi üretmek. "
    "Düşünce sürecin <think>...</think> bloğunda saklı kalacak; "
    "nihai çıktı YALNIZCA Türkçe JSON olacak: {\"text\": \"...\"}. "
    "Veri olmayan oyuncu veya takım bilgisi uydurma. "
    "Cümle tekrarından kaçın; her olay için yeni ve farklı bir anlatım tonu seç.\n"
    "KRITIK: Üretilen metin doğrudan TTS sistemi tarafından okunacak — "
    "parantez, üç nokta, semboller (#, %, &) kullanma. "
    "Cevap kesinlikle sadece JSON olmalı: {\"text\": \"BURADA YORUM\"}"
)


def call_llm(prompt: str, url: str = LLM_URL, model: str = LLM_MODEL, timeout: float = 120.0) -> Tuple[Optional[str], Optional[str]]:
    if httpx is None:
        return None, "httpx yüklü değil"
    endpoint = url.rstrip("/") + "/v1/chat/completions"
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user",   "content": prompt},
        ],
        "temperature": 0.7,
        "max_tokens": MAX_TOKENS,
        "chat_template_kwargs": {"enable_thinking": True},
    }
    try:
        with httpx.Client(timeout=timeout) as client:
            r = client.post(endpoint, json=payload)
            if not r.is_success:
                return None, f"HTTP {r.status_code}: {r.text[:400]}"
            data = r.json()
        raw = str((data.get("choices") or [{}])[0].get("message", {}).get("content") or "").strip()
    except Exception as e:
        return None, str(e)

    # think bloklarını temizle
    clean = _strip_think_blocks(raw)
    text  = _extract_commentary_text_best_effort(clean) or _extract_commentary_text_best_effort(raw)
    return text, None


# ─────────────────────────────────────────────────────────────────────────────
# 6. GÖRÜNTÜLEME
# ─────────────────────────────────────────────────────────────────────────────

def _sep(char="─", n=70): return char * n


def print_result(
    item: Dict,
    variant: str,
    prompt: str,
    llm_output: Optional[str],
    llm_error: Optional[str],
    ref_snippet: Optional[str],
    elapsed: float,
    dry_run: bool,
):
    tc    = item.get("event_timecode") or item.get("timecode", "?")
    label = item.get("event_label", "?")
    print(_sep("═"))
    print(f"[{tc}] {label}  |  Variant: {variant}  |  ({elapsed:.1f}s)")
    print(_sep())

    print("PROMPT (son 600 karakter):")
    print(prompt[-600:] if len(prompt) > 600 else prompt)
    print()

    if dry_run:
        print("── DRY RUN: LLM çağrısı atlandı ──")
    elif llm_error:
        print(f"LLM HATA: {llm_error}")
    else:
        print("LLM ÇIKTISI:")
        print(f"  → {llm_output}")

    if ref_snippet:
        print()
        print("REFERANS METİN (galjuvgol1.txt):")
        for line in ref_snippet.strip().splitlines():
            print(f"  | {line}")
    print()


# ─────────────────────────────────────────────────────────────────────────────
# 7. KAYDETME
# ─────────────────────────────────────────────────────────────────────────────

def save_results(results: List[Dict], variant: str):
    out_dir = Path(__file__).parent / "lab_results"
    out_dir.mkdir(exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = out_dir / f"lab_{ts}_variant{variant}.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"[KAYIT] {out_path}")


# ─────────────────────────────────────────────────────────────────────────────
# 8. ANA AKIŞ
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="LLM Commentary Lab")
    parser.add_argument("--item",    default="all",
                        help="'all', 'goal', 'foul', veya 0-indexed sayı")
    parser.add_argument("--variant", default="C",
                        help="A | B | C | all")
    parser.add_argument("--dry",     action="store_true",
                        help="LLM çağrısı yapma, sadece prompt'u göster")
    parser.add_argument("--save",    action="store_true",
                        help="Sonuçları lab_results/ altına kaydet")
    parser.add_argument("--url",     default=LLM_URL,
                        help=f"LLM URL (varsayılan: {LLM_URL})")
    parser.add_argument("--model",   default=LLM_MODEL)
    args = parser.parse_args()

    # ── Veri yükle ────────────────────────────────────────────────────────
    llm_url   = args.url
    llm_model = args.model

    data = load_data()
    ci        = data["commentary_input"]
    jbt       = data["jersey_by_track"]
    roster_raw = data["roster_raw"]
    ref_text  = data["ref_text"]

    all_items = ci.get("items", [])
    match_ctx = ci.get("match_context")

    # roster_lookup: jersey# → isim
    roster_lookup: Dict[str, str] = {}
    team_names: Dict[str, str] = {}
    if roster_raw:
        mc = _build_match_context(roster_json_str=roster_raw, jersey_by_track=jbt)
        if mc:
            if not match_ctx:
                match_ctx = mc
            team_names = {str(k): v for k, v in (mc.get("team_names") or {}).items()}
            for team_players in (mc.get("rosters") or {}).values():
                for p in (team_players if isinstance(team_players, list) else []):
                    if isinstance(p, dict) and p.get("name") and p.get("number") is not None:
                        try:
                            roster_lookup[str(int(p["number"]))] = str(p["name"])
                        except Exception:
                            pass
    # jersey_by_track'ten de player_name ekle (varsa)
    for tid, info in jbt.items():
        jnum = str(info.get("jersey_number") or "-1")
        if jnum != "-1" and not info.get("player_name"):
            pname = roster_lookup.get(jnum)
            if pname:
                info["player_name"] = pname

    ref_segments = parse_reference(ref_text) if ref_text else []

    # ── Hangi item'lar? ───────────────────────────────────────────────────
    if args.item == "all":
        target_items = list(enumerate(all_items))
    elif args.item == "goal":
        target_items = [(i, it) for i, it in enumerate(all_items)
                        if it.get("event_label", "").lower() in ("goal", "own goal", "penalty - goal")]
    elif args.item == "foul":
        target_items = [(i, it) for i, it in enumerate(all_items)
                        if it.get("event_label", "").lower() == "foul"]
    elif args.item.isdigit():
        idx = int(args.item)
        target_items = [(idx, all_items[idx])] if idx < len(all_items) else []
    else:
        target_items = [(i, it) for i, it in enumerate(all_items)
                        if args.item.lower() in it.get("event_label", "").lower()]

    if not target_items:
        print(f"Hiç item bulunamadı: --item {args.item!r}")
        return

    # ── Variant'lar ───────────────────────────────────────────────────────
    variants = ["A", "B", "C"] if args.variant == "all" else [args.variant.upper()]

    # ── Çalıştır ──────────────────────────────────────────────────────────
    all_results = []
    recent_texts: List[str] = []

    for idx, item in target_items:
        event_t = float(item.get("event_t") or item.get("t") or 0.0)
        ref_snippet = find_reference_snippet(event_t, ref_segments)

        for variant in variants:
            t0 = time.perf_counter()

            if variant == "A":
                prompt = build_prompt_variant_a(item, recent_texts, match_ctx)
            elif variant == "B":
                prompt = build_prompt_variant_b(item, recent_texts, match_ctx, jbt, roster_lookup, team_names)
            else:
                prompt = build_prompt_variant_c(item, recent_texts, match_ctx, jbt, roster_lookup, team_names, all_items, idx)

            llm_out, llm_err = (None, None)
            if not args.dry:
                llm_out, llm_err = call_llm(prompt, url=llm_url, model=llm_model)

            elapsed = time.perf_counter() - t0

            print_result(item, variant, prompt, llm_out, llm_err, ref_snippet, elapsed, args.dry)

            result = {
                "timecode": item.get("event_timecode") or item.get("timecode"),
                "event_label": item.get("event_label"),
                "variant": variant,
                "prompt_length": len(prompt),
                "llm_output": llm_out,
                "llm_error": llm_err,
                "reference": ref_snippet,
                "elapsed_sec": round(elapsed, 2),
            }
            all_results.append(result)
            if llm_out:
                recent_texts.append(llm_out)

    if args.save and all_results:
        for v in variants:
            v_results = [r for r in all_results if r["variant"] == v]
            if v_results:
                save_results(v_results, v)


if __name__ == "__main__":
    main()
