from __future__ import annotations

from bisect import bisect_left, bisect_right
import csv
import gc
import json
import os
import subprocess
import sys
import time
import re
import tempfile
import wave
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, FrozenSet, List, Optional, Tuple
import threading

import cv2
import numpy as np
from tqdm import tqdm

try:
    import httpx  # type: ignore
except Exception:  # pragma: no cover
    httpx = None

@dataclass
class FullPipelineConfig:
    start_seconds: float = 0.0
    duration_seconds: Optional[float] = None

    run_calibration: bool = True
    calibration_detector_weights: Optional[str] = None
    calibration_kp_weights: Optional[str] = None
    calibration_line_weights: Optional[str] = None
    calibration_conf_thres: float = 0.30
    calibration_write_frames_jsonl: bool = False
    calibration_frames_stride: int = 1
    calibration_yolo_frame_window: int = 8
    calibration_yolo_selection_mode: str = "ball_priority"
    calibration_interpolation_mode: str = "linear"
    calibration_torch_compile: bool = False
    calibration_torch_compile_mode: str = "reduce-overhead"

    run_tracking: bool = True
    tracking_device: Optional[str] = None
    tracking_config_path: Optional[str] = None
    detector_weights: Optional[str] = None
    reid_weights: Optional[str] = None

    run_action_spotting: bool = True
    tdeed_repo_dir: Optional[str] = None
    action_model_name: str = "SoccerNet_big"
    action_frame_width: int = 398
    action_frame_height: int = 224
    action_threshold: float = 0.50
    action_nms_window_sec: float = 10.0
    run_ball_action_spotting: bool = True
    ball_model_name: str = "SoccerNetBall_challenge2"
    ball_checkpoint_path: Optional[str] = None
    ball_action_threshold: float = 0.20
    features_path: Optional[str] = None
    checkpoint_path: Optional[str] = None

    overlay_event_window_sec: float = 3.0

    run_jersey_number_recognition: bool = True
    qwen_vl_url: str = "http://localhost:8080/"
    qwen_vl_model: str = "Qwen3VL-8B-Instruct-Q4_K_M.gguf"
    qwen_vl_manage_container: bool = True
    qwen_vl_container_id: str = "4d2ba276bce6347e95bb962a538bf43d70057a151d0ea03e35110c85ec0ec36c"
    qwen_vl_stop_before_commentary: bool = True
    qwen_vl_ready_timeout_sec: float = 60.0
    jersey_prompt: str = (
        "You are reading a football jersey.\n\n"
        "Visually read the number printed on the player's shirt.\n\n"
        "Rules:\n\n"
        "* Output digits only.\n"
        "* If no number is visible or readable, output: -1\n"
        "* Do not guess.\n"
        "* Do not describe the image.\n"
        "* Do not output words.\n\n"
        "Return only the final answer.\n"
    )
    jersey_max_tracks: int = 0
    jersey_max_samples_per_track: int = 5
    jersey_min_det_conf: float = 0.55
    jersey_min_box_area: int = 40 * 40
    jersey_min_frame_gap: int = 20
    jersey_frame_topk: int = 5
    jersey_crops_dir: Optional[str] = None
    jersey_vis_filter: bool = False
    jersey_vis_min_score: float = 0.12
    jersey_in_tracking: bool = True
    jersey_merge_same_number: bool = True
    jersey_merge_min_confidence: float = 0.60
    jersey_merge_max_overlap_frames: int = 5

    # Optional roster mapping JSON (raw string content from the uploaded file).
    # Format: {"match_info": {...}, "rosters": {"team_a": [{"name": ..., "number": ...}], ...}}
    player_roster_json: Optional[str] = None

    run_commentary: bool = True
    commentary_max_events: int = 2000000
    commentary_possession_max_age_sec: float = 8.0
    commentary_llm_backend: str = "vllm"
    commentary_llm_url: str = "http://localhost:8001/"
    commentary_llm_model: str = "nvidia/Qwen3-8B-NVFP4"
    commentary_vllm_batch_size: int = 4
    commentary_vllm_enable_thinking: bool = True
    commentary_vllm_max_tokens: int = 1800
    commentary_flush_gpu_before_llm: bool = True
    commentary_context_window_sec: float = 12.0
    commentary_context_stride_sec: float = 1.0
    commentary_context_max_samples: int = 9
    commentary_segment_sec: float = 30.0
    commentary_state_interval_sec: float = 10.0
    commentary_llm_timeout_sec: float = 180.0
    commentary_min_audio_gap_sec: float = 1.0  # gap between clip END and next clip START
    commentary_enable_tts: bool = True
    commentary_tts_backend: str = "xttsv2"
    commentary_speaker_wav: Optional[str] = str(Path(__file__).resolve().parent / "ertem_sener.wav")
    commentary_ambient_audio: Optional[str] = str(Path(__file__).resolve().parent / "crowd.mp3")
    commentary_goal_sfx_audio: Optional[str] = str(Path(__file__).resolve().parent / "goal_cheer.wav")
    commentary_goal_voice_audio: Optional[str] = str(Path(__file__).resolve().parent / "golsesi.wav")  # spiker "GOL!" kaydı
    commentary_filler_gap_sec: float = 4.0  # trigger a filler after this many seconds of silence
    commentary_event_pre_delay_sec: float = 2.0  # announce this many sec BEFORE event fires (non-goal events)
    commentary_min_clip_cooldown_sec: float = 2.0  # min silence between any two TTS clips (filler-to-filler)
    commentary_action_spotting_cooldown_sec: float = 8.0  # min silence after an action-spotting clip before next clip

    possession_dist_norm: float = 0.08
    possession_stable_frames: int = 6
    possession_stride_frames: int = 5
    ball_cls_id: int = 1
    player_cls_id: int = 0


# ─────────────────────────────────────────────────────────────────────────────
# EventEngine Data Model
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class EventEngineMeta:
    """Single frame-level metadata event produced by the EventEngine module.

    EventEngine ürettiği her olayı bu yapı ile temsil eder.
    Örnek kaynak JSON: {"priority": 2, "frame": 9056, "event_text": "..."}
    """
    frame: int
    priority: int            # 1=low (depar), 2=medium (orta), 3=high (acil)
    event_text: str
    speed_ms: Optional[float] = None      # oyuncunun anlık hızı m/s
    zone: Optional[str] = None            # saha bölgesi etiketi
    player_name: Optional[str] = None     # roster'dan çözülmüş isim
    track_id: Optional[int] = None        # tracker track_id

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "EventEngineMeta":
        return cls(
            frame=int(d.get("frame", 0)),
            priority=int(d.get("priority", 1)),
            event_text=str(d.get("event_text", "")),
            speed_ms=float(d["speed_ms"]) if d.get("speed_ms") is not None else None,
            zone=str(d["zone"]) if d.get("zone") else None,
            player_name=str(d["player_name"]) if d.get("player_name") else None,
            track_id=int(d["track_id"]) if d.get("track_id") is not None else None,
        )


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]

def _default_tdeed_repo_dir() -> Optional[str]:
    p = _repo_root() / "model-training" / "action_spotting" / "T-DEED-main"
    return str(p) if p.exists() else None

def _default_tracking_config() -> Optional[str]:
    p = _repo_root() / "model-training" / "tracking-reid-osnet" / "config.yaml"
    return str(p) if p.exists() else None

def _default_calibration_detector_weights() -> Optional[str]:
    p = _repo_root() / "model-training" / "calibration" / "best.pt"
    return str(p) if p.exists() else None

def _default_calibration_kp_weights() -> Optional[str]:
    p = _repo_root() / "model-training" / "calibration" / "SV_kp.pth"
    return str(p) if p.exists() else None

def _default_calibration_line_weights() -> Optional[str]:
    p = _repo_root() / "model-training" / "calibration" / "SV_lines.pth"
    return str(p) if p.exists() else None

def run_calibration_pipeline(
    *,
    video_path: str,
    out_map: str,
    out_events: str,
    out_frames: Optional[str] = None,
    cfg: Optional[FullPipelineConfig] = None,
    progress_cb: Optional[Callable[[str, int, int, str], None]] = None,
    detection_cache_path: Optional[str] = None,
) -> Dict[str, Any]:

    repo = _repo_root()
    script = repo / "model-training" / "calibration" / "run_pipeline_calibration.py"
    if not script.exists():
        raise FileNotFoundError(str(script))

    det_w = None
    kp_w = None
    line_w = None
    conf = 0.30
    yolo_frame_window = 8
    yolo_selection_mode = "ball_priority"
    interpolation_mode = "linear"
    if cfg is not None:
        try:
            det_w = getattr(cfg, "calibration_detector_weights", None)
            kp_w = getattr(cfg, "calibration_kp_weights", None)
            line_w = getattr(cfg, "calibration_line_weights", None)
            conf = float(getattr(cfg, "calibration_conf_thres", 0.30) or 0.30)
            yolo_frame_window = int(getattr(cfg, "calibration_yolo_frame_window", 0) or 0)
            yolo_selection_mode = str(getattr(cfg, "calibration_yolo_selection_mode", "ball_priority") or "ball_priority")
            interpolation_mode = str(getattr(cfg, "calibration_interpolation_mode", "linear") or "linear")
        except Exception:
            det_w, kp_w, line_w, conf = None, None, None, 0.30
            yolo_frame_window, yolo_selection_mode, interpolation_mode = 8, "ball_priority", "linear"

    det_w = str(det_w).strip() if det_w else (_default_calibration_detector_weights() or "")
    kp_w = str(kp_w).strip() if kp_w else (_default_calibration_kp_weights() or "")
    line_w = str(line_w).strip() if line_w else (_default_calibration_line_weights() or "")

    if not det_w or not os.path.isfile(det_w):
        raise FileNotFoundError(f"calibration detector weights not found: {det_w}")
    if not kp_w or not os.path.isfile(kp_w):
        raise FileNotFoundError(f"calibration kp weights not found: {kp_w}")
    if not line_w or not os.path.isfile(line_w):
        raise FileNotFoundError(f"calibration line weights not found: {line_w}")

    calibration_config = {
        "detector_weights": str(Path(det_w).resolve()),
        "kp_weights": str(Path(kp_w).resolve()),
        "line_weights": str(Path(line_w).resolve()),
        "conf_thres": float(conf),
        "yolo_frame_window": int(max(1, int(yolo_frame_window))),
        "yolo_selection_mode": str(yolo_selection_mode).strip().lower(),
        "interpolation_mode": str(interpolation_mode).strip().lower(),
        "torch_compile": bool(getattr(cfg, "calibration_torch_compile", False)),
        "torch_compile_mode": str(getattr(cfg, "calibration_torch_compile_mode", "reduce-overhead") or "reduce-overhead"),
    }

    cmd: List[str] = [
        sys.executable,
        "-u",
        str(script),
        "--source",
        str(Path(video_path).resolve()),
        "--out_map",
        str(Path(out_map).resolve()),
        "--out_events",
        str(Path(out_events).resolve()),
        "--detector",
        str(Path(det_w).resolve()),
        "--kp_weights",
        str(Path(kp_w).resolve()),
        "--line_weights",
        str(Path(line_w).resolve()),
        "--conf",
        str(float(conf)),
        "--yolo_frame_window",
        str(max(1, int(yolo_frame_window))),
        "--yolo_selection_mode",
        str(yolo_selection_mode).strip().lower(),
        "--interpolation_mode",
        str(interpolation_mode).strip().lower(),
    ]

    if calibration_config.get("torch_compile"):
        cmd += ["--torch_compile", "--torch_compile_mode", str(calibration_config["torch_compile_mode"])]

    frames_stride = 1
    if cfg is not None:
        try:
            frames_stride = int(getattr(cfg, "calibration_frames_stride", 1) or 1)
        except Exception:
            frames_stride = 1
    if frames_stride < 1:
        frames_stride = 1

    calibration_config["frames_stride"] = int(frames_stride)
    calibration_config["write_frames_jsonl"] = bool(out_frames and str(out_frames).strip())

    if out_frames and str(out_frames).strip():
        cmd += ["--out_frames", str(Path(str(out_frames)).resolve()), "--frames_stride", str(int(frames_stride))]

    if detection_cache_path and str(detection_cache_path).strip():
        cmd += ["--detection_cache", str(Path(detection_cache_path).resolve())]

    if progress_cb is not None:
        try:
            progress_cb(
                "calibration",
                0,
                1,
                (
                    "Calibration config: "
                    f"window={calibration_config['yolo_frame_window']}, "
                    f"mode={calibration_config['yolo_selection_mode']}, "
                    f"interp={calibration_config['interpolation_mode']}, "
                    f"conf={calibration_config['conf_thres']:.2f}"
                ),
            )
        except Exception:
            pass

    env = os.environ.copy()
    env.setdefault("PYTHONUNBUFFERED", "1")
    env.setdefault("PYTHONFAULTHANDLER", "1")
    env.setdefault("TORCH_SHOW_CPP_STACKTRACES", "1")
    env.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

    p = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8",
        errors="replace",
        env=env,
        bufsize=1,
        universal_newlines=True,
    )

    last_json: Optional[Dict[str, Any]] = None
    tail: List[str] = []
    try:
        assert p.stdout is not None
        for raw_line in p.stdout:
            line = (raw_line or "").strip()
            if not line:
                continue

            tail.append(line)
            if len(tail) > 200:
                tail = tail[-200:]

            if line.startswith("__PROGRESS__"):
                try:
                    payload = json.loads(line[len("__PROGRESS__") :].strip())
                    stage = str(payload.get("stage") or "calibration")
                    cur = int(payload.get("current") or 0)
                    total = int(payload.get("total") or 0)
                    msg = str(payload.get("message") or "")
                    if progress_cb is not None:
                        try:
                            progress_cb(stage, cur, total, msg)
                        except Exception:
                            pass
                except Exception:
                    continue
                continue

            if line.startswith("{") and "map_video_path" in line and "events_json_path" in line:
                try:
                    last_json = json.loads(line)
                except Exception:
                    pass
    finally:
        try:
            if p.stdout is not None:
                p.stdout.close()
        except Exception:
            pass

    rc = p.wait()
    if rc != 0:
        err = "\n".join(tail[-200:])
        raise RuntimeError(f"calibration subprocess failed (code {rc}):\n{err[-8000:]}")

    if isinstance(last_json, dict):
        last_json["config"] = calibration_config
        return last_json

    return {
        "map_video_path": str(Path(out_map).resolve()),
        "events_json_path": str(Path(out_events).resolve()),
        "config": calibration_config,
        **({"frames_jsonl_path": str(Path(str(out_frames)).resolve())} if out_frames and str(out_frames).strip() else {}),
    }

def _timecode(t: float) -> str:
    if t < 0:
        t = 0
    m = int(t // 60)
    s = int(t % 60)
    return f"{m:02d}:{s:02d}"

def _event_desc_tr(label: str) -> str:
    mapping = {
        "Kick-off": "Santra yapıldı, maç başladı.",
        "Throw-in": "Taç atışı kullanıldı.",
        "Goal": "Gol oldu!",
        "Corner": "Korner kullanılıyor.",
        "Free-kick": "Serbest vuruş.",
        "Penalty": "Penaltı!",
        "Offside": "Ofsayt bayrağı kalktı.",
        "Foul": "Faul düdüğü çaldı.",
        "Yellow card": "Sarı kart çıktı.",
        "Red card": "Kırmızı kart!",
        "Substitution": "Oyuncu değişikliği var.",
        "Clearance": "Savunmadan uzaklaştırma.",
        "Shot": "Şut!",
        "Save": "Kaleci kurtardı.",
        "Ball out of play": "Top oyun alanını terk etti.",
        "Direct free-kick": "Direkt serbest vuruş.",
        "Indirect free-kick": "Endirekt serbest vuruş.",
        "Pass": "Pas atıldı.",
        "Drive": "Top sürüldü.",
        "Header": "Kafa vuruşu.",
        "High pass": "Uzun pas atıldı.",
        "Free-kick": "Serbest vuruş.",
        "Shot": "Şut çekildi!",
        "Ball player block": "Oyuncu topu kesti.",
    }
    return mapping.get(label, f"Olay: {label}")

def _clamp_int(v: float, lo: int, hi: int) -> int:
    try:
        iv = int(round(float(v)))
    except Exception:
        iv = int(lo)
    return max(int(lo), min(int(hi), iv))

def _is_special_track_id(track_id: int) -> bool:
    try:
        return int(track_id) >= 800_000_000
    except Exception:
        return False

# Per-event-type offset (seconds) between T-DEED detection time and when speech starts.
# Negative = speak BEFORE the event (anticipation), Positive = speak AFTER the event.
# Goal/own-goal/penalty-goal: start 1s early so voice lands right as the goal is scored.
_EVENT_SPEECH_OFFSET: Dict[str, float] = {
    "goal": +3.75,        # golsesi.wav (5.74s) event_t-2s'de başlar → event_t+3.74'te biter → yorum orada başlar
    "own goal": +3.75,
    "penalty - goal": +3.75,
    "red card": 0.5,
    "penalty": 0.5,
    "var": 0.5,
    "yellow card": 0.3,
}

# Per-event-type commentary guidance injected into the prompt so the LLM knows
# the emotional register and what to focus on for each event type.
_EVENT_COMMENTARY_GUIDANCE: Dict[str, str] = {
    "goal": "GOOOOL! Yorumun ilk kelimesi MUTLAKA 'GOOOOL!' olmalı. Maksimum heyecan, ses tonu dorukta, dramatik ve coşkulu!",
    "own goal": "Kendi kalesine gol — şaşkınlık ve talihsizlik vurgusu yap.",
    "penalty - goal": "Penaltıdan gol — bire bir kazanıldı, heyecanı vurgula.",
    "kick-off": "Yeni oyun fazı başlıyor; sahaya dönüşü ve tempo değişikliğini anlat.",
    "corner": "Korner: ceza sahasında kaos potansiyeli, bekleyiş ve tehlike var.",
    "free-kick": "Serbest vuruş: duran top organizasyonu, tehlike bekleniyor.",
    "direct free-kick": "Doğrudan serbest vuruş: tehlikeli pozisyon, kaleye odaklan.",
    "indirect free-kick": "Endirekt serbest vuruş: kombinasyon oyunu aranıyor.",
    "penalty": "PENALTİ! Gerilim dorukta, bire bir an yaklaşıyor.",
    "yellow card": "Sarı kart çıktı — disiplin uyarısı, oyuncu dikkatli olmak zorunda.",
    "red card": "KIRMIZI KART! Takım 10 kişi kaldı, oyunun dengesi değişti.",
    "offside": "Ofsayt: atak boşa çıktı, savunma kapanı işe yaradı.",
    "foul": "Faul: oyun kesildi, gerilim yükseliyor.",
    "substitution": "Oyuncu değişikliği: teknik ekip yeni bir kart oynuyor.",
    "throw-in": "Taç atışı: top saha dışına çıktı, taçla oyun yeniden başlıyor. ÖNEMLI: 'Topu dışarı/taca gönderen' oyuncu PAS ATMADI — topu saha dışına çıkardı. 'Pas attı' YASAK. Bunun yerine 'uzaklaştırdı', 'taca gönderdi', 'top dışarı çıktı', 'taca attı' de.",
    "ball out of play": "Top oyun dışında: kısa solukluk, ritim yeniden kuruluyor. 'Topu dışarı/taca gönderen' oyuncu PAS ATMADI, topu saha dışına çıkardı.",
    "clearance": "Savunma uzaklaştırması: tehlike savuşturuldu, top hâlâ aktif.",
    "shot": "Şut: kaleye tehlikeli girişim! Pozisyonu ve yönü anlat.",
    "shot on target": "Kaleye isabet: kaleciye iş düştü, kritik an!",
    "save": "Kaleci kurtardı! Refleksi ve pozisyonu öne çıkar.",
    "header": "Kafa vuruşu: hava topunda kritik mücadele.",
    "pass": "Pas hareketi: top akıyor, hangi yönde ilerleniyor anlat.",
    "drive": "Top sürülüyor: bireysel hamle, alan kazanımı.",
    "high pass": "Uzun pas atıldı: derinlik arayışı, kanat açılımı. 'Pas atan' oyuncunun adını mutlaka söyle ve topu kime/nereye attığını anlat.",
    "pass": "Pas hareketi: top akıyor. 'Pas atan' oyuncunun adını ve pas yönünü mutlaka anlat.",
    "ball player block": "Müdahale: pas yolu kesildi, dirençli savunma.",
    "var": "VAR incelemesi: hakem teknolojiyle karar veriyor, gerilim sürüyor.",
}

def _strip_think_blocks(text: str) -> str:
    s = re.sub(r"<think>[\s\S]*?</think>", "", str(text or ""), flags=re.IGNORECASE)
    # Kapanmamış <think> bloğunu da temizle (model max_tokens'a takıldıysa)
    s = re.sub(r"<think>[\s\S]*$", "", s, flags=re.IGNORECASE)
    return s.strip()

_COMMENTARY_SYSTEM_PROMPT = (
    "Sen deneyimli bir Türkçe canlı futbol spikerisin. "
    "Görevin: sana verilen maç verilerini (olay tipi, sahada top konumu, baskı seviyesi, "
    "forma numaraları, önceki olaylar) analiz edip önce İngilizce olarak sahada ne olduğunu "
    "mantıksal olarak çıkarsamak, ardından bu çıkarsamayı heyecanlı, akıcı, özgün Türkçe bir "
    "spiker cümlesi hâline getirmek. "
    "Düşünce sürecin (İngilizce analiz, alternatif anlatım seçenekleri, dramatik yapı kararı) "
    "<think>...</think> bloğunda saklı kalacak; "
    "nihai çıktı YALNIZCA Türkçe JSON olacak: {\"text\": \"...\"}. "
    "Veri olmayan oyuncu veya takım bilgisi uydurma. "
    "Cümle tekrarından kaçın; her olay için yeni ve farklı bir anlatım tonu seç.\n"
    "OYUNCU ADLARI (KRİTİK): Aktör bölümünde oyuncu adı verilmişse MUTLAKA o adı söyle — "
    "'bir oyuncu' veya 'o oyuncu' deme. "
    "Yalnızca forma numarası varsa 'X numaralı oyuncu' de (rakam olarak değil, sözcükle: 'yedi numaralı'). "
    "Pas atan, topu alan veya şut çeken oyuncunun adı biliniyorsa her zaman cümlede geç.\n"
    "YORUM UZUNLUĞU: Gol ve kritik anlarda 2-3 cümle; normal oyun akışında (pas, top taşıma, bölge geçişi) "
    "maksimum 1-2 kısa cümle yeterli. Gereksiz dolgu cümlesi ekleme.\n"
    "FUTBOL DİLİ: 'top ileri taşınıyor', 'topu ilerletiyor', 'topu taşıyor' gibi GENEL ifadeler KULLANMA. "
    "Bunun yerine gerçek futbol söylemlerini kullan: 'hücuma çıkıyor', 'rakip yarı sahaya giriyor', "
    "'savunmayı geçmeye çalışıyor', 'atağı başlatıyor' gibi sahneye özgü ifadeler seç.\n"
    "KRITIK: Üretilen metin doğrudan TTS (metinden sese) sistemi tarafından okunacaktır. "
    "Bu nedenle:\n"
    "- '...', '(...)' gibi söylenmesi güç ifadeler KULLANMA.\n"
    "- Parantez içi açıklamalar, köşeli parantezler, tire ardına getirilen notlar YASAK.\n"
    "- Sadece doğal konuşma dili; her kelime yüksek sesle okunabilir olmalı.\n"
    "- Kısaltma ve sembol kullanma ('%', '&', '+', '#' vb. harf/sözcükle yaz).\n"
    "- Cevap kesinlikle sadece JSON olmalı: {\"text\": \"BURADA YORUM\"}"
)

def _qwen_text_openai_compatible(
    *,
    base_url: str,
    model: str,
    prompt: str,
    timeout_sec: float = 60.0,
    enable_thinking: bool = True,
    max_tokens: int = 1800,
) -> Tuple[Optional[str], Optional[str]]:
    if httpx is None:
        return None, None

    base = _normalize_base_url(base_url)
    if not base:
        return None, None

    endpoint = base + "/v1/chat/completions"
    payload: Dict[str, Any] = {
        "model": str(model or "qwen3vl8b"),
        "messages": [
            {
                "role": "system",
                "content": _COMMENTARY_SYSTEM_PROMPT,
            },
            {
                "role": "user",
                "content": str(prompt or ""),
            },
        ],
        "temperature": 0.72,
        "max_tokens": max_tokens,
        "chat_template_kwargs": {"enable_thinking": bool(enable_thinking)},
    }

    try:
        with httpx.Client(timeout=float(timeout_sec)) as client:
            r = client.post(endpoint, json=payload)
            r.raise_for_status()
            data = r.json()
    except Exception as e:
        return None, str(e)

    raw = ""
    try:
        choices = data.get("choices") or []
        msg = (choices[0] or {}).get("message") or {}
        raw = str(msg.get("content") or "").strip()
    except Exception:
        raw = ""

    # Her zaman think bloklarını temizle; nihai çıktı yorum JSON'u
    raw = _strip_think_blocks(raw)

    return raw, None

def _normalize_commentary_backend(name: str) -> str:
    raw = str(name or "").strip().lower()
    if raw in ("vllm", "vllm-openai", ""):
        return "vllm"
    if raw in ("openai", "openai-compatible", "compat", "llama.cpp"):
        return "openai"
    return raw

def _request_commentary_text(
    *,
    base_url: str,
    model: str,
    prompt: str,
    backend: str,
    timeout_sec: float = 120.0,
    enable_thinking: bool = True,
    max_tokens: int = 1800,
    timecode: Optional[str] = None,
    emit_cb: Optional[Any] = None,
) -> Tuple[Optional[str], Optional[str]]:
    raw, err = _qwen_text_openai_compatible(
        base_url=base_url,
        model=model,
        prompt=prompt,
        timeout_sec=timeout_sec,
        enable_thinking=enable_thinking,
        max_tokens=max_tokens,
    )

    if err and emit_cb is not None:
        try:
            emit_cb(
                "commentary_llm_error",
                0,
                1,
                f"[{timecode or '??:??'}] vLLM hata: {err[:200]}",
            )
        except Exception:
            pass

    return raw, err

async def _request_commentary_batch_async(
    *,
    prompts: List[str],
    endpoint: str,
    payload_template: Dict[str, Any],
    timeout_sec: float,
    enable_thinking: bool,
    batch_size: int,
) -> List[Tuple[Optional[str], Optional[str]]]:
    import asyncio as _asyncio

    if httpx is None:
        return [(None, "httpx not available")] * len(prompts)

    async def _call_one(session: Any, prompt: str) -> Tuple[Optional[str], Optional[str]]:
        payload = dict(payload_template)
        msgs = [dict(m) for m in payload["messages"]]
        msgs[-1] = dict(msgs[-1], content=str(prompt))
        payload["messages"] = msgs
        try:
            r = await session.post(endpoint, json=payload)
            r.raise_for_status()
            data = r.json()
        except Exception as e:
            return None, str(e)
        raw = ""
        try:
            choices = data.get("choices") or []
            msg = (choices[0] or {}).get("message") or {}
            raw = str(msg.get("content") or "").strip()
        except Exception:
            raw = ""
        # Her zaman think bloklarını temizle
        raw = _strip_think_blocks(raw)
        return raw or None, None

    results: List[Tuple[Optional[str], Optional[str]]] = []
    async with httpx.AsyncClient(timeout=float(timeout_sec)) as session:
        for chunk_start in range(0, len(prompts), batch_size):
            chunk = prompts[chunk_start : chunk_start + batch_size]
            chunk_results = await _asyncio.gather(
                *[_call_one(session, p) for p in chunk],
                return_exceptions=True,
            )
            for res in chunk_results:
                if isinstance(res, BaseException):
                    results.append((None, str(res)))
                else:
                    results.append(res)
    return results

def _request_commentary_batch(
    *,
    prompts: List[str],
    base_url: str,
    model: str,
    backend: str,
    timeout_sec: float = 120.0,
    enable_thinking: bool = True,
    max_tokens: int = 1800,
    batch_size: int = 4,
    emit_cb: Optional[Any] = None,
    timecodes: Optional[List[str]] = None,
) -> List[Tuple[Optional[str], Optional[str]]]:
    import asyncio as _asyncio

    _mode = _normalize_commentary_backend(backend)

    if httpx is None:
        return [
            _request_commentary_text(
                base_url=base_url,
                model=model,
                prompt=p,
                backend=backend,
                timeout_sec=timeout_sec,
                enable_thinking=enable_thinking,
                max_tokens=max_tokens,
                timecode=str((timecodes or [])[i]) if timecodes and i < len(timecodes) else None,
                emit_cb=emit_cb,
            )
            for i, p in enumerate(prompts)
        ]

    base = _normalize_base_url(base_url)
    if not base:
        return [(None, "invalid base_url")] * len(prompts)

    payload_template: Dict[str, Any] = {
        "model": str(model),
        "messages": [
            {"role": "system", "content": _COMMENTARY_SYSTEM_PROMPT},
            {"role": "user", "content": ""},
        ],
        "temperature": 0.72,
        "max_tokens": max_tokens,
        "chat_template_kwargs": {"enable_thinking": bool(enable_thinking)},
    }

    try:
        loop = _asyncio.get_event_loop()
        if loop.is_running():
            import concurrent.futures as _cf
            fut = _cf.Future()
            async def _run():
                try:
                    fut.set_result(await _request_commentary_batch_async(
                        prompts=prompts,
                        endpoint=base + "/v1/chat/completions",
                        payload_template=payload_template,
                        timeout_sec=timeout_sec,
                        enable_thinking=enable_thinking,
                        batch_size=batch_size,
                    ))
                except Exception as exc:
                    fut.set_exception(exc)
            _asyncio.ensure_future(_run())
            results = fut.result(timeout=timeout_sec * len(prompts))
        else:
            results = loop.run_until_complete(
                _request_commentary_batch_async(
                    prompts=prompts,
                    endpoint=base + "/v1/chat/completions",
                    payload_template=payload_template,
                    timeout_sec=timeout_sec,
                    enable_thinking=enable_thinking,
                    batch_size=batch_size,
                )
            )
    except RuntimeError:
        results = _asyncio.run(
            _request_commentary_batch_async(
                prompts=prompts,
                endpoint=base + "/v1/chat/completions",
                payload_template=payload_template,
                timeout_sec=timeout_sec,
                enable_thinking=enable_thinking,
                batch_size=batch_size,
            )
        )

    if emit_cb is not None:
        for i, (raw, err) in enumerate(results):
            if err:
                tc = str((timecodes or [])[i]) if timecodes and i < len(timecodes) else str(i)
                try:
                    emit_cb(
                        "commentary_llm_error",
                        0,
                        1,
                        f"[{tc}] vLLM batch hata (idx={i}): {err[:200]}",
                    )
                except Exception:
                    pass

    return results

def _flush_gpu_vram() -> Dict[str, Any]:
    info: Dict[str, Any] = {
        "gc_collected": 0,
        "torch_available": False,
        "cuda_available": False,
        "empty_cache_called": False,
        "ipc_collect_called": False,
    }
    started = time.perf_counter()

    try:
        info["gc_collected"] = int(gc.collect())
    except Exception as e:
        info["gc_error"] = str(e)

    try:
        import torch  # type: ignore

        info["torch_available"] = True
        cuda_ok = bool(torch.cuda.is_available())
        info["cuda_available"] = cuda_ok
        if cuda_ok:
            try:
                if hasattr(torch.cuda, "mem_get_info"):
                    free_before, total_before = torch.cuda.mem_get_info()
                    info["free_bytes_before"] = int(free_before)
                    info["total_bytes"] = int(total_before)
            except Exception:
                pass

            torch.cuda.empty_cache()
            info["empty_cache_called"] = True

            if hasattr(torch.cuda, "ipc_collect"):
                try:
                    torch.cuda.ipc_collect()
                    info["ipc_collect_called"] = True
                except Exception as e:
                    info["ipc_collect_error"] = str(e)

            try:
                if hasattr(torch.cuda, "mem_get_info"):
                    free_after, _total_after = torch.cuda.mem_get_info()
                    info["free_bytes_after"] = int(free_after)
            except Exception:
                pass
    except Exception as e:
        info["torch_error"] = str(e)

    info["elapsed_sec"] = round(time.perf_counter() - started, 3)
    return info

def _build_commentary_prompt(items: List[Dict[str, Any]]) -> str:
    return (
        "Sen güçlü bir futbol maç spikerisin. Aşağıda her olay için zaman penceresi boyunca toplanmış maç durumu verisi var.\n"
        "Her olay için en fazla 2 cümlelik, doğal, akıcı ve sahadaki oyunun akışını anlatan Türkçe yorum yaz.\n"
        "Kurallar:\n"
        "- Sadece Türkçe yaz.\n"
        "- Aynı zaman damgasını koru.\n"
        "- Oyuncu bilgisi varsa (#track_id, forma_no, takım_id) onu kullan.\n"
        "- Kalibrasyon çerçevelerindeki top konumu, baskı seviyesi, kanat/merkez geçişi, ceza sahası çevresi, topun yönü gibi bağlamı kullan.\n"
        "- Pencere içindeki possession ve calibration event bilgisini birlikte değerlendir.\n"
        "- Veri açıkça desteklemiyorsa kesin hüküm verme; belirsizse genel ama maç akışına uygun konuş.\n"
        "- ÇIKTI FORMAT: Sadece JSON array döndür.\n"
        "JSON şeması: [{\"t\": <float saniye>, \"timecode\": \"MM:SS\", \"text\": \"...\"}]\n\n"
        "Her olay girdisinde şunlar bulunabilir: olay etiketi, aktör bilgisi, zaman penceresi, kalibrasyon örnek kareleri, topun saha içi akışı, baskı özeti, pencere içi olaylar.\n\n"
        "Olaylar JSON:\n"
        + json.dumps(items, ensure_ascii=False, indent=2)
    )

def _commentary_style_for_item(item: Dict[str, Any]) -> str:
    styles = [
        "yüksek tempolu ve coşkulu anlatım",
        "taktik odaklı ama akıcı anlatım",
        "kanat oyunu ve alan kullanımı odaklı anlatım",
        "orta saha mücadelesi ve baskı odaklı anlatım",
        "ceza sahası çevresi tehdidini öne çıkaran anlatım",
    ]
    seed = f"{item.get('timecode')}|{item.get('event_label')}|{item.get('event_source')}"
    try:
        idx = sum(ord(ch) for ch in seed) % len(styles)
    except Exception:
        idx = 0
    return styles[idx]

# Oyuncu adını söylemek yeterli olan düşük öncelikli event engine olayları
_NAME_ONLY_ENGINE_LABELS: FrozenSet[str] = frozenset({
    "possession_change", "zone_change", "ball_carry", "sprint",
})

def _try_name_only_text(item: Dict[str, Any]) -> Optional[str]:
    """For low-priority engine events with a known player name, return just the name without LLM."""
    label = str(item.get("event_label") or "").strip().lower()
    if label not in _NAME_ONLY_ENGINE_LABELS:
        return None
    actor_info = item.get("actor_info") or {}
    actor = actor_info.get("actor") or {}
    name = str(actor.get("player_name") or "").strip()
    if not name:
        return None
    return name


def _build_commentary_item_prompt(item: Dict[str, Any], recent_texts: List[str], match_context: Optional[Dict[str, Any]] = None) -> str:
    window = item.get("window") if isinstance(item, dict) else None
    window = window if isinstance(window, dict) else {}
    period_sec = float(window.get("duration_sec", item.get("segment_duration_sec", 10.0)) or 10.0)

    event_label = str(item.get("event_label") or "").strip()
    event_confidence = float(item.get("event_confidence") or 0.0)
    timecode = str(item.get("event_timecode") or item.get("timecode") or "").strip()
    label_lower = event_label.lower()

    max_sentences = _commentary_sentence_budget(period_sec, event_label)
    max_words = _commentary_word_budget(period_sec, event_label)

    # ── team_id → takım adı çözümleyici (bu fonksiyon boyunca kullanılır) ─
    _ctx = match_context or (item.get("match_context") if isinstance(item, dict) else None)
    _ctx = _ctx if isinstance(_ctx, dict) else {}
    _team_names_ctx: Dict[str, str] = _ctx.get("team_names") or {}

    def _tn(team_id: Any) -> str:
        """team_id (int/str/None) → 'Galatasaray' gibi gerçek isim; bilinmiyorsa boş string."""
        if team_id is None:
            return ""
        return _team_names_ctx.get(str(team_id), "")

    # Roster cross-validation için isim→takım_id haritası
    # NOT: rosters key'leri takım ismi ("Galatasaray"...) — team_names ise "0"/"1"
    _team_name_to_id_ctx: Dict[str, str] = {v: k for k, v in _team_names_ctx.items()}
    _name_to_roster_team_ctx: Dict[str, str] = {}
    for _rt_ctx, _rps_ctx in (_ctx.get("rosters") or {}).items():
        _rt_ctx_id = _team_name_to_id_ctx.get(str(_rt_ctx), str(_rt_ctx))
        for _rp_ctx in (_rps_ctx if isinstance(_rps_ctx, list) else []):
            if isinstance(_rp_ctx, dict) and _rp_ctx.get("name"):
                _name_to_roster_team_ctx[str(_rp_ctx["name"])] = _rt_ctx_id

    sections: List[str] = []

    # ── MAÇ BAĞLAMI ───────────────────────────────────────────────────────
    # commentary_input.json'dan veya parametre olarak gelen match_context
    if _ctx:
        ctx_lines: List[str] = []
        if _ctx.get("teams"):
            ctx_lines.append(f"Maç: {_ctx['teams']}")
        if _ctx.get("competition"):
            ctx_lines.append(f"Turnuva: {_ctx['competition']}")
        if _ctx.get("date"):
            ctx_lines.append(f"Tarih: {_ctx['date']}")
        _team_names = _ctx.get("team_names") or {}
        if _team_names:
            tn_parts = [f"Takım {k} = {v}" for k, v in sorted(_team_names.items())]
            ctx_lines.append("Takım kimlikleri: " + ", ".join(tn_parts))
        _rosters = _ctx.get("rosters") or {}
        for team_label, players in _rosters.items():
            if not isinstance(players, list):
                continue
            player_parts = []
            for p in players:
                if not isinstance(p, dict):
                    continue
                name = str(p.get("name") or "").strip()
                num = p.get("number")
                if name and num is not None:
                    player_parts.append(f"#{num} {name}")
            if player_parts:
                ctx_lines.append(f"{team_label} kadrosu: {', '.join(player_parts)}")
        if ctx_lines:
            sections.append("## MAÇ BAĞLAMI\n" + "\n".join(ctx_lines))

    # ── OLAY ──────────────────────────────────────────────────────────────
    conf_str = f"  (güven %{round(event_confidence * 100)})" if event_confidence > 0 else ""
    event_line = f"Tür: {event_label}{conf_str}"
    if timecode:
        event_line += f"  |  Dakika: {timecode}"
    guidance = _EVENT_COMMENTARY_GUIDANCE.get(label_lower, "")
    event_block = "## OLAY\n" + event_line
    if guidance:
        event_block += f"\nYönlendirme: {guidance}"
    sections.append(event_block)

    # ── GOL ÖNCESİ ŞUT (gol olaylarında şut-gol zinciri) ────────────────
    pre_shot = item.get("pre_shot_event") if isinstance(item, dict) else None
    if pre_shot and label_lower in ("goal", "own goal", "penalty - goal"):
        _shot_lbl = str(pre_shot.get("label") or "Şut").strip()
        _shot_tc = str(pre_shot.get("timecode") or "").strip()
        _shot_dist = float(pre_shot.get("distance_to_goal_sec") or 0.0)
        _shot_conf = float(pre_shot.get("confidence") or 0.0)
        _shot_conf_str = f" (güven %{round(_shot_conf * 100)})" if _shot_conf > 0 else ""
        sections.append(
            f"## GOL ÖNCESİ HAREKETE\n"
            f"Bu golün {round(_shot_dist, 1)} saniye öncesinde '{_shot_lbl}' tespit edildi"
            + (f" ({_shot_tc})" if _shot_tc else "") + f".{_shot_conf_str}\n"
            "Yorumun GOL anlatımından önce bu şutu da mutlaka kapsamalı: "
            "'Şutu attı ve ağları havalandırdı!' gibi bir akış kur."
        )

    # ── AKTÖR (top sahibi ve önceki top sahibi) ───────────────────────────
    actor_info = item.get("actor_info") if isinstance(item, dict) else None
    actor_info = actor_info if isinstance(actor_info, dict) else {}
    actor = actor_info.get("actor") if isinstance(actor_info, dict) else None
    from_actor = actor_info.get("from_actor") if isinstance(actor_info, dict) else None
    if actor or from_actor:
        actor_lines: List[str] = []
        if actor:
            a_parts = []
            if actor.get("player_name"):
                a_parts.append(str(actor["player_name"]))
            if actor.get("jersey_number") and str(actor["jersey_number"]) not in ("-1", ""):
                a_parts.append(f"#{actor['jersey_number']}")
            _a_team = _tn(actor.get("team_id"))
            if _a_team:
                a_parts.append(f"({_a_team})")
            elif actor.get("team_id") is not None:
                a_parts.append(f"(Takım {actor['team_id']})")
            if a_parts:
                actor_lines.append("Top sahibi: " + " ".join(a_parts))
        if from_actor:
            f_parts = []
            if from_actor.get("player_name"):
                f_parts.append(str(from_actor["player_name"]))
            if from_actor.get("jersey_number") and str(from_actor["jersey_number"]) not in ("-1", ""):
                f_parts.append(f"#{from_actor['jersey_number']}")
            _f_team = _tn(from_actor.get("team_id"))
            if _f_team:
                f_parts.append(f"({_f_team})")
            elif from_actor.get("team_id") is not None:
                f_parts.append(f"(Takım {from_actor['team_id']})")
            if f_parts:
                # Taç/dışarı olaylarında from_actor topu taca atan kişidir, pas atan değil
                # → LLM'in "pas attı" demesini önlemek için olaya özel etiket kullan
                if label_lower in ("throw-in", "ball out of play", "clearance"):
                    _from_label = "Topu dışarı/taca gönderen"
                elif label_lower in ("high pass", "pass"):
                    _from_label = "Pas atan"
                else:
                    _from_label = "Önceki top sahibi"
                actor_lines.append(f"{_from_label}: " + " ".join(f_parts))
        if actor_lines:
            actor_section = "## AKTÖR\n" + "\n".join(actor_lines)
            # LLM'e açık direktif: isim biliniyorsa mutlaka söyle
            has_name = bool(
                (actor and actor.get("player_name")) or
                (from_actor and from_actor.get("player_name"))
            )
            if has_name:
                actor_section += "\n→ Yorumda bu oyuncunun ADINI MUTLAKA söyle."
            sections.append(actor_section)

    # ── SAHADAKİ TABLO (kalibrasyon özeti) ────────────────────────────────
    _flow_note = ""
    match_state = item.get("match_state") if isinstance(item, dict) else None
    match_state = match_state if isinstance(match_state, dict) else {}
    _frame_samples_all = list(match_state.get("frame_samples") or [])
    if _frame_samples_all:
        _fs_n = len(_frame_samples_all)
        _fs_ball = sum(1 for _f in _frame_samples_all if _f.get("ball") is not None)
        _fs_ppl = sum(1 for _f in _frame_samples_all if int(_f.get("player_count") or 0) > 2)
        if _fs_n > 0 and _fs_ball / _fs_n < 0.25 and _fs_ppl / _fs_n < 0.3:
            _flow_note = "Oyun duraksadı; top ve oyuncu tespiti yok — muhtemelen oyun kesildi."
        elif _fs_n > 0 and _fs_ball / _fs_n < 0.45:
            _flow_note = "Oyun ritmi yavaş; top hareketleri sınırlı."
    state_summary = match_state.get("state_summary") if match_state else None
    state_summary = state_summary if isinstance(state_summary, dict) else {}
    calib_win = match_state.get("window") if match_state else None
    calib_win = calib_win if isinstance(calib_win, dict) else {}
    if state_summary or _flow_note:
        w_start = str(calib_win.get("start_timecode", "")).strip()
        w_end = str(calib_win.get("end_timecode", "")).strip()
        tablo_header = "## SAHADAKİ TABLO" + (f" ({w_start}–{w_end})" if w_start and w_end else "")
        tablo_lines: List[str] = []

        if _flow_note:
            tablo_lines.append(f"- Oyun akışı: {_flow_note}")

        progression = str((state_summary or {}).get("ball_progression") or "").strip()
        if progression and "kalibrasyon kare" not in progression:
            tablo_lines.append(f"- Top hareketi: {progression}")

        pressure = str((state_summary or {}).get("pressure_level") or "").strip()
        if pressure and pressure != "bilinmiyor":
            avg_p = (state_summary or {}).get("avg_nearby_pressure_count")
            avg_str = f" ({avg_p:.1f} yakın oyuncu)" if isinstance(avg_p, (int, float)) else ""
            tablo_lines.append(f"- Baskı: {pressure}{avg_str}")

        tags = [str(t) for t in ((state_summary or {}).get("state_tags") or []) if str(t).strip()]
        if tags:
            tablo_lines.append(f"- Bölge/durum: {', '.join(tags)}")

        focus = (state_summary or {}).get("focus_players") or []
        if focus:
            fp_parts = []
            for fp in focus[:3]:
                j = fp.get("jersey_number")
                tid = fp.get("team_id")
                dist = fp.get("distance_to_ball_m")
                pname = fp.get("player_name")
                tname = fp.get("team_name")
                fp_str_parts = []
                if pname:
                    fp_str_parts.append(str(pname))
                elif j is not None and str(j) not in ("-1", ""):
                    fp_str_parts.append(f"#{j}")
                # tname: enrich_item'dan geliyorsa doğrudan kullan, yoksa _tn() ile çöz
                _resolved_tname = tname or _tn(tid)
                if _resolved_tname:
                    fp_str_parts.append(f"({_resolved_tname})")
                elif tid is not None:
                    fp_str_parts.append(f"(Takım {tid})")
                if dist is not None:
                    fp_str_parts.append(f"{dist}m")
                if fp_str_parts:
                    fp_parts_display = ", ".join(fp_str_parts)
                    fp_parts.append(fp_parts_display)
                    # Kameradaki oyuncuları _known_player_names'e de ekle
                    if pname and str(pname).strip() and str(pname).strip() not in _known_player_names:
                        _known_player_names.append(str(pname).strip())
            if fp_parts:
                tablo_lines.append(f"- Topa yakın: {'; '.join(fp_parts)}")

        if tablo_lines:
            sections.append(tablo_header + "\n" + "\n".join(tablo_lines))

    # ── EVENT ENGINE CONTEXT (aksiyonlar + oyuncu hareketleri) ────────────
    # ── EVENT ENGINE CONTEXT (aksiyonlar + oyuncu hareketleri) ──────────────
    event_engine_ctx = item.get("event_engine_context") if isinstance(item, dict) else None
    event_engine_ctx = event_engine_ctx if isinstance(event_engine_ctx, list) else []
    _known_player_names: List[str] = []
    _known_scoring_team: str = ""   # en yüksek öncelikli event'in takım adı

    # actor_info'dan isimleri önceden topla (event_engine_context olmasa da çalışsın)
    _ai = item.get("actor_info") if isinstance(item, dict) else {}
    _ai = _ai if isinstance(_ai, dict) else {}
    for _ai_key in ("actor", "from_actor"):
        _ai_p = _ai.get(_ai_key)
        if isinstance(_ai_p, dict) and _ai_p.get("player_name"):
            _pn = str(_ai_p["player_name"]).strip()
            if _pn and _pn not in _known_player_names:
                _known_player_names.append(_pn)
        if isinstance(_ai_p, dict) and not _known_scoring_team:
            _known_scoring_team = _tn(_ai_p.get("team_id")) or _known_scoring_team

    if event_engine_ctx:
        ee_lines: List[str] = []
        # Throw-in / ball out of play: engine'deki "pas" aslında topu saha dışına çıkaran harekettir
        _is_throw_or_out = label_lower in ("throw-in", "ball out of play")
        _ee_header_note = (
            "NOT: Aşağıdaki 'pas' hareketi topu SAHA DIŞINA çıkardı — bu bir pas değil taç/dışarı hareketi.\n"
            if _is_throw_or_out else ""
        )
        for ec in sorted(event_engine_ctx, key=lambda x: -(x.get("priority") or 0))[:4]:
            ec_text = str(ec.get("text") or "").strip()
            if not ec_text:
                continue
            # Throw-in / ball out of play: engine "pas" kelimesini → "dışarı/taca gönderme" ile değiştir
            if _is_throw_or_out:
                ec_text = re.sub(r"uzun bir pas", "uzun top (saha dışına çıktı)", ec_text)
                ec_text = re.sub(r"kısa bir pas", "kısa bir hareket (saha dışına çıktı)", ec_text)
                ec_text = re.sub(r"kısa bir yerden pas", "kısa bir hareket (top dışarı çıktı)", ec_text)
                ec_text = re.sub(r"\bpas\b", "dışarı gönderme", ec_text)
            pname = ec.get("player_name")
            suffix = f" [{pname}]" if pname else ""
            ee_lines.append(f"- {ec_text}{suffix}")
            if pname:
                if pname not in _known_player_names:
                    _known_player_names.append(str(pname))
            # player_name yoksa team_id'den takım adını al
            if not _known_scoring_team:
                _ec_team = _tn(ec.get("team_id"))
                if _ec_team:
                    _known_scoring_team = _ec_team
        if ee_lines:
            sections.append("## AKSİYON\n" + _ee_header_note + "\n".join(ee_lines))

    # ── SON OLAYLAR (narratif bağlam) ─────────────────────────────────────
    nearby_events = item.get("nearby_events") if isinstance(item, dict) else None
    nearby_events = nearby_events if isinstance(nearby_events, list) else []
    if nearby_events:
        ne_lines = [
            f"- {ne.get('timecode', '')}: {ne.get('label', '')}"
            for ne in nearby_events
            if ne.get("label")
        ]
        if ne_lines:
            sections.append("## SON OLAYLAR (önceki 15s)\n" + "\n".join(ne_lines))

    # ── ÖNCEKİ YORUMLAR (tekrar engeli) ──────────────────────────────────
    recent_valid = [t for t in recent_texts[-3:] if t]
    if recent_valid:
        sections.append(
            "## ÖNCEKİ YORUMLAR (aynı ifadeleri tekrar etme)\n"
            + "\n".join(f"- {t}" for t in recent_valid)
        )

    # ── GÖREV & KURALLAR ─────────────────────────────────────────────────
    _goal_rule = (
        "- Yorumun MUTLAKA 'GOOOOL!' veya 'GOOOL!' ile başlaması ZORUNLUDUR.\n"
        if label_lower in ("goal", "own goal", "penalty - goal") else ""
    )
    _is_pause = label_lower in ("ball out of play",) or str(item.get("event_source") or "") == "pause_detector"
    _pause_duration = float(item.get("pause_duration_sec") or 0.0)

    if _is_pause:
        _pause_step3 = (
            f"3. Top {round(_pause_duration, 0):.0f} saniyedir görülmüyor. "
            "Bu sessizliği kullanan, seyirciye nefes aldıran, bir sonraki aksiyonu meraklandıran bir köprü cümlesi yaz.\n"
        )
        _pause_step4 = "4. Kısa, sakin ama beklenti yaratan 1 cümle formüle et.\n"
    else:
        _pause_step3 = "3. Önceki yorumlarla çakışmayacak, farklı bir duygusal ton ve giriş kelimesi belirle.\n"
        _pause_step4 = "4. En güçlü 1–2 cümlelik anlatım seçeneğini formüle et.\n"

    sections.append(
        "## GÖREV\n"
        "Sen canlı yayındaki bir futbol yorumcususun.\n\n"
        "**Düşünce adımların (think bloğunda yap, Türkçe veya İngilizce):**\n"
        "1. Olayın sahada ne anlama geldiğini analiz et (konum, baskı, momentum).\n"
        "2. Bu olayın maç ritmini nasıl etkilediğini değerlendir.\n"
        + _pause_step3
        + _pause_step4
        + "\n**Çıktı kuralları:**\n"
        "- SADECE şu JSON'u döndür: {\"text\": \"YORUM\"}\n"
        f"- En fazla {max_sentences} cümle, en fazla {max_words} kelime.\n"
        f"{_goal_rule}"
        "- Çıktı YALNIZCA Türkçe olmalı.\n"
        "- 'olabilir', 'belki', 'muhtemelen' gibi belirsizlik ifadeleri kullanma.\n"
        + (
            f"- NOT: Şu oyuncu adlarını yorumunda KULLANABİLİRSİN (bunlar gerçek veri, uydurma değil): "
            f"{', '.join(_known_player_names)}\n"
            if _known_player_names else ""
        )
        + "- Forma numarası veya oyuncu ismi UYDURMA; yalnızca yukarıda verilen veriyi kullan.\n"
        "- METİN DOĞRUDAN TTS (KONUŞMA SENTEZİ) TARAFINDAN OKUNACAK:\n"
        "  · '...', '(açıklama)', '[not]' gibi konuşulamaz ifadeler YASAK.\n"
        "  · Parantez, köşeli parantez, tire notları, semboller (#, %, &) kullanma.\n"
        "  · Her kelime yüksek sesle okunabilir doğal Türkçe konuşma dili olmalı.\n"
        "- Başka açıklama ekleme, sadece JSON döndür."
    )

    _intro = (
        "Sen profesyonel bir Türkçe canlı futbol yorumcususun — tribünleri coşturan, "
        "sahadan heyecan taşıyan, ritimli ve özgün yorumlar yapan bir spiker.\n"
        "Maç durum verisini (top konumu, baskı seviyesi, kanat/merkez geçişleri, ceza sahası yakınlığı) "
        "yorumlayarak sahada gerçekten ne olduğunu anlamlandır.\n\n"
    )

    # Eğer sahadaki oyuncu isimleri biliniyorsa → prompt'un başına koy
    if _known_player_names:
        _preamble = (
            "SAHADA KİM VAR (Bu isimleri yorumunda kullan):\n"
            + "\n".join(f"- {n}" for n in _known_player_names)
            + "\n\n"
        )
    elif _known_scoring_team:
        # İsim bilinmiyor ama takım biliniyor → en azından takım adını vurgula
        _preamble = f"SAHADA KİM VAR: {_known_scoring_team} oyuncusu (isim tespit edilemedi)\n\n"
    else:
        _preamble = ""

    return _intro + _preamble + "\n\n".join(sections)

def _extract_commentary_text_best_effort(raw: str) -> Optional[str]:
    s = str(raw or "").strip()
    if not s:
        return None
    # Tamamlanmış <think>...</think> bloklarını sil
    s = re.sub(r"<think>[\s\S]*?</think>", "", s, flags=re.IGNORECASE).strip()
    # Kapanmamış <think> bloğunu sil (model max_tokens'a takıldıysa JSON üretememiş olabilir)
    s = re.sub(r"<think>[\s\S]*$", "", s, flags=re.IGNORECASE).strip()
    if not s:
        return None
    s = re.sub(r"^```(?:json)?\s*", "", s.strip(), flags=re.IGNORECASE)
    s = re.sub(r"\s*```$", "", s.strip())

    try:
        obj = json.loads(s)
        if isinstance(obj, dict):
            txt = str(obj.get("text") or "").strip()
            return txt or None
        if isinstance(obj, list) and obj:
            first = obj[0]
            if isinstance(first, dict):
                txt = str(first.get("text") or "").strip()
                return txt or None
            if isinstance(first, str):
                return str(first).strip() or None
    except Exception:
        pass

    m = re.search(r'"text"\s*:\s*"([^"]+)"', s)
    if m:
        return str(m.group(1)).strip() or None

    return s.strip() or None

def _match_state_fallback_sentence(match_state: Dict[str, Any]) -> str:
    state = match_state.get("state_summary") if isinstance(match_state, dict) else None
    state = state if isinstance(state, dict) else {}
    progression = str(state.get("ball_progression") or "").strip()
    tags = [str(x).strip() for x in (state.get("state_tags") or []) if str(x).strip()]
    pressure = str(state.get("pressure_level") or "").strip().lower()
    regions = [str(x).strip() for x in (state.get("ball_regions") or []) if str(x).strip()]

    parts: List[str] = []
    if progression and "net izlenemiyor" not in progression.lower():
        parts.append(progression[:1].upper() + progression[1:])
    elif "kanat kullanımı" in tags:
        parts.append("Oyun kenarlara açılıyor")
    elif regions:
        parts.append(f"Oyun {regions[0]} üzerinden şekilleniyor")
    else:
        parts.append("Takımlar oyunu yeniden kuruyor")

    if pressure == "yüksek":
        parts.append("Orta sahada baskı çok sert")
    elif pressure == "orta":
        parts.append("Merkezde temas ve baskı artıyor")
    elif "ceza sahası çevresi" in tags:
        parts.append("Ceza sahası çevresinde hareketlilik var")

    text = ". ".join(p.rstrip(".!") for p in parts if p).strip()
    return (text + ".") if text else "Oyun akışı tempolu biçimde sürüyor."

def _normalize_commentary_compare(text: str) -> str:
    s = str(text or "").lower()
    s = re.sub(r"[^a-z0-9çğıöşü\s]", " ", s, flags=re.IGNORECASE)
    s = re.sub(r"\s+", " ", s).strip()
    return s

def _is_repetitive_commentary(text: str, recent_texts: List[str]) -> bool:
    cur = _normalize_commentary_compare(text)
    if not cur:
        return True
    cur_head = " ".join(cur.split()[:7])
    for prev in recent_texts[-5:]:
        pp = _normalize_commentary_compare(prev)
        if not pp:
            continue
        if cur == pp:
            return True
        if cur_head and cur_head == " ".join(pp.split()[:7]):
            return True
    return False

def _fallback_commentary_text(item: Dict[str, Any], recent_texts: Optional[List[str]] = None) -> str:
    recent_texts = recent_texts or []
    label = str(item.get("event_label") or "Oyun akışı").strip()
    desc = str(item.get("description_tr") or "").strip()
    state_line = _match_state_fallback_sentence(item.get("match_state") or {})
    style = _commentary_style_for_item(item)

    event_sentence = ""
    if label == "Throw-in":
        variants = [
            "Taç çizgisinde oyun yeniden başlıyor.",
            "Kenarda bekleyiş bitti, taçla oyun tekrar akıyor.",
            "Top çizgiden oyuna dönüyor, yerleşim hızla kuruluyor.",
        ]
        event_sentence = variants[sum(ord(ch) for ch in style) % len(variants)]
    elif label == "Ball out of play":
        variants = [
            "Top oyun alanının dışına çıktı, tempo kısa süreliğine durdu.",
            "Akış bir anlığına kesildi, top çizginin dışına taşmış durumda.",
            "Ritim kısa bir an duruyor, top dışarıda kaldı.",
        ]
        event_sentence = variants[sum(ord(ch) for ch in style) % len(variants)]
    elif label == "Match State":
        variants = [
            "Sahada oyun akışı yeniden şekilleniyor.",
            "Takımlar yerleşimi güncelliyor, oyunun yönü yeniden kuruluyor.",
            "Maçın ritmi yeni bir faza giriyor.",
        ]
        event_sentence = variants[sum(ord(ch) for ch in style) % len(variants)]
    elif desc:
        event_sentence = desc if desc.endswith((".", "!")) else desc + "."
    else:
        event_sentence = f"{label} anı yaşanıyor."

    merged = (event_sentence + " " + state_line).strip()
    sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+", merged) if s.strip()]
    out = " ".join(sentences[:2]) if sentences else "Oyun akışı tempolu biçimde sürüyor."
    if _is_repetitive_commentary(out, recent_texts):
        out = out.replace("Sahada", "Bu bölümde").replace("Top", "Oyun")
    return out

def _sanitize_commentary_text(text: str, item: Dict[str, Any], recent_texts: Optional[List[str]] = None) -> str:
    recent_texts = recent_texts or []
    raw = str(text or "").strip()
    if not raw:
        # LLM hiçbir şey üretemedi — match_state'e dayalı fallback kullan
        match_state = item.get("match_state") if isinstance(item, dict) else None
        if match_state:
            return _match_state_fallback_sentence(match_state)
        return ""

    # ── TTS güvenli temizlik ─────────────────────────────────────────────
    # Parantez içi açıklamaları kaldır
    raw = re.sub(r"\([^)]{0,60}\)", "", raw)
    raw = re.sub(r"\[[^\]]{0,60}\]", "", raw)
    # Üç nokta ve söylünemez sembolleri temizle
    raw = re.sub(r"\.{2,}", ".", raw)          # ... → .
    raw = re.sub(r"\u2026", ".", raw)           # … → .
    raw = re.sub(r"[&+#@\\|<>{}~^`]", " ", raw)
    # Sembol/kısaltma yazımı
    raw = re.sub(r"%", " yüzde ", raw)
    # Tekli tire öncesi/sonrası bofluğu olan ifade nota şablonunu kaldır (  - not )
    raw = re.sub(r"\s+-\s+[a-zçğıöşü]{1,30}(?=\s|$)", "", raw, flags=re.IGNORECASE)
    # Çoklu boşluk temizle
    raw = re.sub(r"\s{2,}", " ", raw).strip()
    if not raw:
        return ""

    period_sec = _commentary_item_period_sec(item)
    event_label_for_budget = str((item or {}).get("event_label") or "")
    max_sentences = _commentary_sentence_budget(period_sec, event_label_for_budget)
    max_words = _commentary_word_budget(period_sec, event_label_for_budget)

    out = _trim_commentary_text(raw, max_sentences=max_sentences, max_words=max_words)
    if not out:
        return ""
    if not out.endswith((".", "!")):
        out += "."
    return out

def _commentary_item_period_sec(item: Dict[str, Any]) -> float:
    window = item.get("window") if isinstance(item, dict) else None
    window = window if isinstance(window, dict) else {}
    duration = window.get("duration_sec", item.get("segment_duration_sec", 10.0))
    try:
        return max(4.0, float(duration or 10.0))
    except Exception:
        return 10.0

def _commentary_sentence_budget(period_sec: float, event_label: str = "") -> int:
    label_lower = str(event_label or "").lower()
    # Gol, şut, kırmızı kart gibi yüksek öncelikli olaylar her zaman min 2 cümle
    if label_lower in ("goal", "own goal", "penalty - goal", "red card", "penalty"):
        return 2
    return 1 if float(period_sec) <= 15.0 else 2

def _commentary_word_budget(period_sec: float, event_label: str = "") -> int:
    label_lower = str(event_label or "").lower()
    # Gol olayları için daha geniş kelime bütçesi
    if label_lower in ("goal", "own goal", "penalty - goal"):
        return 40
    # 10s segment → ~20 words; 30s → 35 words
    return max(18, min(35, int(round(float(period_sec) * 1.2))))

def _trim_commentary_text(text: str, *, max_sentences: int, max_words: int) -> str:
    sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+", str(text or "").strip()) if s.strip()]
    if not sentences:
        return ""
    clipped = " ".join(sentences[: max(1, int(max_sentences))]).strip()
    words = clipped.split()
    if len(words) > int(max_words):
        clipped = " ".join(words[: int(max_words)]).strip(" ,")
    clipped = clipped.strip()
    clipped = re.sub(r"\s+", " ", clipped)
    clipped = clipped.strip(" ,")
    clipped = re.sub(r"[,:;]+$", "", clipped)
    return clipped


def _extract_actor_for_commentary(
    *,
    action_t: float,
    events: List[Dict[str, Any]],
    jersey_by_track: Dict[int, Dict[str, Any]],
    track_id_remap: Dict[int, int],
    max_age_sec: float = 8.0,
) -> Dict[str, Any]:
    """Find ball-possessing actor closest in time to action_t via possession events."""
    best_e: Optional[Dict[str, Any]] = None
    best_dt = float("inf")
    for e in events:
        etype = str(e.get("type") or "")
        if etype not in ("possession_start", "possession_change"):
            continue
        try:
            et = float(e.get("t", 0.0))
        except Exception:
            continue
        dt = float(action_t) - et
        if dt < -2.0 or dt > float(max_age_sec):
            continue
        if abs(dt) < abs(best_dt):
            best_dt = dt
            best_e = e

    result: Dict[str, Any] = {}
    if best_e is None:
        return result

    def _lookup_player(pid: Any) -> Optional[Dict[str, Any]]:
        if pid is None:
            return None
        try:
            tid = int(pid)
        except Exception:
            return None
        if _is_special_track_id(tid):
            return None
        canon = int((track_id_remap or {}).get(tid, tid))
        info = (jersey_by_track or {}).get(canon) or (jersey_by_track or {}).get(tid)
        out: Dict[str, Any] = {"track_id": canon}
        if info:
            j = info.get("jersey_number")
            if j is not None and str(j).strip() not in ("", "-1"):
                out["jersey_number"] = str(j).strip()
            pname = info.get("player_name")
            if pname and str(pname).strip():
                out["player_name"] = str(pname).strip()
            tid_val = info.get("team_id")
            if tid_val is not None:
                try:
                    out["team_id"] = int(tid_val)
                except Exception:
                    pass
        return out

    actor = _lookup_player(best_e.get("player_track_id"))
    if actor:
        ev_jersey = best_e.get("jersey_number")
        if ev_jersey is not None and str(ev_jersey).strip() not in ("", "-1"):
            actor.setdefault("jersey_number", str(ev_jersey).strip())
        result["actor"] = actor

    from_actor = _lookup_player(best_e.get("from_player_track_id"))
    if from_actor:
        ev_from_jersey = best_e.get("from_jersey_number")
        if ev_from_jersey is not None and str(ev_from_jersey).strip() not in ("", "-1"):
            from_actor.setdefault("jersey_number", str(ev_from_jersey).strip())
        result["from_actor"] = from_actor

    return result


def _extract_nearby_events_for_commentary(
    *,
    action_t: float,
    events: List[Dict[str, Any]],
    before_sec: float = 15.0,
    max_count: int = 4,
) -> List[Dict[str, Any]]:
    """Return last N significant events before action_t for narrative context."""
    lo = float(action_t) - float(before_sec)
    hi = float(action_t) - 0.5  # exclude the event itself

    _SKIP_TYPES = {"ball_uncontrolled", "possession_start", "possession_change"}
    nearby: List[Dict[str, Any]] = []
    for e in events:
        try:
            et = float(e.get("t", 0.0))
        except Exception:
            continue
        if et < lo or et > hi:
            continue
        label = str(e.get("label") or e.get("type") or "").strip()
        if not label or label in _SKIP_TYPES:
            continue
        nearby.append({
            "t": round(et, 1),
            "timecode": _timecode_mmss(et),
            "label": label,
        })

    nearby.sort(key=lambda x: x["t"])
    return nearby[-max_count:]


def _build_commentary_items(
    *,
    events: List[Dict[str, Any]],
    action_events: List[Dict[str, Any]],
    calibration_frames: List[Dict[str, Any]],
    calibration_frame_times: List[float],
    jersey_by_track: Dict[int, Dict[str, Any]],
    track_id_remap: Dict[int, int],
    cfg: FullPipelineConfig,
    engine_meta_events: Optional[List[Any]] = None,
    video_fps: float = 25.0,
) -> List[Dict[str, Any]]:
    segment_sec = max(10.0, float(getattr(cfg, "commentary_segment_sec", 30.0) or 30.0))
    # Minimum gap between two consecutive commentary items to avoid rapid-fire speech
    min_item_gap_sec = max(5.0, float(getattr(cfg, "commentary_min_item_gap_sec", 8.0) or 8.0))

    # Priority tiers for T-DEED events (lower number = higher priority)
    _TIER1_LABELS: set = {"goal", "red card", "penalty", "own goal", "goal (handball)", "penalty - goal"}
    _TIER2_LABELS: set = {"yellow card", "corner", "shot on target", "foul", "var"}
    _TIER3_LABELS: set = {"throw-in", "ball out of play"}
    # T-DEED events not in any tier → tier 4 (lowest T-DEED priority, still above engine events)

    def _tdeed_prio(label: str) -> int:
        l = label.strip().lower()
        if l in _TIER1_LABELS:
            return 0
        if l in _TIER2_LABELS:
            return 1
        if l in _TIER3_LABELS:
            return 2
        return 3

    # ── Step 1: ALL T-DEED events always get a slot ─────────────────────────
    # Sort by time; deduplicate near-identical detections (same event, < min_item_gap_sec apart).
    _fps = max(1.0, float(video_fps or 25.0))

    tdeed_raw: List[Tuple[float, int, Dict[str, Any]]] = []
    for e in action_events:
        try:
            label = str(e.get("label") or e.get("type") or "").strip().lower()
            tdeed_raw.append((float(e.get("t", 0.0)), _tdeed_prio(label), e))
        except Exception:
            continue
    tdeed_raw.sort(key=lambda x: x[0])

    # Deduplicate: if two T-DEED events are within min_item_gap_sec, keep higher-priority one
    tdeed_chosen: List[Tuple[float, int, Dict[str, Any]]] = []
    for t, prio, e in tdeed_raw:
        if tdeed_chosen and (t - tdeed_chosen[-1][0]) < min_item_gap_sec:
            if prio < tdeed_chosen[-1][1]:  # current is higher priority → replace
                tdeed_chosen[-1] = (t, prio, e)
        else:
            tdeed_chosen.append((t, prio, e))

    # ── Step 2: EventEngine fills gaps between T-DEED events ─────────────────
    # Only include PRIORITY_MID (2) and PRIORITY_HIGH (3) engine events, not spammy low-priority ones.
    _ENGINE_GAP_EXCLUDE: frozenset = frozenset({
        "possession_change", "zone_change", "ball_carry", "sprint", "loose_ball", "pass",
    })

    engine_gap_anchors: List[Tuple[float, Dict[str, Any]]] = []
    for em in (engine_meta_events or []):
        try:
            if int(em.priority) < 2:
                continue  # skip low-priority engine events
            # Also skip events whose text matches excluded types
            txt_lower = str(em.event_text or "").lower()
            if any(exc in txt_lower for exc in _ENGINE_GAP_EXCLUDE):
                continue
            t = float(em.frame) / _fps
            engine_gap_anchors.append((t, {
                "t": t,
                "label": str(em.event_text or ""),
                "type": "engine",
                "source": "event_engine",
                "confidence": float(em.priority) / 3.0,
                "event_text": str(em.event_text or ""),
                "track_id": em.track_id,
            }))
        except Exception:
            continue
    engine_gap_anchors.sort(key=lambda x: x[0])

    # ── Step 3: Determine timeline and fill gaps ──────────────────────────────
    timeline_end = 0.0
    if calibration_frame_times:
        timeline_end = float(calibration_frame_times[-1])
    elif events:
        try:
            timeline_end = max(float(e.get("t", 0.0) or 0.0) for e in events)
        except Exception:
            timeline_end = 0.0
    if timeline_end <= 0.0:
        all_ts = [x[0] for x in tdeed_chosen] + [x[0] for x in engine_gap_anchors]
        if all_ts:
            timeline_end = max(all_ts)
    if timeline_end <= 0.0:
        timeline_end = float(segment_sec)

    # Build gap list: [start, end) intervals with no T-DEED event
    # Gap threshold: fill gap only if > segment_sec
    tdeed_times = sorted(t for t, _, _ in tdeed_chosen)
    gap_intervals: List[Tuple[float, float]] = []
    boundaries = [0.0] + tdeed_times + [timeline_end]
    for i in range(len(boundaries) - 1):
        g_start = boundaries[i]
        g_end = boundaries[i + 1]
        if (g_end - g_start) > segment_sec:
            # Split long gap into segment_sec chunks
            cur = g_start
            while cur < g_end - 1.0:
                gap_intervals.append((cur, min(g_end, cur + segment_sec)))
                cur += segment_sec

    # For each gap interval, pick the best engine event (highest priority, then closest to center)
    engine_chosen: List[Tuple[float, int, Dict[str, Any]]] = []
    for g_start, g_end in gap_intervals:
        cands = [(t, e) for t, e in engine_gap_anchors if g_start <= t < g_end]
        if not cands:
            continue
        best_t, best_e = sorted(cands, key=lambda x: (
            -float(x[1].get("confidence", 0.0)),
            abs(x[0] - (g_start + g_end) * 0.5),
        ))[0]
        # Avoid items too close to adjacent T-DEED items
        too_close = any(abs(best_t - tt) < min_item_gap_sec for tt in tdeed_times)
        if not too_close:
            engine_chosen.append((best_t, 10, best_e))  # prio=10 → always below T-DEED

    # ── Step 4: Merge and sort all chosen items ───────────────────────────────
    all_chosen_raw = [(t, p, e) for t, p, e in tdeed_chosen] + engine_chosen
    all_chosen_raw.sort(key=lambda x: x[0])

    # Build (seg_start, seg_end, event_t, prio, event) tuples for downstream use
    chosen: List[Tuple[float, float, float, int, Dict[str, Any]]] = []
    for t, prio, e in all_chosen_raw:
        seg_s = max(0.0, t - segment_sec / 2)
        seg_e = min(timeline_end, t + segment_sec / 2)
        chosen.append((seg_s, seg_e, t, prio, e))

    _cfg_max = int(getattr(cfg, "commentary_max_events", 0) or 0)
    if _cfg_max > 0:
        chosen = chosen[:_cfg_max]

    calib_window_sec = max(4.0, float(getattr(cfg, "commentary_context_window_sec", 12.0) or 12.0))
    calib_stride_sec = max(0.25, float(getattr(cfg, "commentary_context_stride_sec", 1.0) or 1.0))
    calib_max_samples = max(3, int(getattr(cfg, "commentary_context_max_samples", 9) or 9))
    actor_max_age_sec = max(2.0, float(getattr(cfg, "commentary_possession_max_age_sec", 8.0) or 8.0))

    items_in: List[Dict[str, Any]] = []
    for seg_start, seg_end, event_t, _prio, ae in chosen:
        event_label = str(ae.get("label") or ae.get("type") or "")
        label_lower = event_label.strip().lower()

        # speech_t: offset from event_t.
        # Positive offset → speak after the event (e.g. card, VAR reveal).
        # Negative offset → speak before the event (e.g. goal anticipation).
        # Zero (default) → use pre_delay_sec before the event.
        speech_offset = _EVENT_SPEECH_OFFSET.get(label_lower, 0.0)
        pre_delay_sec = max(0.0, float(getattr(cfg, "commentary_event_pre_delay_sec", 1.0) or 1.0))
        if speech_offset > 0.0:
            speech_t = max(float(seg_start), float(event_t) + float(speech_offset))
        elif speech_offset < 0.0:
            speech_t = max(float(seg_start), float(event_t) + float(speech_offset))  # offset negatif → erken başla
        else:
            speech_t = max(float(seg_start), float(event_t) - float(pre_delay_sec))

        # Calibration summary: ball movement, pressure, zones around event_t
        match_state: Dict[str, Any] = {}
        try:
            match_state = _summarize_calibration_window(
                action_t=float(event_t),
                calibration_frames=calibration_frames,
                calibration_times=calibration_frame_times,
                jersey_by_track=jersey_by_track,
                track_id_remap=track_id_remap,
                window_sec=calib_window_sec,
                stride_sec=calib_stride_sec,
                max_samples=calib_max_samples,
            )
        except Exception:
            pass

        # Actor: ball possessor nearest to event_t from possession tracking events
        actor_info: Dict[str, Any] = {}
        try:
            actor_info = _extract_actor_for_commentary(
                action_t=float(event_t),
                events=events,
                jersey_by_track=jersey_by_track,
                track_id_remap=track_id_remap,
                max_age_sec=actor_max_age_sec,
            )
        except Exception:
            pass

        # Nearby preceding events for narrative continuity in prompt
        nearby_events: List[Dict[str, Any]] = []
        try:
            nearby_events = _extract_nearby_events_for_commentary(
                action_t=float(event_t),
                events=events,
                before_sec=15.0,
                max_count=4,
            )
        except Exception:
            pass

        item: Dict[str, Any] = {
            "t": float(seg_start),
            "speech_t": float(speech_t),
            "timecode": _timecode_mmss(float(speech_t)),
            "event_t": float(event_t),
            "event_timecode": _timecode_mmss(float(event_t)),
            "segment_duration_sec": round(float(seg_end - seg_start), 3),
            "window": {
                "start_t": round(float(seg_start), 3),
                "end_t": round(float(seg_end), 3),
                "start_timecode": _timecode_mmss(float(seg_start)),
                "end_timecode": _timecode_mmss(float(seg_end)),
                "duration_sec": round(float(seg_end - seg_start), 3),
            },
            "event_label": event_label,
            "event_confidence": float(ae.get("confidence", 0.0) or 0.0),
            "event_source": str(ae.get("source") or "unknown"),
            "description_tr": str(ae.get("description_tr") or ""),
            "spotting_model": str(ae.get("spotting_model") or ""),
            "spotting_variant": str(ae.get("spotting_variant") or ""),
        }
        if match_state:
            item["match_state"] = match_state
        if actor_info:
            item["actor_info"] = actor_info
        if nearby_events:
            item["nearby_events"] = nearby_events

        # ── Gol öncesi şut linki ─────────────────────────────────────────
        # Gol olayı tespit edildiğinde, ±10s içinde shot/shot-on-target varsa
        # LLM'e bildir ki "şutu attı ve golle sonuçlandı" şeklinde anlat.
        if label_lower in ("goal", "own goal", "penalty - goal"):
            _SHOT_LABELS = {"shot", "shot on target"}
            _pre_shot: Optional[Dict[str, Any]] = None
            _best_dist = 999.0
            for _ae_s in action_events:
                _lbl_s = str(_ae_s.get("label") or _ae_s.get("type") or "").strip().lower()
                if _lbl_s not in _SHOT_LABELS:
                    continue
                _ts = float(_ae_s.get("t", 0.0))
                _dist = abs(_ts - float(event_t))
                if _dist <= 10.0 and _dist < _best_dist:
                    _best_dist = _dist
                    _pre_shot = _ae_s
            if _pre_shot is not None:
                item["pre_shot_event"] = {
                    "label": str(_pre_shot.get("label") or _pre_shot.get("type") or ""),
                    "t": float(_pre_shot.get("t", 0.0)),
                    "timecode": _timecode_mmss(float(_pre_shot.get("t", 0.0))),
                    "confidence": float(_pre_shot.get("confidence", 0.0) or 0.0),
                    "distance_to_goal_sec": round(_best_dist, 2),
                }

        items_in.append(item)

    # ── Top Durduğu Anlar (pause commentary) ────────────────────────────
    # Kalibrasyon frame'lerinde topun görülmediği, uzun sessiz dönemleri
    # tespit edip ayrı bir commentary item olarak ekle.
    if calibration_frames and calibration_frame_times:
        _PAUSE_MIN_SEC = 4.0      # en az 4s top yok → "duraklama" say
        _PAUSE_MAX_GAP = 60.0     # 60s'den uzun boşlukları atla (kesik video)
        _pause_start: Optional[float] = None
        _prev_ft = 0.0

        def _frame_has_ball(f: Dict[str, Any]) -> bool:
            return f.get("ball") is not None

        _existing_t_set = {round(it["event_t"], 1) for it in items_in}
        pause_items: List[Dict[str, Any]] = []

        for _fi, (_cf, _ft) in enumerate(zip(calibration_frames, calibration_frame_times)):
            _ft = float(_ft)
            if _fi > 0 and (_ft - _prev_ft) > _PAUSE_MAX_GAP:
                _pause_start = None
            if not _frame_has_ball(_cf):
                if _pause_start is None:
                    _pause_start = _ft
            else:
                if _pause_start is not None:
                    _dur = _ft - _pause_start
                    if _dur >= _PAUSE_MIN_SEC:
                        _mid = (_pause_start + _ft) * 0.5
                        # Var olan event'le çakışıyorsa ekleme
                        _too_close = any(
                            abs(_mid - float(it["event_t"])) < 8.0
                            for it in items_in
                        )
                        if not _too_close:
                            _p_speech_t = _pause_start + 1.0
                            _p_item: Dict[str, Any] = {
                                "t": round(_pause_start, 3),
                                "speech_t": round(_p_speech_t, 3),
                                "timecode": _timecode_mmss(_p_speech_t),
                                "event_t": round(_mid, 3),
                                "event_timecode": _timecode_mmss(_mid),
                                "segment_duration_sec": round(_dur, 3),
                                "window": {
                                    "start_t": round(_pause_start, 3),
                                    "end_t": round(_ft, 3),
                                    "start_timecode": _timecode_mmss(_pause_start),
                                    "end_timecode": _timecode_mmss(_ft),
                                    "duration_sec": round(_dur, 3),
                                },
                                "event_label": "ball out of play",
                                "event_confidence": 0.9,
                                "event_source": "pause_detector",
                                "description_tr": f"Top {round(_dur, 1)}s boyunca sahada görülmedi",
                                "pause_duration_sec": round(_dur, 3),
                                "spotting_model": "",
                                "spotting_variant": "",
                            }
                            # Bu anın match_state ve nearby'sini de ekle
                            try:
                                _p_ms = _summarize_calibration_window(
                                    action_t=_mid,
                                    calibration_frames=calibration_frames,
                                    calibration_times=calibration_frame_times,
                                    jersey_by_track=jersey_by_track,
                                    track_id_remap=track_id_remap,
                                    window_sec=calib_window_sec,
                                    stride_sec=calib_stride_sec,
                                    max_samples=calib_max_samples,
                                )
                                if _p_ms:
                                    _p_item["match_state"] = _p_ms
                            except Exception:
                                pass
                            try:
                                _p_nearby = _extract_nearby_events_for_commentary(
                                    action_t=_mid,
                                    events=events,
                                    before_sec=15.0,
                                    max_count=4,
                                )
                                if _p_nearby:
                                    _p_item["nearby_events"] = _p_nearby
                            except Exception:
                                pass
                            pause_items.append(_p_item)
                    _pause_start = None
            _prev_ft = _ft

        # Pause item'larını ana listeye ekle ve zamana göre sırala
        items_in.extend(pause_items)
        items_in.sort(key=lambda x: float(x.get("speech_t", x.get("t", 0.0))))

    return items_in

def _extract_json_array_best_effort(raw: str) -> Optional[str]:
    s = str(raw or "").strip()
    if not s:
        return None
    s = re.sub(r"^```(?:json)?\s*", "", s.strip(), flags=re.IGNORECASE)
    s = re.sub(r"\s*```$", "", s.strip())
    start = s.find("[")
    end = s.rfind("]")
    if start == -1 or end == -1 or end <= start:
        return None
    return s[start : end + 1]

def _timecode_mmss(t: float) -> str:
    return _timecode(float(t))

def _assign_actor_track_id_to_action(
    *,
    action_t: float,
    possession_events: List[Dict[str, Any]],
    max_age_sec: float,
) -> Optional[int]:
    best_t = -1e9
    best_id: Optional[int] = None
    for e in possession_events:
        try:
            et = float(e.get("t", 0.0))
        except Exception:
            continue
        if et > float(action_t):
            break
        pid = e.get("player_track_id")
        if pid is None:
            continue
        try:
            pid_i = int(pid)
        except Exception:
            continue
        if _is_special_track_id(pid_i):
            continue
        if et >= best_t:
            best_t = et
            best_id = pid_i

    if best_id is None:
        return None
    if float(action_t) - float(best_t) > float(max_age_sec):
        return None
    return int(best_id)

def _load_calibration_frames_jsonl(path: Optional[str]) -> Tuple[List[Dict[str, Any]], List[float]]:
    frames: List[Dict[str, Any]] = []
    times: List[float] = []
    p = str(path or "").strip()
    if not p or not os.path.isfile(p):
        return frames, times

    with open(p, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            s = str(line or "").strip()
            if not s:
                continue
            try:
                obj = json.loads(s)
                tt = float(obj.get("t", 0.0) or 0.0)
            except Exception:
                continue
            frames.append(obj)
            times.append(tt)

    return frames, times

def _lane_label(x: float) -> str:
    if x <= -14.0:
        return "sol kanat"
    if x >= 14.0:
        return "sağ kanat"
    return "merkez"

def _band_label(y: float) -> str:
    if y <= -18.0:
        return "savunmadan çıkış bölgesi"
    if y >= 18.0:
        return "hücum bölgesi"
    return "orta saha"

def _is_penalty_area_zone(x: float, y: float) -> bool:
    return abs(float(x)) <= 20.0 and abs(float(y)) >= 24.0

def _compact_event_for_commentary(e: Dict[str, Any]) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for key in (
        "t",
        "timecode",
        "type",
        "label",
        "source",
        "description_tr",
        "player_track_id",
        "from_player_track_id",
        "jersey_number",
        "from_jersey_number",
        "confidence",
    ):
        if key in e:
            out[key] = e.get(key)
    return out

def _summarize_calibration_window(
    *,
    action_t: float,
    calibration_frames: List[Dict[str, Any]],
    calibration_times: List[float],
    jersey_by_track: Dict[int, Dict[str, Any]],
    track_id_remap: Dict[int, int],
    window_sec: float,
    stride_sec: float,
    max_samples: int,
) -> Dict[str, Any]:
    if not calibration_frames or not calibration_times:
        start_t = max(0.0, float(action_t) - float(window_sec) / 2.0)
        end_t = max(start_t, float(action_t) + float(window_sec) / 2.0)
        return {
            "window": {
                "start_t": round(start_t, 3),
                "end_t": round(end_t, 3),
            },
            "frame_samples": [],
            "state_summary": {
                "ball_progression": "kalibrasyon kare verisi yok",
                "pressure_level": "bilinmiyor",
                "state_tags": [],
            },
        }

    half_window = max(2.0, float(window_sec) / 2.0)
    start_t = max(0.0, float(action_t) - half_window)
    end_t = max(start_t, float(action_t) + half_window)

    left = bisect_left(calibration_times, start_t)
    right = bisect_right(calibration_times, end_t)
    indices = list(range(left, right))

    if not indices:
        nearest = min(max(0, bisect_left(calibration_times, float(action_t))), len(calibration_times) - 1)
        indices = [nearest]

    stride = max(0.25, float(stride_sec))
    desired_ts: List[float] = []
    cur = start_t
    while cur <= end_t + 1e-6:
        desired_ts.append(round(cur, 3))
        cur += stride
    desired_ts.append(round(float(action_t), 3))

    selected: List[int] = []
    for tt in desired_ts:
        idx = bisect_left(calibration_times, tt)
        cand: List[int] = []
        if idx < len(calibration_times):
            cand.append(idx)
        if idx > 0:
            cand.append(idx - 1)
        if not cand:
            continue
        best = min(cand, key=lambda ii: abs(float(calibration_times[ii]) - float(tt)))
        if best < left or best >= right:
            continue
        if best not in selected:
            selected.append(best)

    if not selected:
        selected = indices[:]

    selected.sort()
    max_keep = max(3, int(max_samples))
    if len(selected) > max_keep:
        pos = np.linspace(0, len(selected) - 1, num=max_keep)
        selected = sorted({selected[min(len(selected) - 1, max(0, int(round(x))))] for x in pos})

    frame_samples: List[Dict[str, Any]] = []
    ball_points: List[Tuple[float, float]] = []
    lane_history: List[str] = []
    band_history: List[str] = []
    near_ball_counts: List[int] = []
    penalty_hits = 0
    focus_players: Dict[int, Dict[str, Any]] = {}

    for idx in selected:
        frame = calibration_frames[idx]
        try:
            frame_idx = int(frame.get("frame_idx", 0) or 0)
            tt = float(frame.get("t", 0.0) or 0.0)
        except Exception:
            continue

        data = frame.get("data") if isinstance(frame, dict) else None
        data = data if isinstance(data, dict) else {}
        ball = data.get("ball") if isinstance(data, dict) else None
        players = data.get("players") if isinstance(data, dict) else None
        players = players if isinstance(players, list) else []

        sample: Dict[str, Any] = {
            "t": round(tt, 3),
            "timecode": _timecode_mmss(tt),
            "frame_idx": frame_idx,
            "player_count": int(len(players)),
        }

        if isinstance(ball, dict) and isinstance(ball.get("world_xy"), (list, tuple)) and len(ball.get("world_xy")) >= 2:
            try:
                bx = float(ball["world_xy"][0])
                by = float(ball["world_xy"][1])
                lane = _lane_label(bx)
                band = _band_label(by)
                in_box = _is_penalty_area_zone(bx, by)
                ball_points.append((bx, by))
                lane_history.append(lane)
                band_history.append(band)
                if in_box:
                    penalty_hits += 1

                nearby: List[Tuple[float, Dict[str, Any]]] = []
                for p in players:
                    try:
                        pxy = p.get("world_xy")
                        if not isinstance(pxy, (list, tuple)) or len(pxy) < 2:
                            continue
                        px = float(pxy[0])
                        py = float(pxy[1])
                        dist = float(np.sqrt((bx - px) ** 2 + (by - py) ** 2))
                        track_id_raw = int(p.get("track_id", 0) or 0)
                        track_id = _resolve_track_id_remap(track_id_raw, track_id_remap) if track_id_raw > 0 else track_id_raw
                        info = jersey_by_track.get(int(track_id), {}) if track_id > 0 else {}
                        team_id = p.get("team_id")
                        # -1 ve None durumlarında jersey_by_track'ten override et
                        if (team_id is None or int(team_id) == -1) and info:
                            team_id = info.get("team_id")
                        _entry: Dict[str, Any] = {
                            "track_id": int(track_id) if track_id > 0 else track_id_raw,
                            "jersey_number": info.get("jersey_number") if info else None,
                            "team_id": team_id,
                            "distance_to_ball_m": round(dist, 2),
                        }
                        _pname = info.get("player_name") if info else None
                        if _pname:
                            _entry["player_name"] = str(_pname)
                        nearby.append((dist, _entry))
                    except Exception:
                        continue

                nearby.sort(key=lambda item: item[0])
                near_cnt = sum(1 for dist, _ in nearby if dist <= 8.0)
                near_ball_counts.append(int(near_cnt))
                nearest_players = [entry for _dist, entry in nearby[:4]]
                for entry in nearest_players:
                    tid = int(entry.get("track_id") or 0)
                    if tid > 0 and tid not in focus_players:
                        focus_players[tid] = entry

                sample["ball"] = {
                    "world_xy": [round(bx, 2), round(by, 2)],
                    "lane": lane,
                    "band": band,
                    "penalty_area_proximity": bool(in_box),
                    "nearby_pressure_count": int(near_cnt),
                    "nearest_players": nearest_players,
                }
            except Exception:
                sample["ball"] = None
        else:
            sample["ball"] = None

        frame_samples.append(sample)

    progression = "topun konumu net izlenemiyor"
    if len(ball_points) >= 2:
        sx, sy = ball_points[0]
        ex, ey = ball_points[-1]
        dx = ex - sx
        dy = ey - sy
        if abs(dx) < 4.0 and abs(dy) < 4.0:
            progression = "top dar bir alanda dolaşıyor"
        else:
            move_parts: List[str] = []
            if _lane_label(sx) != _lane_label(ex):
                move_parts.append(f"{_lane_label(sx)}dan {_lane_label(ex)}e geçiş var")
            elif abs(dx) >= 5.0:
                move_parts.append(f"top {_lane_label(ex)} koridorunda taşınıyor")
            if dy >= 7.0:
                move_parts.append("top ileri doğru taşınıyor")
            elif dy <= -7.0:
                move_parts.append("top geriye veya savunma yönüne dönüyor")
            if not move_parts:
                move_parts.append("topun yönü sık sık değişiyor")
            progression = ", ".join(move_parts)

    avg_pressure = float(sum(near_ball_counts)) / float(len(near_ball_counts)) if near_ball_counts else 0.0
    pressure_level = "düşük"
    if avg_pressure >= 4.0:
        pressure_level = "yüksek"
    elif avg_pressure >= 2.0:
        pressure_level = "orta"

    state_tags: List[str] = []
    if penalty_hits > 0:
        state_tags.append("ceza sahası çevresi")
    if "sol kanat" in lane_history or "sağ kanat" in lane_history:
        state_tags.append("kanat kullanımı")
    if "merkez" in lane_history:
        state_tags.append("merkez bağlantısı")
    if pressure_level == "yüksek":
        state_tags.append("yoğun baskı")

    start_frame = int(frame_samples[0].get("frame_idx", 0)) if frame_samples else None
    end_frame = int(frame_samples[-1].get("frame_idx", 0)) if frame_samples else None
    return {
        "window": {
            "start_t": round(start_t, 3),
            "end_t": round(end_t, 3),
            "start_timecode": _timecode_mmss(start_t),
            "end_timecode": _timecode_mmss(end_t),
            "start_frame": start_frame,
            "end_frame": end_frame,
            "sample_stride_sec": float(stride),
        },
        "state_summary": {
            "ball_progression": progression,
            "ball_regions": list(dict.fromkeys(lane_history + band_history))[:6],
            "pressure_level": pressure_level,
            "avg_nearby_pressure_count": round(avg_pressure, 2),
            "state_tags": state_tags,
            "focus_players": list(focus_players.values())[:6],
        },
        "frame_samples": frame_samples,
    }

def _mix_commentary_audio_into_video(
    *,
    base_video_path: str,
    out_path: str,
    clips: List[Tuple[float, str]],
    ambient_path: Optional[str] = None,
    sfx_clips: Optional[List[Tuple]] = None,   # (t, path) or (t, path, volume)
) -> Optional[str]:
    """Mix commentary clips (+ optional ambient loop + goal SFX) into the video.

    Layout:
      - ambient_path (crowd.mp3): looped for full video duration at volume=0.15 (background)
      - sfx_clips: placed at given timestamps; each entry is (t, path) or (t, path, volume).
                   Default volume when not specified = 0.65.
      - clips (TTS commentary): placed at scheduled timestamps at volume=1.0
    All streams are mixed with normalize=0 (additive) so volumes stay independent.
    """
    ffmpeg_bin = _ffmpeg_exe()
    if not ffmpeg_bin:
        return None
    if not clips and not sfx_clips:
        return None

    kept: List[Tuple[float, str]] = []
    for t, p in (clips or []):
        try:
            if os.path.isfile(p) and os.path.getsize(p) > 0:
                kept.append((float(t), str(p)))
        except Exception:
            continue

    kept_sfx: List[Tuple[float, str, float]] = []
    for item_sfx in (sfx_clips or []):
        try:
            t_sfx = float(item_sfx[0])
            p_sfx = str(item_sfx[1])
            vol_sfx = float(item_sfx[2]) if len(item_sfx) > 2 else 0.65
            if os.path.isfile(p_sfx) and os.path.getsize(p_sfx) > 0:
                kept_sfx.append((t_sfx, p_sfx, vol_sfx))
        except Exception:
            continue

    if not kept and not kept_sfx:
        return None

    use_ambient = bool(ambient_path and os.path.isfile(str(ambient_path)))

    duration_sec: Optional[float] = None
    try:
        cap = cv2.VideoCapture(str(base_video_path))
        if cap is not None and cap.isOpened():
            fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
            frames = float(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0.0)
            if fps > 1e-3 and frames > 1:
                duration_sec = float(frames / fps)
        try:
            cap.release()
        except Exception:
            pass
    except Exception:
        duration_sec = None

    # Build input list: [0]=video, then ambient (if any), then kept clips, then sfx clips
    inputs = ["-i", str(Path(base_video_path).resolve())]
    input_idx = 1  # 0 is the video

    ambient_input_idx: Optional[int] = None
    if use_ambient:
        inputs += ["-stream_loop", "-1", "-i", str(Path(str(ambient_path)).resolve())]
        ambient_input_idx = input_idx
        input_idx += 1

    commentary_input_start = input_idx
    for _, ap in kept:
        inputs += ["-i", str(Path(ap).resolve())]
        input_idx += 1

    sfx_input_start = input_idx
    for _, ap, _vol in kept_sfx:
        inputs += ["-i", str(Path(ap).resolve())]
        input_idx += 1

    parts: List[str] = []
    amix_inputs: List[str] = []

    # --- Ambient track: loop + trim to video duration + low volume ---
    if use_ambient and ambient_input_idx is not None and duration_sec and duration_sec > 0.25:
        dur = max(0.25, float(duration_sec))
        parts.append(
            f"[{ambient_input_idx}:a]aformat=sample_fmts=fltp:sample_rates=44100:channel_layouts=stereo,"
            f"atrim=0:{dur:.3f},asetpts=N/SR/TB,volume=0.15[amb]"
        )
        amix_inputs.append("[amb]")
    elif duration_sec is not None and duration_sec > 0.25:
        # Fallback silence base (original behaviour when no ambient)
        dur = max(0.25, float(duration_sec))
        parts.append(
            "anullsrc=channel_layout=stereo:sample_rate=44100,atrim=0:{:.3f},asetpts=N/SR/TB[sil]".format(dur)
        )
        amix_inputs.append("[sil]")

    # --- TTS commentary clips ---
    for i, (t, _ap) in enumerate(kept):
        delay_ms = max(0, int(round(float(t) * 1000.0)))
        cidx = commentary_input_start + i
        tag = f"ac{i}"
        parts.append(
            f"[{cidx}:a]aformat=sample_fmts=fltp:sample_rates=44100:channel_layouts=stereo,"
            f"adelay={delay_ms}|{delay_ms}[{tag}]"
        )
        amix_inputs.append(f"[{tag}]")

    # --- Goal SFX clips — per-clip volume ---
    for j, (t, _ap, vol) in enumerate(kept_sfx):
        delay_ms = max(0, int(round(float(t) * 1000.0)))
        sidx = sfx_input_start + j
        tag = f"as{j}"
        parts.append(
            f"[{sidx}:a]aformat=sample_fmts=fltp:sample_rates=44100:channel_layouts=stereo,"
            f"adelay={delay_ms}|{delay_ms},volume={vol:.2f}[{tag}]"
        )
        amix_inputs.append(f"[{tag}]")

    # normalize=0: additive mix so individual volumes stay independent
    mix_duration = "first" if (use_ambient or duration_sec) else "longest"
    parts.append(
        "".join(amix_inputs)
        + f"amix=inputs={len(amix_inputs)}:duration={mix_duration}:dropout_transition=0:normalize=0[outa]"
    )
    filter_complex = ";".join(parts)

    out_path = str(Path(out_path).resolve())
    os.makedirs(str(Path(out_path).parent), exist_ok=True)

    def _build_cmd(extra_v_flags: List[str]) -> List[str]:
        return [
            ffmpeg_bin,
            "-y",
            *inputs,
            "-filter_complex",
            filter_complex,
            "-map", "0:v:0",
            "-map", "[outa]",
            *extra_v_flags,
            "-c:a", "aac",
            "-b:a", "160k",
            "-movflags", "+faststart",
            out_path,
        ]

    try:
        subprocess.run(_build_cmd(["-c:v", "copy"]), capture_output=True, text=True, check=True)
        return out_path
    except subprocess.CalledProcessError as _e1:
        _stderr1 = (_e1.stderr or "")[-2000:]
        try:
            subprocess.run(
                _build_cmd(["-c:v", "libx264", "-preset", "fast", "-crf", "20", "-pix_fmt", "yuv420p"]),
                capture_output=True, text=True, check=True,
            )
            return out_path
        except subprocess.CalledProcessError as _e2:
            _stderr2 = (_e2.stderr or "")[-2000:]
            raise RuntimeError(f"ffmpeg mix failed.\ncopy stderr: {_stderr1}\nlibx264 stderr: {_stderr2}") from _e2
    except Exception:
        return None

def _audio_duration_sec(audio_path: str) -> float:
    try:
        with wave.open(str(audio_path), "rb") as wf:
            frames = int(wf.getnframes())
            rate = int(wf.getframerate())
            if rate <= 0:
                return 0.0
            return max(0.0, float(frames) / float(rate))
    except Exception:
        return 0.0

def _jersey_crop_from_player_bbox(
    frame_bgr: np.ndarray,
    *,
    x1: int,
    y1: int,
    x2: int,
    y2: int,
) -> Optional[np.ndarray]:
    h, w = frame_bgr.shape[:2]
    x1 = _clamp_int(x1, 0, w - 1)
    x2 = _clamp_int(x2, 0, w - 1)
    y1 = _clamp_int(y1, 0, h - 1)
    y2 = _clamp_int(y2, 0, h - 1)
    if x2 <= x1 or y2 <= y1:
        return None

    bw = x2 - x1
    bh = y2 - y1
    cx1 = x1 + int(0.15 * bw)
    cx2 = x2 - int(0.15 * bw)
    cy1 = y1 + int(0.20 * bh)
    cy2 = y1 + int(0.90 * bh)
    cx1 = _clamp_int(cx1, 0, w - 1)
    cx2 = _clamp_int(cx2, 0, w - 1)
    cy1 = _clamp_int(cy1, 0, h - 1)
    cy2 = _clamp_int(cy2, 0, h - 1)
    if cx2 <= cx1 or cy2 <= cy1:
        crop = frame_bgr[y1:y2, x1:x2]
    else:
        crop = frame_bgr[cy1:cy2, cx1:cx2]
    if crop is None or crop.size == 0:
        return None
    return crop

def _parse_jersey_number_from_text(text: str) -> Optional[str]:
    if not text:
        return None
    t = str(text).strip()

    # Handle JSON output: {"number": "7", "team": "A"}
    try:
        obj = json.loads(t)
        if isinstance(obj, dict):
            t = str(obj.get("number") or "").strip()
    except Exception:
        pass

    if t == "-1":
        return "-1"

    t = t.split()[0] if t.split() else t

    if not re.fullmatch(r"\d{1,2}", t):
        return None
    try:
        n = int(t)
    except Exception:
        return None
    if 0 <= n <= 99:
        return str(n)
    return None


def _parse_jersey_team_from_text(text: str) -> Optional[str]:
    """Extract team label A or B from Qwen JSON response."""
    if not text:
        return None
    try:
        obj = json.loads(str(text).strip())
        if isinstance(obj, dict):
            team = str(obj.get("team") or "").strip().upper()
            if team in ("A", "B"):
                return team
    except Exception:
        pass
    return None


def _snap_jersey_to_roster(num_str: str, valid_nums: List[str]) -> str:
    """Snap a jersey number to the nearest valid roster number."""
    if not valid_nums or not num_str or num_str == "-1":
        return num_str
    try:
        n = int(num_str)
    except ValueError:
        return num_str
    valid_ints: List[int] = []
    for v in valid_nums:
        try:
            valid_ints.append(int(v))
        except ValueError:
            pass
    if not valid_ints:
        return num_str
    return str(min(valid_ints, key=lambda x: abs(x - n)))

def _normalize_base_url(url: str) -> str:
    u = str(url or "").strip()
    if not u:
        return ""
    return u[:-1] if u.endswith("/") else u

def _normalize_model_name_token(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", str(value or "").strip().lower())

def _fetch_qwen_vl_models(base_url: str, timeout_sec: float = 10.0) -> List[str]:
    if httpx is None:
        return []

    base = _normalize_base_url(base_url)
    if not base:
        return []

    try:
        with httpx.Client(timeout=float(timeout_sec)) as client:
            resp = client.get(base + "/v1/models")
        if int(resp.status_code) >= 400:
            return []
        obj = resp.json()
    except Exception:
        return []

    candidates: List[str] = []
    seen: set[str] = set()

    def add_candidate(value: Any) -> None:
        text = str(value or "").strip()
        if not text or text in seen:
            return
        seen.add(text)
        candidates.append(text)

    data = []
    if isinstance(obj, dict):
        raw_data = obj.get("data")
        if isinstance(raw_data, list):
            data.extend(raw_data)
        raw_models = obj.get("models")
        if isinstance(raw_models, list):
            data.extend(raw_models)

    for item in data:
        if not isinstance(item, dict):
            continue
        add_candidate(item.get("id"))
        add_candidate(item.get("model"))
        add_candidate(item.get("name"))

    return candidates

def _resolve_qwen_vl_model_name(base_url: str, preferred_model: str) -> str:
    preferred = str(preferred_model or "").strip()
    models = _fetch_qwen_vl_models(base_url=base_url)
    if not models:
        return preferred or "Qwen3VL-8B-Instruct-Q4_K_M.gguf"

    if preferred:
        pref_norm = _normalize_model_name_token(preferred)
        for candidate in models:
            cand_norm = _normalize_model_name_token(candidate)
            if candidate == preferred or cand_norm == pref_norm or (pref_norm and pref_norm in cand_norm):
                return candidate

    return str(models[0])

def _qwen_vl_openai_compatible(
    *,
    crop_bgr: np.ndarray,
    base_url: str,
    model: str,
    prompt: str,
    timeout_sec: float = 30.0,
) -> Tuple[Optional[str], Optional[str]]:
    try:
        import base64
        import httpx
    except Exception:
        return None, None

    try:
        ok, buf = cv2.imencode(".png", crop_bgr)
        if not ok:
            return None, None
        b64 = base64.b64encode(buf.tobytes()).decode("utf-8")
        data_url = f"data:image/png;base64,{b64}"
    except Exception:
        return None, None

    endpoint = _normalize_base_url(base_url) + "/v1/chat/completions"
    payload = {
        "model": str(model or "qwen3vl8b"),
        "temperature": 0,
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {"type": "image_url", "image_url": {"url": data_url}},
                ],
            }
        ],
    }

    try:
        with httpx.Client(timeout=float(timeout_sec)) as client:
            r = client.post(endpoint, json=payload)
            if r.status_code >= 400:
                return None, None
            obj = r.json()
    except Exception:
        return None, None

    raw_text: Optional[str] = None
    try:
        raw_text = (
            obj.get("choices", [{}])[0]
            .get("message", {})
            .get("content")
        )
        if raw_text is None:
            raw_text = str(obj)
    except Exception:
        raw_text = None

    jersey = _parse_jersey_number_from_text(str(raw_text or ""))
    return jersey, raw_text


def _build_roster_hint(roster_json_str: Optional[str]) -> Tuple[str, Dict[str, List[str]]]:
    """Parse roster JSON and return (hint_text, team_label_to_numbers).

    team_label_to_numbers maps "A" / "B" to the list of valid jersey number strings.
    Returns ("", {}) if the input is missing or malformed.
    """
    if not roster_json_str:
        return "", {}
    try:
        data = json.loads(roster_json_str)
    except Exception:
        return "", {}
    rosters = data.get("rosters")
    if not isinstance(rosters, dict) or not rosters:
        return "", {}

    team_label_map: Dict[str, List[str]] = {}
    lines: List[str] = ["Valid jersey numbers in this match:"]
    label_letters = ("A", "B", "C")
    for idx, (team_key, players) in enumerate(rosters.items()):
        if not isinstance(players, list):
            continue
        numbers: List[str] = []
        for p in players:
            if isinstance(p, dict) and p.get("number") is not None:
                try:
                    numbers.append(str(int(p["number"])))
                except Exception:
                    pass
        if numbers:
            letter = label_letters[idx] if idx < len(label_letters) else str(idx)
            team_name = str(team_key).replace("_", " ").title()
            lines.append(f"- Team {letter} ({team_name}): {', '.join(numbers)}")
            team_label_map[letter] = numbers
    if len(lines) <= 1:
        return "", {}
    return "\n".join(lines), team_label_map


def _build_jersey_prompt_with_roster(
    base_prompt: str, roster_json_str: Optional[str]
) -> Tuple[str, Dict[str, List[str]]]:
    """Enhance base jersey prompt with valid number constraints and JSON output instruction.

    Returns (enhanced_prompt, team_label_to_numbers).
    """
    hint, team_label_map = _build_roster_hint(roster_json_str)
    if not hint:
        # No roster — keep digits-only prompt, no team info
        return base_prompt, {}

    enhanced = (
        base_prompt.rstrip()
        + "\n\n"
        + hint
        + "\n\nLook at the jersey COLOR to decide which team (A or B) the player belongs to."
        + "\nPick the closest matching number from that team's list."
        + '\nOutput ONLY valid JSON (no extra text): {"number": "X", "team": "A"}'
        + "\nIf no number is visible, output: {\"number\": \"-1\", \"team\": \"?\"}"
    )
    return enhanced, team_label_map


def _build_roster_lookup(roster_json_str: Optional[str]) -> Dict[str, str]:
    """Return a dict mapping jersey number string -> player name from the roster JSON."""
    if not roster_json_str:
        return {}
    try:
        data = json.loads(roster_json_str)
    except Exception:
        return {}
    rosters = data.get("rosters")
    if not isinstance(rosters, dict):
        return {}
    lookup: Dict[str, str] = {}
    for players in rosters.values():
        if not isinstance(players, list):
            continue
        for p in players:
            if not isinstance(p, dict):
                continue
            name = str(p.get("name") or "").strip()
            num = p.get("number")
            if name and num is not None:
                try:
                    lookup[str(int(num))] = name
                except Exception:
                    pass
    return lookup


def _build_match_context(
    roster_json_str: Optional[str],
    jersey_by_track: Optional[Dict[int, Any]] = None,
) -> Optional[Dict[str, Any]]:
    """Parse match_info + rosters from the mapping JSON.

    Also resolves team_id → team name by cross-referencing jersey_by_track with
    known roster numbers: whichever team's roster numbers appear most under a
    given team_id wins that label.

    Returns a dict like:
    {
        "teams": "Galatasaray vs Juventus",
        "competition": "UEFA Champions League",
        "date": "17.02.2026",
        "team_names": {0: "Galatasaray", 1: "Juventus"},
        "rosters": {
            "Galatasaray": [{"name": "...", "number": N}, ...],
            "Juventus": [...]
        }
    }
    or None if no roster JSON available.
    """
    if not roster_json_str:
        return None
    try:
        data = json.loads(roster_json_str)
    except Exception:
        return None

    match_info = data.get("match_info") if isinstance(data, dict) else {}
    match_info = match_info if isinstance(match_info, dict) else {}
    rosters_raw = data.get("rosters") if isinstance(data, dict) else {}
    rosters_raw = rosters_raw if isinstance(rosters_raw, dict) else {}

    if not match_info and not rosters_raw:
        return None

    # ── team_id → takım adı cross-reference ──────────────────────────────
    # Her takım anahtarı için jersey numaralarını set'e al
    roster_numbers: Dict[str, set] = {}
    for team_key, players in rosters_raw.items():
        if not isinstance(players, list):
            continue
        nums: set = set()
        for p in players:
            if isinstance(p, dict) and p.get("number") is not None:
                try:
                    nums.add(str(int(p["number"])))
                except Exception:
                    pass
        if nums:
            roster_numbers[str(team_key)] = nums

    team_names: Dict[int, str] = {}
    if jersey_by_track and roster_numbers:
        # Her jersey_by_track kaydı için hangi takıma ait olduğuna bak
        votes: Dict[int, Dict[str, int]] = {}  # team_id → {team_key: count}
        for tid_info in jersey_by_track.values():
            if not isinstance(tid_info, dict):
                continue
            jnum = str(tid_info.get("jersey_number") or "").strip()
            raw_tid = tid_info.get("team_id")
            if jnum in ("-1", "", None) or raw_tid is None:
                continue
            try:
                tid_int = int(raw_tid)
            except Exception:
                continue
            if tid_int < 0:
                continue
            for team_key, nums in roster_numbers.items():
                if jnum in nums:
                    v = votes.setdefault(tid_int, {})
                    v[team_key] = v.get(team_key, 0) + 1

        for tid_int, tally in votes.items():
            if tally:
                best_key = max(tally, key=lambda k: tally[k])
                team_names[tid_int] = str(best_key).replace("_", " ").title()

    # ── Roster'ı takım adıyla yeniden anahtarla ───────────────────────────
    rosters_out: Dict[str, Any] = {}
    for team_key, players in rosters_raw.items():
        label = str(team_key).replace("_", " ").title()
        rosters_out[label] = players

    ctx: Dict[str, Any] = {}
    if match_info.get("teams"):
        ctx["teams"] = str(match_info["teams"])
    if match_info.get("competition"):
        ctx["competition"] = str(match_info["competition"])
    if match_info.get("date"):
        ctx["date"] = str(match_info["date"])
    if team_names:
        ctx["team_names"] = {str(k): v for k, v in team_names.items()}
    if rosters_out:
        ctx["rosters"] = rosters_out

    return ctx if ctx else None


# ─────────────────────────────────────────────────────────────────────────────
# [1/3] T-DEED + EventEngine Metadata Merger
# ─────────────────────────────────────────────────────────────────────────────

# T-DEED etiketleri → SpeakerStateManager'ın "yüksek öncelikli" saydığı küme.
# Bu olaylar konuşma ortasında gelirse mevcut TTS iptal edilerek yeni olay okunur.
_HIGH_PRIORITY_TDEED_LABELS: FrozenSet[str] = frozenset({
    "goal", "own goal", "penalty - goal",
    "shot", "shot on target",
    "penalty",
    "red card",
    "foul",
    "var",
})


def merge_tdeed_with_event_engine(
    *,
    tdeed_event: Dict[str, Any],
    engine_events: List[EventEngineMeta],
    fps: float,
    frame_window: int = 90,          # ±3 saniye (30 fps'de 90 kare)
    max_context_events: int = 3,
) -> Dict[str, Any]:
    """T-DEED ana olayını, zaman bazında yakın EventEngine metadatasıyla birleştirir.

    Çatışma Çözümü (Conflict Resolution):
    - T-DEED olayı her zaman "main_event" (Ana Olay) olarak üst konuma yerleştirilir.
    - EventEngine verileri "context" listesi olarak eklenir (Renk/Bağlam).

    Dönen dict doğrudan :func:`_build_live_commentary_prompt` ve
    :func:`request_live_commentary_vllm` fonksiyonlarına gönderilebilir.
    """
    # T-DEED kare numarasını hesapla (yoksa zamandan çevir)
    tdeed_frame = int(tdeed_event.get("frame", 0))
    if not tdeed_frame and fps > 0:
        t = float(tdeed_event.get("t", tdeed_event.get("timestamp_sec", 0.0)))
        tdeed_frame = int(t * fps)

    # Zaman penceresindeki EventEngine olaylarını mesafeye + önceliğe göre sırala
    nearby: List[Tuple[int, EventEngineMeta]] = []
    for em in engine_events:
        dist = abs(em.frame - tdeed_frame)
        if dist <= frame_window:
            nearby.append((dist, em))

    nearby.sort(key=lambda x: (x[0], -x[1].priority))   # yakın ve yüksek önce
    context_events = [em for _, em in nearby[:max_context_events]]

    return {
        # ── Ana Olay (T-DEED) ──────────────────────────────────────────────
        "main_event": {
            "source": "tdeed",
            "label": str(tdeed_event.get("label") or tdeed_event.get("type") or ""),
            "confidence": float(
                tdeed_event.get("confidence", tdeed_event.get("score", 1.0)) or 1.0
            ),
            "frame": tdeed_frame,
            "timecode": str(
                tdeed_event.get("timecode")
                or _timecode_mmss(tdeed_frame / max(float(fps), 1.0))
            ),
        },
        # ── Bağlam (EventEngine) ───────────────────────────────────────────
        "context": [
            {
                "source": "event_engine",
                "priority": em.priority,
                "frame": em.frame,
                "text": em.event_text,
                **({"speed_ms": round(em.speed_ms, 1)} if em.speed_ms is not None else {}),
                **({"zone": em.zone} if em.zone else {}),
                **({"player_name": em.player_name} if em.player_name else {}),
                **({"track_id": em.track_id} if em.track_id is not None else {}),
            }
            for em in context_events
        ],
    }


# ─────────────────────────────────────────────────────────────────────────────
# [2/3] vLLM (Qwen3 8B) Canlı Yorum Bağlantısı
# ─────────────────────────────────────────────────────────────────────────────

_LIVE_COMMENTARY_SYSTEM_PROMPT: str = (
    "Sen gerçek zamanlı ve çok coşkulu bir futbol spikerisin. "
    "Sana T-DEED modelinden ana olay, EventEngine'den ise renk katacak yan oyuncu bilgileri gelecek. "
    "Görevin bu iki veriyi harmanlayıp kısa, tek ve akıcı bir spiker cümlesine çevirmek. "
    "Şut, Gol, Kırmızı Kart gibi olaylarda heyecanını maksimuma çıkar. "
    "'top ileri taşınıyor', 'topu ilerletiyor', 'topu taşıyor' gibi genel ifadeler KULLANMA; "
    "sahneye özgü futbol söylemleri seç (örn: 'hücuma çıkıyor', 'rakip ceza sahasına yaklaşıyor'). "
    "SADECE şu JSON'u döndür: {\"text\": \"...\"}"
)


def _build_live_commentary_prompt(
    merged: Dict[str, Any],
    roster_lookup: Optional[Dict[str, str]] = None,
) -> str:
    """Birleştirilmiş T-DEED+EventEngine objesinden vLLM user-turn prompt'u oluşturur."""
    main = merged.get("main_event") or {}
    ctx = merged.get("context") or []

    label = str(main.get("label", "")).strip()
    conf = float(main.get("confidence", 1.0))
    timecode = str(main.get("timecode", "")).strip()

    conf_str = f" (güven: %{round(conf * 100)})" if conf < 1.0 else ""
    tc_str = f" | {timecode}" if timecode else ""
    parts = [f"ANA OLAY: {label}{conf_str}{tc_str}"]

    if ctx:
        ctx_lines: List[str] = []
        for c in ctx:
            line = str(c.get("text", "")).strip()
            spd = c.get("speed_ms")
            pname = c.get("player_name")
            zone = c.get("zone")
            if pname:
                line = f"{pname}: {line}"
            if spd is not None:
                line += f" [{spd} m/s]"
            if zone:
                line += f" [{zone}]"
            ctx_lines.append(line)
        parts.append("BAĞLAM (EventEngine):\n" + "\n".join(f"- {l}" for l in ctx_lines))

    parts.append(
        "GÖREV: Yukarıdaki ANA OLAY ve BAĞLAM verisini harmanlayarak "
        "tek, akıcı ve coşkulu bir Türkçe spiker cümlesi yaz.\n"
        'ÇIKTI: Sadece {"text": "..."} JSON, başka hiçbir şey.'
    )

    return "\n\n".join(parts)


def request_live_commentary_vllm(
    *,
    merged_event: Dict[str, Any],
    base_url: str,
    model: str,
    roster_lookup: Optional[Dict[str, str]] = None,
    timeout_sec: float = 15.0,
    enable_thinking: bool = False,
) -> Tuple[Optional[str], Optional[str]]:
    """Birleştirilmiş olayı vLLM'e (Qwen3 8B, OpenAI compat.) gönderir.

    Dönüş: (commentary_text, error_str)
    """
    if httpx is None:
        return None, "httpx not available"

    base = _normalize_base_url(base_url)
    if not base:
        return None, "Invalid vLLM base URL"

    prompt = _build_live_commentary_prompt(merged_event, roster_lookup=roster_lookup)

    payload: Dict[str, Any] = {
        "model": str(model),
        "messages": [
            {"role": "system", "content": _LIVE_COMMENTARY_SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ],
        "temperature": 0.75,
        "max_tokens": 120,
    }
    if not enable_thinking:
        payload["chat_template_kwargs"] = {"enable_thinking": False}

    try:
        with httpx.Client(timeout=float(timeout_sec)) as client:
            r = client.post(base + "/v1/chat/completions", json=payload)
            r.raise_for_status()
            data = r.json()
    except Exception as e:
        return None, str(e)

    raw = ""
    try:
        raw = str(data["choices"][0]["message"]["content"] or "").strip()
    except Exception:
        pass

    if not enable_thinking:
        raw = _strip_think_blocks(raw)

    text = _extract_commentary_text_best_effort(raw)
    return text, None


# ─────────────────────────────────────────────────────────────────────────────
# [3/3] Spiker Durum Yöneticisi (SpeakerStateManager)
# ─────────────────────────────────────────────────────────────────────────────

class SpeakerStateManager:
    """Thread-safe TTS concurrency guard (Spiker Koruma Katmanı).

    KORUMA 1 – Ignore (Çöpe At):
        Spiker konuşuyorsa ve gelen olay T-DEED kaynaklı DEĞİLSE
        (priority 1/2 EventEngine olayı), sessizce iptal edilir.

    KORUMA 2 – Interrupt (Kes ve Geç):
        Spiker konuşuyor olsa bile gelen olay T-DEED'den yüksek
        öncelikli (Goal, Shot, Foul …) ise mevcut TTS thread'i
        stop_event ile hemen durdurulur ve yeni olay okutulur.

    Kullanım::

        speaker = SpeakerStateManager()

        # TTS fonksiyonu stop_event parametresi almalıdır:
        def my_tts(text: str, stop_event: threading.Event) -> None:
            ...

        dispatched = speaker.dispatch(merged_event, my_tts, text="Şut!")
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self.is_speaking: bool = False
        self._current_thread: Optional[threading.Thread] = None
        self._stop_event: threading.Event = threading.Event()

    # ── dahili yardımcılar ────────────────────────────────────────────────────

    def _is_high_priority(self, event: Dict[str, Any]) -> bool:
        source = str(event.get("source", "")).lower()
        label = str(
            event.get("label")
            or (event.get("main_event") or {}).get("label")
            or ""
        ).strip().lower()
        return source == "tdeed" and label in _HIGH_PRIORITY_TDEED_LABELS

    def _is_tdeed_source(self, event: Dict[str, Any]) -> bool:
        source = str(event.get("source", "")).lower()
        if source == "tdeed":
            return True
        # merge_tdeed_with_event_engine çıktısına da bak
        main = event.get("main_event") or {}
        return str(main.get("source", "")).lower() == "tdeed"

    # ── public API ─────────────────────────────────────────────────────────────

    def dispatch(
        self,
        event: Dict[str, Any],
        tts_fn: Callable[..., Any],
        *tts_args: Any,
        **tts_kwargs: Any,
    ) -> bool:
        """Yeni TTS işini kuyruğa almaya çalışır; koruma kurallarına göre reddedebilir.

        :param event:      merge_tdeed_with_event_engine() çıktısı (veya source/label içeren dict).
        :param tts_fn:     Senkron TTS fonksiyonu.  ``stop_event`` keyword argümanını ALMALI.
        :param tts_args:   tts_fn'e geçirilecek pozisyonel argümanlar.
        :param tts_kwargs: tts_fn'e geçirilecek keyword argümanlar (stop_event hariç).
        :returns:          True → iş başlatıldı,  False → KORUMA 1 nedeniyle çöpe atıldı.
        """
        is_tdeed = self._is_tdeed_source(event)
        high_priority = self._is_high_priority(event)

        prev_thread: Optional[threading.Thread] = None

        with self._lock:
            if self.is_speaking:
                if not is_tdeed:
                    # ── KORUMA 1: EventEngine olayı, konuşma var → çöpe at ──
                    return False
                if not high_priority:
                    # T-DEED ama düşük öncelikli (Throw-in, Ball out of play …)
                    # konuşma ortasında kesme, bırak bitsin
                    return False
                # ── KORUMA 2: Yüksek öncelikli T-DEED → kes ──────────────
                self._stop_event.set()          # mevcut thread'e dur sinyali
                prev_thread = self._current_thread
                self._current_thread = None
                self.is_speaking = False

        # join() lock dışında — deadlock riski yok
        if prev_thread is not None and prev_thread.is_alive():
            prev_thread.join(timeout=1.0)

        # Her iş için ayrı bir stop_event; eski sinyal temizlenir
        stop_ev = threading.Event()

        def _run() -> None:
            try:
                tts_fn(*tts_args, stop_event=stop_ev, **tts_kwargs)
            except Exception:
                pass
            finally:
                with self._lock:
                    # Sadece bu thread hâlâ aktifse temizle
                    if self._current_thread is threading.current_thread():
                        self.is_speaking = False
                        self._current_thread = None

        with self._lock:
            self._stop_event = stop_ev
            t = threading.Thread(target=_run, daemon=True, name="live-tts-worker")
            self._current_thread = t
            self.is_speaking = True

        t.start()
        return True

    def wait_until_done(self, timeout: float = 30.0) -> None:
        """Mevcut TTS işi bitene kadar bekler."""
        with self._lock:
            t = self._current_thread
        if t is not None:
            t.join(timeout=timeout)


# ─────────────────────────────────────────────────────────────────────────────
# Ana Orkestrasyoncu: dispatch_live_event
# ─────────────────────────────────────────────────────────────────────────────

def dispatch_live_event(
    *,
    tdeed_event: Dict[str, Any],
    engine_events: List[EventEngineMeta],
    fps: float,
    vllm_base_url: str,
    vllm_model: str,
    speaker: SpeakerStateManager,
    tts_fn: Callable[..., Any],
    roster_lookup: Optional[Dict[str, str]] = None,
    frame_window: int = 90,
    llm_timeout_sec: float = 15.0,
) -> bool:
    """T-DEED olayını + EventEngine bağlamını vLLM'e gönderir ve TTS'e iletir.

    Hızlı yol: Spiker konuşuyorsa ve bu yüksek öncelikli T-DEED DEĞİLSE, LLM
    çağrısı yapılmadan direkt ``False`` döner (gereksiz GPU zamanı harcanmaz).

    Tipik çağrı (tracking/spotting döngüsü içinden)::

        speaker = SpeakerStateManager()  # pipeline başlangıcında bir kez

        for tdeed_ev in tdeed_events:
            dispatch_live_event(
                tdeed_event=tdeed_ev,
                engine_events=current_engine_buffer,
                fps=video_fps,
                vllm_base_url=cfg.commentary_llm_url,
                vllm_model=cfg.commentary_llm_model,
                speaker=speaker,
                tts_fn=my_tts_speak,
                roster_lookup=roster_lookup,
            )
    """
    # Hızlı ön kontrol — LLM maliyetinden önce filtrele
    mock_event = {
        "source": "tdeed",
        "label": str(tdeed_event.get("label") or tdeed_event.get("type") or ""),
    }
    is_tdeed = True
    high_priority = speaker._is_high_priority(mock_event)

    if speaker.is_speaking and not high_priority:
        # KORUMA 1 hızlı yolu: LLM bile çağrılmaz
        return False

    # Metadata birleştir
    merged = merge_tdeed_with_event_engine(
        tdeed_event=tdeed_event,
        engine_events=engine_events,
        fps=fps,
        frame_window=frame_window,
    )

    # LLM'den yorum al (bloke edici ~1–5 s)
    text, err = request_live_commentary_vllm(
        merged_event=merged,
        base_url=vllm_base_url,
        model=vllm_model,
        roster_lookup=roster_lookup,
        timeout_sec=llm_timeout_sec,
    )
    if not text or err:
        return False

    # SpeakerStateManager üzerinden TTS'e ilet (koruma kuralları burada uygulanır)
    dispatch_event = dict(merged)
    dispatch_event["source"] = "tdeed"
    dispatch_event["label"] = str(mock_event["label"])
    return speaker.dispatch(dispatch_event, tts_fn, text=text)


def _qwen_vl_ask_jersey_number(
    *,
    crop_bgr: np.ndarray,
    qwen_url: str,
    model: Optional[str] = None,
    prompt: Optional[str] = None,
) -> Tuple[Optional[str], Optional[str]]:
    prompt = prompt or (
        "You are reading a football jersey.\n\n"
        "Visually read the number printed on the player's shirt.\n\n"
        "Rules:\n\n"
        "* Output digits only.\n"
        "* If no number is visible or readable, output: -1\n"
        "* Do not guess.\n"
        "* Do not describe the image.\n"
        "* Do not output words.\n\n"
        "Return only the final answer.\n"
    )

    jersey_http, raw_http = _qwen_vl_openai_compatible(
        crop_bgr=crop_bgr,
        base_url=str(qwen_url),
        model=str(model or "qwen3vl8b"),
        prompt=str(prompt),
    )
    if jersey_http is not None:
        return jersey_http, raw_http

    try:
        from gradio_client import Client, handle_file  # type: ignore
    except Exception:
        return None, None

    try:
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tf:
            tmp_path = tf.name
        ok, buf = cv2.imencode(".png", crop_bgr)
        if not ok:
            try:
                os.remove(tmp_path)
            except Exception:
                pass
            return None, None
        with open(tmp_path, "wb") as f:
            f.write(buf.tobytes())
    except Exception:
        return None, None

    raw_text: Optional[str] = None
    try:
        client = Client(qwen_url)
        img = handle_file(tmp_path)

        attempts: List[Tuple[Optional[str], Tuple[Any, ...]]] = [
            ("/predict", (prompt, img)),
            ("/predict", (img, prompt)),
            ("/chat", (prompt, img)),
            ("/chat", (img, prompt)),
            (None, (prompt, img)),
            (None, (img, prompt)),
        ]

        last_exc: Optional[Exception] = None
        for api_name, args in attempts:
            try:
                if api_name is None:
                    res = client.predict(*args)
                else:
                    res = client.predict(*args, api_name=api_name)

                if isinstance(res, str):
                    raw_text = res
                elif isinstance(res, (list, tuple)) and res:
                    cand = None
                    for item in res:
                        if isinstance(item, str):
                            cand = item
                            break
                        if isinstance(item, dict) and "text" in item and isinstance(item["text"], str):
                            cand = item["text"]
                            break
                    if cand is None:
                        raw_text = str(res[0])
                    else:
                        raw_text = cand
                elif isinstance(res, dict):
                    raw_text = str(res.get("text") or res.get("output") or res)
                else:
                    raw_text = str(res)

                jersey = _parse_jersey_number_from_text(raw_text)
                return jersey, raw_text
            except Exception as e:
                last_exc = e
                continue

        _ = last_exc
        return None, raw_text
    finally:
        try:
            os.remove(tmp_path)
        except Exception:
            pass

def _select_jersey_samples_from_tracks_csv(
    *,
    tracks_csv_path: str,
    cfg: FullPipelineConfig,
) -> Dict[int, List[Dict[str, Any]]]:
    per_track: Dict[int, List[Dict[str, Any]]] = {}

    def can_add(existing: List[Dict[str, Any]], frame_id: int) -> bool:
        gap = max(1, int(cfg.jersey_min_frame_gap))
        for s in existing:
            try:
                if abs(int(s["frame_id"]) - int(frame_id)) < gap:
                    return False
            except Exception:
                continue
        return True

    with open(tracks_csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                cls_id = int(float(row.get("cls_id", -1)))
                if cls_id != int(cfg.player_cls_id):
                    continue
                track_id = int(float(row.get("track_id", 0)))
                if track_id <= 0:
                    continue
                if _is_special_track_id(track_id):
                    continue
                frame_id = int(float(row.get("frame_id", 0)))
                conf = float(row.get("conf", 0.0) or 0.0)
                if conf < float(cfg.jersey_min_det_conf):
                    continue

                x1 = int(float(row.get("x1", 0)))
                y1 = int(float(row.get("y1", 0)))
                x2 = int(float(row.get("x2", 0)))
                y2 = int(float(row.get("y2", 0)))
                area = max(0, x2 - x1) * max(0, y2 - y1)
                if area < int(cfg.jersey_min_box_area):
                    continue

                score = float(area) * float(conf)
            except Exception:
                continue

            lst = per_track.setdefault(track_id, [])
            if not can_add(lst, frame_id):
                continue
            lst.append(
                {
                    "frame_id": frame_id,
                    "x1": x1,
                    "y1": y1,
                    "x2": x2,
                    "y2": y2,
                    "conf": conf,
                    "score": score,
                }
            )
            lst.sort(key=lambda s: float(s.get("score", 0.0)), reverse=True)
            if len(lst) > int(cfg.jersey_max_samples_per_track):
                del lst[int(cfg.jersey_max_samples_per_track) :]

    track_scores: List[Tuple[float, int]] = []
    for tid, samples in per_track.items():
        if not samples:
            continue
        track_scores.append((float(samples[0].get("score", 0.0)), tid))
    track_scores.sort(reverse=True)

    keep = set(tid for _, tid in track_scores[: int(cfg.jersey_max_tracks)])
    return {tid: per_track[tid] for tid in keep if tid in per_track}

def infer_jersey_numbers_from_tracking(
    *,
    video_path: str,
    tracks_csv_path: str,
    out_dir: str,
    cfg: FullPipelineConfig,
    progress_cb: Optional[Callable[[str, int, int, str], None]] = None,
) -> Dict[int, Dict[str, Any]]:
    if not cfg.run_jersey_number_recognition:
        return {}

    if not os.path.isfile(tracks_csv_path) or not os.path.isfile(video_path):
        return {}

    by_frame = _parse_tracks_csv(tracks_csv_path)
    if not by_frame:
        return {}

    frame_ids = sorted(by_frame.keys())

    track_best_score: Dict[int, float] = {}
    track_first_frame: Dict[int, int] = {}
    track_last_frame: Dict[int, int] = {}
    track_team_counts: Dict[int, Dict[int, int]] = {}

    for fid in frame_ids:
        rows = by_frame.get(fid, [])
        for r in rows:
            try:
                cls_id = int(float(r.get("cls_id", -1)))
            except Exception:
                continue
            if cls_id != int(cfg.player_cls_id):
                continue
            try:
                track_id = int(float(r.get("track_id", 0)))
            except Exception:
                continue
            if track_id <= 0:
                continue
            if _is_special_track_id(track_id):
                continue
            try:
                conf = float(r.get("conf", 0.0))
            except Exception:
                conf = 0.0
            try:
                x1 = float(r.get("x1", 0.0))
                y1 = float(r.get("y1", 0.0))
                x2 = float(r.get("x2", 0.0))
                y2 = float(r.get("y2", 0.0))
                area = max(0.0, x2 - x1) * max(0.0, y2 - y1)
            except Exception:
                continue
            if area <= 1.0:
                continue

            score = float(area) * float(conf)
            prev = float(track_best_score.get(track_id, 0.0))
            if score > prev:
                track_best_score[track_id] = score

            track_first_frame[track_id] = min(int(track_first_frame.get(track_id, fid)), int(fid))
            track_last_frame[track_id] = max(int(track_last_frame.get(track_id, fid)), int(fid))

            team_raw = r.get("team_id", None)
            if team_raw is not None and str(team_raw).strip() != "":
                try:
                    team_id = int(float(team_raw))
                    counts = track_team_counts.setdefault(track_id, {})
                    counts[team_id] = int(counts.get(team_id, 0)) + 1
                except Exception:
                    pass

    if not track_best_score:
        return {}

    limit = int(getattr(cfg, "jersey_max_tracks", 0) or 0)
    if limit <= 0:
        keep_tracks = set(int(tid) for tid in track_best_score.keys())
    else:
        top = sorted(((s, tid) for tid, s in track_best_score.items()), reverse=True)
        keep_tracks = set(tid for _, tid in top[:limit])

    track_team: Dict[int, Optional[int]] = {}
    for tid, counts in track_team_counts.items():
        if not counts:
            track_team[tid] = None
            continue
        best_team = sorted(counts.items(), key=lambda kv: (kv[1], -kv[0]), reverse=True)[0][0]
        track_team[tid] = int(best_team)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return {}
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)

    crop_dir = str(Path(out_dir) / "jersey_crops")
    os.makedirs(crop_dir, exist_ok=True)

    gap = max(1, int(cfg.jersey_min_frame_gap))
    max_attempts = max(1, int(cfg.jersey_max_samples_per_track))

    per_track_votes: Dict[int, Dict[str, int]] = {}
    per_track_raw: Dict[int, List[str]] = {}
    per_track_attempts: Dict[int, int] = {}
    per_track_last_query: Dict[int, int] = {}
    jersey_final: Dict[int, str] = {}
    jersey_conf: Dict[int, float] = {}
    jersey_source: Dict[int, int] = {}
    resolved_qwen_model = _resolve_qwen_vl_model_name(
        base_url=str(getattr(cfg, "qwen_vl_url", "") or ""),
        preferred_model=str(getattr(cfg, "qwen_vl_model", "") or "Qwen3VL-8B-Instruct-Q4_K_M.gguf"),
    )

    # Build roster-enhanced prompt if a mapping JSON was supplied.
    _roster_json_str: Optional[str] = getattr(cfg, "player_roster_json", None) or None
    _effective_jersey_prompt, _roster_team_label_map = _build_jersey_prompt_with_roster(
        str(getattr(cfg, "jersey_prompt", "") or ""),
        _roster_json_str,
    )
    # Also build a number->name lookup for post-resolution enrichment.
    _roster_lookup: Dict[str, str] = _build_roster_lookup(_roster_json_str)
    # Per-track team label votes (A/B from Qwen color detection)
    per_track_team_votes: Dict[int, Dict[str, int]] = {}

    done = 0
    total_est = int(len(keep_tracks) * max_attempts)

    def best_vote(votes: Dict[str, int]) -> Tuple[Optional[str], float, int, int]:
        if not votes:
            return (None, 0.0, 0, 0)
        items = sorted(votes.items(), key=lambda kv: (kv[1], kv[0]), reverse=True)
        best_lbl, best_cnt = items[0][0], int(items[0][1])
        tot = int(sum(int(x) for x in votes.values()))
        conf = float(best_cnt) / float(max(1, tot))
        return (str(best_lbl), conf, best_cnt, tot)

    for fid in frame_ids:
        rows = by_frame.get(fid, [])

        cand: Dict[int, Dict[str, Any]] = {}
        cand_score: Dict[int, float] = {}
        cand_conf: Dict[int, float] = {}
        cand_area: Dict[int, float] = {}
        for r in rows:
            try:
                cls_id = int(float(r.get("cls_id", -1)))
            except Exception:
                continue
            if cls_id != int(cfg.player_cls_id):
                continue
            try:
                track_id = int(float(r.get("track_id", 0)))
            except Exception:
                continue
            if track_id <= 0 or track_id not in keep_tracks:
                continue
            if track_id in jersey_final:
                continue

            try:
                conf = float(r.get("conf", 0.0))
            except Exception:
                conf = 0.0

            try:
                x1 = float(r.get("x1", 0.0))
                y1 = float(r.get("y1", 0.0))
                x2 = float(r.get("x2", 0.0))
                y2 = float(r.get("y2", 0.0))
                area = max(0.0, x2 - x1) * max(0.0, y2 - y1)
            except Exception:
                continue
            if area <= 1.0:
                continue
            score = float(area) * float(conf)
            prev = float(cand_score.get(track_id, -1.0))
            if score > prev:
                cand_score[track_id] = score
                cand[track_id] = r
                cand_conf[track_id] = float(conf)
                cand_area[track_id] = float(area)

        for track_id, r in list(cand.items()):
            src_raw = r.get("relink_source_id", None)
            if src_raw is None or str(src_raw).strip() == "":
                continue
            try:
                src_id = int(float(src_raw))
            except Exception:
                continue
            if src_id > 0 and src_id in jersey_final:
                jersey_final[track_id] = str(jersey_final[src_id])
                jersey_conf[track_id] = float(jersey_conf.get(src_id, 1.0))
                jersey_source[track_id] = int(src_id)

        to_query: List[Tuple[int, Dict[str, Any]]] = []
        for track_id, r in cand.items():
            if track_id in jersey_final:
                continue
            attempts = int(per_track_attempts.get(track_id, 0))
            if attempts >= max_attempts:
                continue
            last_q = per_track_last_query.get(track_id, None)
            if last_q is not None and int(fid) - int(last_q) < gap:
                continue

            if attempts > 0:
                c = float(cand_conf.get(track_id, 0.0))
                a = float(cand_area.get(track_id, 0.0))
                if c < float(cfg.jersey_min_det_conf) or a < float(cfg.jersey_min_box_area):
                    continue
            to_query.append((track_id, r))

        if not to_query:
            continue

        try:
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(fid))
            ret, frame = cap.read()
        except Exception:
            ret, frame = False, None
        if not ret or frame is None:
            continue

        if (w and h) and (frame.shape[1] != w or frame.shape[0] != h):
            try:
                frame = cv2.resize(frame, (w, h))
            except Exception:
                pass

        for track_id, r in to_query:
            try:
                x1 = int(float(r.get("x1", 0)))
                y1 = int(float(r.get("y1", 0)))
                x2 = int(float(r.get("x2", 0)))
                y2 = int(float(r.get("y2", 0)))
            except Exception:
                continue

            crop = _jersey_crop_from_player_bbox(frame, x1=x1, y1=y1, x2=x2, y2=y2)
            if crop is None:
                continue

            crop_path = str(Path(crop_dir) / f"track{track_id}_frame{fid}.png")
            try:
                cv2.imwrite(crop_path, crop)
            except Exception:
                pass

            per_track_last_query[int(track_id)] = int(fid)
            per_track_attempts[int(track_id)] = int(per_track_attempts.get(int(track_id), 0)) + 1

            jersey, raw = _qwen_vl_ask_jersey_number(
                crop_bgr=crop,
                qwen_url=str(cfg.qwen_vl_url),
                model=str(resolved_qwen_model or "Qwen3VL-8B-Instruct-Q4_K_M.gguf"),
                prompt=_effective_jersey_prompt,
            )
            if raw:
                per_track_raw.setdefault(int(track_id), []).append(str(raw))
                team_label = _parse_jersey_team_from_text(str(raw))
                if team_label:
                    tv = per_track_team_votes.setdefault(int(track_id), {})
                    tv[team_label] = tv.get(team_label, 0) + 1

            done += 1
            if progress_cb is not None:
                try:
                    progress_cb("jersey", done, max(1, total_est), f"Forma no okunuyor ({done})")
                except Exception:
                    pass

            if jersey is None:
                continue
            jersey_s = str(jersey).strip()
            if jersey_s == "-1":
                continue

            votes = per_track_votes.setdefault(int(track_id), {})
            votes[jersey_s] = int(votes.get(jersey_s, 0)) + 1
            best_lbl, conf, best_cnt, tot = best_vote(votes)

            if best_lbl is not None and best_cnt >= 1:
                jersey_final[int(track_id)] = str(best_lbl)
                jersey_conf[int(track_id)] = float(conf)

    cap.release()

    out: Dict[int, Dict[str, Any]] = {}
    attempted_ids = set(int(t) for t in per_track_attempts.keys())
    for tid in attempted_ids:
        jersey_num = str(jersey_final.get(int(tid), "-1")).strip()
        votes = per_track_votes.get(int(tid), {})
        raws = per_track_raw.get(int(tid), [])

        # Team-aware roster snap: if Qwen identified a team label and we have roster numbers,
        # snap the jersey number to the closest valid number in that team's list.
        if jersey_num != "-1" and _roster_team_label_map:
            team_votes = per_track_team_votes.get(int(tid), {})
            if team_votes:
                best_team = max(team_votes, key=lambda k: team_votes[k])
                team_nums = _roster_team_label_map.get(best_team, [])
                if team_nums:
                    snapped = _snap_jersey_to_roster(jersey_num, team_nums)
                    if snapped != jersey_num:
                        jersey_num = snapped

        player_name = _roster_lookup.get(jersey_num) if jersey_num != "-1" else None
        out[int(tid)] = {
            "track_id": int(tid),
            "jersey_number": jersey_num,
            "confidence": float(jersey_conf.get(int(tid), 0.0 if jersey_num == "-1" else 1.0)),
            "votes": votes,
            "raw": raws[-3:],
            **({"player_name": str(player_name)} if player_name else {}),
            **({"relink_source_id": int(jersey_source[tid])} if int(tid) in jersey_source else {}),
            **(
                {"team_id": int(track_team[tid])}
                if tid in track_team and track_team[tid] is not None
                else {}
            ),
            **(
                {"first_frame": int(track_first_frame.get(tid, 0)), "last_frame": int(track_last_frame.get(tid, 0))}
                if tid in track_first_frame and tid in track_last_frame
                else {}
            ),
        }

    return out

def _resolve_track_id_remap(track_id: int, remap: Dict[int, int]) -> int:
    cur = int(track_id)
    seen = set()
    while cur in remap and cur not in seen:
        seen.add(cur)
        cur = int(remap[cur])
    return int(cur)

def _build_track_id_remap_from_jerseys(
    *,
    jersey_by_track: Dict[int, Dict[str, Any]],
    cfg: FullPipelineConfig,
) -> Dict[int, int]:

    if not bool(getattr(cfg, "jersey_merge_same_number", False)):
        return {}

    min_conf = float(getattr(cfg, "jersey_merge_min_confidence", 0.6) or 0.6)
    max_overlap = int(getattr(cfg, "jersey_merge_max_overlap_frames", 5) or 5)

    groups: Dict[Tuple[int, str], List[int]] = {}
    for tid, info in (jersey_by_track or {}).items():
        try:
            jersey = str(info.get("jersey_number", "")).strip()
            if not jersey or jersey == "-1":
                continue
            conf = float(info.get("confidence", 0.0))
            if conf < min_conf:
                continue
            team_raw = info.get("team_id", None)
            if team_raw is None:
                continue
            team_id = int(team_raw)
        except Exception:
            continue
        groups.setdefault((team_id, jersey), []).append(int(tid))

    remap: Dict[int, int] = {}

    def lifetime(tid: int) -> Tuple[int, int]:
        info = jersey_by_track.get(int(tid), {})
        try:
            a = int(info.get("first_frame", 0))
            b = int(info.get("last_frame", 0))
        except Exception:
            return (0, 0)
        return (min(a, b), max(a, b))

    for (_, _), tids in groups.items():
        uniq = sorted(set(int(x) for x in tids if int(x) > 0))
        if len(uniq) < 2:
            continue

        ok = True
        ranges = {tid: lifetime(tid) for tid in uniq}
        for i in range(len(uniq)):
            for j in range(i + 1, len(uniq)):
                a0, a1 = ranges[uniq[i]]
                b0, b1 = ranges[uniq[j]]
                overlap = max(0, min(a1, b1) - max(a0, b0))
                if overlap > max_overlap:
                    ok = False
                    break
            if not ok:
                break
        if not ok:
            continue

        uniq.sort(key=lambda tid: (lifetime(tid)[0], tid))
        canonical = int(uniq[0])
        for tid in uniq[1:]:
            remap[int(tid)] = canonical

    for k in list(remap.keys()):
        remap[k] = _resolve_track_id_remap(remap[k], remap)
    return remap

def extract_segment_to_mp4(*, src_path: str, out_path: str, start_sec: float, duration_sec: Optional[float]) -> str:
    src_path = str(Path(src_path).resolve())
    out_path = str(Path(out_path).resolve())

    out_dir = str(Path(out_path).parent)
    os.makedirs(out_dir, exist_ok=True)

    ffmpeg_bin = _ffmpeg_exe()
    if ffmpeg_bin:
        pbar = tqdm(total=1, desc="Segment", unit="job")
        cmd = [
            ffmpeg_bin,
            "-y",
            "-ss",
            str(float(start_sec or 0.0)),
            "-i",
            src_path,
        ]
        if duration_sec is not None:
            cmd += ["-t", str(float(duration_sec))]
        cmd += [
            "-c:v",
            "libx264",
            "-preset",
            "fast",
            "-c:a",
            "aac",
            "-movflags",
            "+faststart",
            out_path,
        ]
        subprocess.run(cmd, capture_output=True, text=True, check=True)
        try:
            pbar.update(1)
            pbar.close()
        except Exception:
            pass
    else:
        cap = cv2.VideoCapture(src_path)
        if not cap.isOpened():
            raise RuntimeError(f"Failed to open video: {src_path}")

        fps = float(cap.get(cv2.CAP_PROP_FPS) or 25.0)
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 640)
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 480)

        start_frame = int(float(start_sec or 0.0) * fps)
        if duration_sec is None:
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
            frames_to_write = max(0, total_frames - start_frame)
        else:
            frames_to_write = int(float(duration_sec) * fps)

        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(out_path, fourcc, fps, (w, h))
        if not writer.isOpened():
            cap.release()
            raise RuntimeError(f"Failed to create writer: {out_path}")

        written = 0
        pbar = tqdm(total=max(1, int(frames_to_write)), desc="Segment", unit="frame")
        for _ in range(frames_to_write):
            ret, frame = cap.read()
            if not ret or frame is None:
                break
            if frame.shape[1] != w or frame.shape[0] != h:
                frame = cv2.resize(frame, (w, h))
            writer.write(frame)
            written += 1
            try:
                pbar.update(1)
            except Exception:
                pass
        try:
            pbar.close()
        except Exception:
            pass

        writer.release()
        cap.release()
        if written == 0:
            try:
                os.remove(out_path)
            except Exception:
                pass
            raise RuntimeError("No frames written during extraction")

    if not os.path.isfile(out_path) or os.path.getsize(out_path) == 0:
        raise RuntimeError(f"Extraction failed, output missing: {out_path}")

    return out_path

def _which(name: str) -> Optional[str]:
    exts = [""]
    if os.name == "nt":
        exts = [".exe", ".cmd", ".bat", ""]

    paths = os.environ.get("PATH", "").split(os.pathsep)
    for base in paths:
        base = base.strip('"')
        for ext in exts:
            p = os.path.join(base, name + ext)
            if os.path.isfile(p):
                return p
    return None

def _ffmpeg_exe() -> Optional[str]:
    p = _which("ffmpeg")
    if p:
        return p
    try:
        import imageio_ffmpeg  # type: ignore
        exe = imageio_ffmpeg.get_ffmpeg_exe()
        if exe and os.path.isfile(exe):
            return exe
    except Exception:
        pass
    # Conda env fallback: aktif env veya bilinen konumları tara
    _conda_candidates = []
    _conda_prefix = os.environ.get("CONDA_PREFIX") or os.environ.get("CONDA_DEFAULT_ENV")
    if _conda_prefix and os.path.isdir(_conda_prefix):
        _conda_candidates.append(os.path.join(_conda_prefix, "Library", "bin", "ffmpeg.exe"))
    # Bilinen miniconda env yolları
    _home = os.path.expanduser("~")
    for _base in [
        os.path.join(_home, "miniconda3"),
        os.path.join(_home, "anaconda3"),
        r"C:\ProgramData\miniconda3",
        r"C:\ProgramData\anaconda3",
    ]:
        for _env in ["envs\\yolov8", "envs\\torch", ""]:
            _conda_candidates.append(os.path.join(_base, _env, "Library", "bin", "ffmpeg.exe"))
    for _c in _conda_candidates:
        if _c and os.path.isfile(_c):
            return _c
    return None

def _docker_exe() -> Optional[str]:
    return _which("docker")

def _docker_container_state(container_id: str) -> Tuple[Optional[str], Optional[str]]:
    docker_bin = _docker_exe()
    if not docker_bin:
        return None, "docker executable not found"

    cid = str(container_id or "").strip()
    if not cid:
        return None, "container id is empty"

    cmd = [docker_bin, "inspect", "-f", "{{.State.Status}}", cid]
    try:
        res = subprocess.run(cmd, capture_output=True, text=True, check=True)
    except Exception as e:
        return None, str(e)

    state = str((res.stdout or "").strip()).lower()
    return (state or None), None

def _wait_for_http_ready(
    *,
    base_url: str,
    timeout_sec: float = 60.0,
    paths: Optional[List[str]] = None,
) -> Dict[str, Any]:
    if httpx is None:
        return {"ok": False, "error": "httpx unavailable"}

    base = _normalize_base_url(base_url)
    if not base:
        return {"ok": False, "error": "base url is empty"}

    targets = paths or ["/v1/models", "/"]
    deadline = time.time() + max(1.0, float(timeout_sec))
    last_error: Optional[str] = None

    while time.time() <= deadline:
        for path in targets:
            try:
                with httpx.Client(timeout=5.0) as client:
                    resp = client.get(base + path)
                if 200 <= int(resp.status_code) < 500:
                    return {"ok": True, "url": base + path, "status_code": int(resp.status_code)}
            except Exception as e:
                last_error = str(e)
        time.sleep(0.5)

    return {"ok": False, "error": last_error or "timeout waiting for http readiness"}

def _wait_for_docker_container_state(
    *,
    container_id: str,
    desired_states: List[str],
    timeout_sec: float = 60.0,
    poll_sec: float = 0.5,
) -> Dict[str, Any]:
    wanted = {str(x or "").strip().lower() for x in desired_states if str(x or "").strip()}
    deadline = time.time() + max(1.0, float(timeout_sec))
    last_state: Optional[str] = None
    last_error: Optional[str] = None

    while time.time() <= deadline:
        state, err = _docker_container_state(container_id)
        last_state = state
        last_error = err
        if state in wanted:
            return {
                "ok": True,
                "state": state,
            }
        time.sleep(max(0.1, float(poll_sec)))

    return {
        "ok": False,
        "state": last_state,
        "error": last_error or f"timeout waiting for states {sorted(wanted)}",
    }

def _set_qwen_vl_container_state(
    *,
    cfg: FullPipelineConfig,
    action: str,
    progress_cb: Optional[Callable[[str, int, int, str], None]] = None,
) -> Dict[str, Any]:
    info: Dict[str, Any] = {
        "action": str(action),
        "enabled": False,
    }

    manage = bool(getattr(cfg, "qwen_vl_manage_container", False))
    cid = str(getattr(cfg, "qwen_vl_container_id", "") or "").strip()
    info["enabled"] = manage
    info["container_id"] = cid
    if not manage:
        info["skipped"] = "container management disabled"
        return info
    if not cid:
        info["skipped"] = "container id not configured"
        return info

    docker_bin = _docker_exe()
    if not docker_bin:
        info["error"] = "docker executable not found"
        return info

    before, before_err = _docker_container_state(cid)
    info["state_before"] = before
    if before_err:
        info["state_before_error"] = before_err

    desired = str(action or "").strip().lower()
    if desired not in ("start", "stop"):
        info["error"] = f"unsupported action: {desired}"
        return info

    if desired == "start" and before == "running":
        info["changed"] = False
        info["state_after"] = before
        ready = _wait_for_http_ready(
            base_url=str(getattr(cfg, "qwen_vl_url", "") or ""),
            timeout_sec=float(getattr(cfg, "qwen_vl_ready_timeout_sec", 60.0) or 60.0),
        )
        info["http_ready"] = ready
        if not bool(ready.get("ok")):
            info["error"] = str(ready.get("error") or "qwen vl http server not ready")
        return info
    if desired == "stop" and before in ("exited", "created", "dead"):
        info["changed"] = False
        info["state_after"] = before
        return info

    if progress_cb is not None:
        try:
            msg = "Qwen-VL container başlatılıyor" if desired == "start" else "Qwen-VL container durduruluyor"
            progress_cb("qwen_vl_container", 0, 1, msg)
        except Exception:
            pass

    cmd = [docker_bin, desired, cid]
    try:
        res = subprocess.run(cmd, capture_output=True, text=True, check=True)
        info["stdout"] = str(res.stdout or "").strip()
        info["stderr"] = str(res.stderr or "").strip()
        info["changed"] = True
    except Exception as e:
        info["error"] = str(e)
        return info

    after, after_err = _docker_container_state(cid)
    info["state_after"] = after
    if after_err:
        info["state_after_error"] = after_err

    wait_states = ["running"] if desired == "start" else ["exited", "dead", "created"]
    waited = _wait_for_docker_container_state(container_id=cid, desired_states=wait_states)
    info["wait"] = waited
    if waited.get("state"):
        info["state_after"] = waited.get("state")
    if not bool(waited.get("ok")):
        info["error"] = str(waited.get("error") or "container state transition not confirmed")
        return info

    if desired == "start":
        ready = _wait_for_http_ready(
            base_url=str(getattr(cfg, "qwen_vl_url", "") or ""),
            timeout_sec=float(getattr(cfg, "qwen_vl_ready_timeout_sec", 60.0) or 60.0),
        )
        info["http_ready"] = ready
        if not bool(ready.get("ok")):
            info["error"] = str(ready.get("error") or "qwen vl http server not ready")
            return info

    if progress_cb is not None:
        try:
            msg = "Qwen-VL container hazır" if desired == "start" else "Qwen-VL container durdu"
            progress_cb("qwen_vl_container", 1, 1, msg)
        except Exception:
            pass

    return info

def _write_tracks_csv_with_jersey(
    *,
    in_csv_path: str,
    out_csv_path: str,
    jersey_by_track: Dict[int, Dict[str, Any]],
    track_id_remap: Optional[Dict[int, int]] = None,
    player_cls_id: int = 0,
) -> str:

    track_id_remap = track_id_remap or {}
    jersey_by_track = jersey_by_track or {}

    with open(in_csv_path, "r", encoding="utf-8", errors="ignore") as rf:
        reader = csv.DictReader(rf)
        fieldnames = list(reader.fieldnames or [])
        if "jersey_number" not in fieldnames:
            fieldnames.append("jersey_number")

        os.makedirs(str(Path(out_csv_path).resolve().parent), exist_ok=True)
        with open(out_csv_path, "w", encoding="utf-8", newline="") as wf:
            writer = csv.DictWriter(wf, fieldnames=fieldnames)
            writer.writeheader()

            for row in reader:
                try:
                    cls_id = int(float(row.get("cls_id", -1)))
                except Exception:
                    cls_id = -1

                jersey_val = ""
                if cls_id == int(player_cls_id):
                    try:
                        tid_raw = int(float(row.get("track_id", 0)))
                    except Exception:
                        tid_raw = 0

                    if tid_raw > 0:
                        tid = _resolve_track_id_remap(int(tid_raw), track_id_remap)
                        info = jersey_by_track.get(int(tid))
                        if info is not None:
                            jersey_val = str(info.get("jersey_number", "-1")).strip()
                        else:
                            jersey_val = "-1"

                row["jersey_number"] = jersey_val
                writer.writerow(row)

    return out_csv_path

def run_tracking_reid_osnet(
    *,
    video_path: str,
    out_dir: str,
    device: Optional[str],
    config_path: Optional[str],
    detector_weights: Optional[str],
    reid_weights: Optional[str],
    cfg: Optional[FullPipelineConfig] = None,
    progress_cb: Optional[Callable[[str, int, int, str], None]] = None,
    detection_cache_path: Optional[str] = None,
) -> Dict[str, Any]:
    repo = _repo_root()
    script = repo / "model-training" / "tracking-reid-osnet" / "run_botsort_team_reid.py"
    if not script.exists():
        raise FileNotFoundError(str(script))

    run_id = datetime.utcnow().strftime("%Y%m%d%H%M%S") + f"_{int(time.time()*1000)%100000:05d}"

    config_path = config_path or _default_tracking_config()
    if not config_path:
        raise FileNotFoundError("tracking config.yaml not found")

    try:
        import yaml  # type: ignore

        cfg_obj = None
        with open(config_path, "r", encoding="utf-8") as f:
            cfg_obj = yaml.safe_load(f) or {}

        osnet = cfg_obj.get("osnet") if isinstance(cfg_obj, dict) else None
        if isinstance(osnet, dict) and bool(osnet.get("enabled", False)):
            w_rel = str(osnet.get("weights", "") or "").strip()
            weights_path = Path(w_rel)
            if w_rel and not weights_path.is_absolute():
                weights_path = (Path(config_path).resolve().parent / weights_path).resolve()
            if (not w_rel) or (not weights_path.exists()):
                cfg_obj.setdefault("osnet", {})
                cfg_obj["osnet"]["enabled"] = False
                tmp_cfg = Path(out_dir) / f"tracking_config_{run_id}_no_osnet.yaml"
                with open(tmp_cfg, "w", encoding="utf-8") as wf:
                    yaml.safe_dump(cfg_obj, wf, sort_keys=False, allow_unicode=True)
                config_path = str(tmp_cfg)
    except Exception:
        pass

    save_video = str(Path(out_dir) / f"tracking_{run_id}.mp4")
    save_txt = str(Path(out_dir) / f"tracking_{run_id}.csv")
    log_path = str(Path(out_dir) / f"tracking_{run_id}.log")

    cmd = [
        sys.executable,
        "-u",
        str(script),
        "--config",
        str(Path(config_path).resolve()),
        "--video",
        str(Path(video_path).resolve()),
        "--save_video",
        save_video,
        "--save_txt",
        save_txt,
    ]

    jersey_json_path = str(Path(out_dir) / f"tracking_{run_id}.jersey.json")
    qwen_url = ""
    qwen_model = "Qwen3VL-8B-Instruct-Q4_K_M.gguf"
    if cfg is not None:
        try:
            qwen_url = str(getattr(cfg, "qwen_vl_url", "") or "").strip()
        except Exception:
            qwen_url = ""
        try:
            qwen_model = _resolve_qwen_vl_model_name(
                base_url=qwen_url,
                preferred_model=str(getattr(cfg, "qwen_vl_model", "") or "Qwen3VL-8B-Instruct-Q4_K_M.gguf"),
            )
        except Exception:
            qwen_model = str(getattr(cfg, "qwen_vl_model", "") or "Qwen3VL-8B-Instruct-Q4_K_M.gguf")

    jersey_cmd_enabled = False
    jersey_args: List[str] = []
    if (
        cfg is not None
        and bool(getattr(cfg, "run_jersey_number_recognition", False))
        and bool(getattr(cfg, "jersey_in_tracking", True))
        and bool(qwen_url)
    ):
        jersey_cmd_enabled = True
        jersey_args = [
            "--jersey_enable",
            "1",
            "--jersey_qwen_url",
            qwen_url,
            "--jersey_model",
            str(qwen_model or "Qwen3VL-8B-Instruct-Q4_K_M.gguf"),
            "--jersey_prompt",
            _build_jersey_prompt_with_roster(
                str(getattr(cfg, "jersey_prompt", "") or ""),
                getattr(cfg, "player_roster_json", None) or None,
            )[0],
            "--jersey_min_frame_gap",
            str(int(getattr(cfg, "jersey_min_frame_gap", 20))),
            "--jersey_max_samples_per_track",
            str(int(getattr(cfg, "jersey_max_samples_per_track", 3))),
            "--jersey_min_det_conf",
            str(float(getattr(cfg, "jersey_min_det_conf", 0.55))),
            "--jersey_min_box_area",
            str(int(getattr(cfg, "jersey_min_box_area", 1600))),
            "--jersey_frame_topk",
            str(int(getattr(cfg, "jersey_frame_topk", 5))),
            "--jersey_vis_filter",
            "1" if bool(getattr(cfg, "jersey_vis_filter", False)) else "0",
            "--jersey_vis_min_score",
            str(float(getattr(cfg, "jersey_vis_min_score", 0.12))),
            "--jersey_out_json",
            jersey_json_path,
        ]

        cmd += jersey_args

        crops_dir = getattr(cfg, "jersey_crops_dir", None)
        if crops_dir is not None and str(crops_dir).strip():
            jersey_args += ["--jersey_crops_dir", str(crops_dir)]
            cmd += ["--jersey_crops_dir", str(crops_dir)]

    if device:
        cmd += ["--device", device]
    if detector_weights:
        cmd += ["--detector_weights", detector_weights]
    if reid_weights:
        cmd += ["--reid_weights", reid_weights]
    if detection_cache_path and str(detection_cache_path).strip():
        cmd += ["--detection_cache", str(Path(detection_cache_path).resolve())]

    def _run_tracking_subprocess(cmd_to_run: List[str], log_file: str) -> subprocess.Popen:
        os.makedirs(str(Path(log_file).resolve().parent), exist_ok=True)
        lf = open(log_file, "w", encoding="utf-8", errors="ignore")
        env = os.environ.copy()
        env.setdefault("PYTHONUNBUFFERED", "1")
        env.setdefault("PYTHONFAULTHANDLER", "1")
        env.setdefault("TORCH_SHOW_CPP_STACKTRACES", "1")
        p = subprocess.Popen(
            cmd_to_run,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
            bufsize=1,
            env=env,
        )
        p._pipeline_log_handle = lf  # type: ignore[attr-defined]
        return p

    def _strip_jersey_flags(argv: List[str]) -> List[str]:
        jersey_flags_1 = {
            "--jersey_enable",
            "--jersey_qwen_url",
            "--jersey_model",
            "--jersey_prompt",
            "--jersey_min_frame_gap",
            "--jersey_max_samples_per_track",
            "--jersey_min_det_conf",
            "--jersey_min_box_area",
            "--jersey_frame_topk",
            "--jersey_vis_filter",
            "--jersey_vis_min_score",
            "--jersey_out_json",
            "--jersey_crops_dir",
        }

        out: List[str] = []
        i = 0
        n = len(argv)
        while i < n:
            tok = argv[i]
            if tok in jersey_flags_1:
                i += 2  # drop flag + its value
                continue
            out.append(tok)
            i += 1
        return out

    start_ts = time.time()
    p = _run_tracking_subprocess(cmd, log_path)

    progress_re = re.compile(
        r"\[(?P<cur>\d+)\/(?P<total>\d+)\]\s+(?P<pct>[0-9.]+)%\s+\|\s+(?P<fps>[0-9.]+)\s+FPS\s+\|\s+ETA\s+(?P<eta>\d\d:\d\d:\d\d)"
    )

    last_frame = 0
    inferred_total: Optional[int] = None
    pbar: Optional[tqdm] = None
    jersey_from_stdout: Optional[List[Dict[str, Any]]] = None

    try:
        lf = getattr(p, "_pipeline_log_handle", None)
        assert p.stdout is not None
        for raw_line in p.stdout:
            line = raw_line.rstrip("\n")
            if lf is not None:
                try:
                    lf.write(line + "\n")
                    lf.flush()
                except Exception:
                    pass

            if line.startswith("__JERSEY_RESULT__"):
                try:
                    jersey_from_stdout = json.loads(line[len("__JERSEY_RESULT__"):].strip())
                except Exception:
                    pass
                continue

            m = progress_re.search(line)
            if not m:
                continue

            try:
                cur = int(m.group("cur"))
                total = int(m.group("total"))
                fps_val = float(m.group("fps"))
                eta = str(m.group("eta"))
            except Exception:
                continue

            if inferred_total is None and total > 0:
                inferred_total = total
                try:
                    pbar = tqdm(total=inferred_total, desc="Tracking", unit="frame")
                except Exception:
                    pbar = None

            if cur >= last_frame:
                delta = cur - last_frame
                last_frame = cur
                if pbar is not None and delta > 0:
                    try:
                        pbar.update(delta)
                    except Exception:
                        pass

            if progress_cb is not None and total > 0:
                try:
                    pct = int((cur * 100) / max(1, total))
                    progress_cb(
                        "tracking",
                        cur,
                        total,
                        f"Tracking {pct}% | {fps_val:.2f} FPS | ETA {eta}",
                    )
                except Exception:
                    pass

        p.wait()
    finally:
        try:
            if p.stdout is not None:
                p.stdout.close()
        except Exception:
            pass
        try:
            if pbar is not None:
                pbar.close()
        except Exception:
            pass

        try:
            lh = getattr(p, "_pipeline_log_handle", None)
            if lh is not None:
                lh.close()
        except Exception:
            pass

    tracking_retry_log_path: Optional[str] = None
    if p.returncode != 0 and jersey_cmd_enabled:
        try:
            tracking_retry_log_path = str(Path(out_dir) / f"tracking_{run_id}_retry_no_jersey.log")

            for pp in (save_video, save_txt):
                try:
                    if os.path.isfile(pp):
                        os.remove(pp)
                except Exception:
                    pass

            cmd_no_jersey = _strip_jersey_flags(list(cmd))

            p = _run_tracking_subprocess(cmd_no_jersey, tracking_retry_log_path)

            retry_last_frame = 0
            retry_total: Optional[int] = None
            try:
                lf2 = getattr(p, "_pipeline_log_handle", None)
                assert p.stdout is not None
                for raw_line2 in p.stdout:
                    line2 = raw_line2.rstrip("\n")
                    if lf2 is not None:
                        try:
                            lf2.write(line2 + "\n")
                            lf2.flush()
                        except Exception:
                            pass

                    m2 = progress_re.search(line2)
                    if not m2:
                        continue
                    try:
                        cur2 = int(m2.group("cur"))
                        total2 = int(m2.group("total"))
                        fps2 = float(m2.group("fps"))
                        eta2 = str(m2.group("eta"))
                    except Exception:
                        continue

                    if retry_total is None and total2 > 0:
                        retry_total = total2
                    if cur2 >= retry_last_frame:
                        retry_last_frame = cur2

                    if progress_cb is not None and total2 > 0:
                        try:
                            pct2 = int((cur2 * 100) / max(1, total2))
                            progress_cb(
                                "tracking",
                                cur2,
                                total2,
                                f"Tracking (retry) {pct2}% | {fps2:.2f} FPS | ETA {eta2}",
                            )
                        except Exception:
                            pass

                p.wait()
            finally:
                try:
                    if p.stdout is not None:
                        p.stdout.close()
                except Exception:
                    pass
                try:
                    lh2 = getattr(p, "_pipeline_log_handle", None)
                    if lh2 is not None:
                        lh2.close()
                except Exception:
                    pass
            log_path = tracking_retry_log_path
        except Exception:
            pass

    if p.returncode != 0:
        tail = ""
        try:
            with open(log_path, "r", encoding="utf-8", errors="ignore") as rf:
                tail = rf.read()[-4000:]
        except Exception:
            tail = ""
        rc = p.returncode
        rc_hex = None
        try:
            if isinstance(rc, int) and rc < 0:
                rc_hex = hex((1 << 32) + rc)
            elif isinstance(rc, int):
                rc_hex = hex(rc)
        except Exception:
            rc_hex = None
        raise RuntimeError(
            "tracking-reid-osnet failed\n"
            f"cmd: {' '.join(cmd)}\n"
            f"returncode: {rc} ({rc_hex})\n"
            f"log: {log_path}\n"
            f"tail: {tail}"
        )

    if not os.path.isfile(save_video):
        raise RuntimeError(f"Tracking video not produced: {save_video}\nlog: {log_path}")
    if not os.path.isfile(save_txt):
        raise RuntimeError(f"Tracking CSV not produced: {save_txt}\nlog: {log_path}")

    jersey_by_track: Dict[int, Dict[str, Any]] = {}
    track_id_remap: Dict[int, int] = {}
    out_csv_path = save_txt

    jersey_in_tracking_cfg = bool(getattr(cfg, "jersey_in_tracking", True)) if cfg is not None else False
    if cfg is not None and bool(getattr(cfg, "run_jersey_number_recognition", False)) and jersey_in_tracking_cfg:
        try:
            items: List[Dict[str, Any]] = []
            if jersey_from_stdout is not None:
                items = jersey_from_stdout
            elif os.path.isfile(jersey_json_path):
                with open(jersey_json_path, "r", encoding="utf-8") as f:
                    items = json.load(f) or []
            for it in items:
                try:
                    tid = int(it.get("track_id"))
                    jersey_by_track[int(tid)] = dict(it)
                except Exception:
                    continue
        except Exception:
            jersey_by_track = {}

        if not jersey_by_track:
            try:
                jersey_by_track = infer_jersey_numbers_from_tracking(
                    video_path=video_path,
                    tracks_csv_path=save_txt,
                    out_dir=out_dir,
                    cfg=cfg,
                    progress_cb=progress_cb,
                )
            except Exception:
                jersey_by_track = {}

        try:
            track_id_remap = _build_track_id_remap_from_jerseys(jersey_by_track=jersey_by_track, cfg=cfg)
        except Exception:
            track_id_remap = {}

        if track_id_remap and jersey_by_track:
            canonical: Dict[int, Dict[str, Any]] = {}
            for tid, info in jersey_by_track.items():
                canon = _resolve_track_id_remap(int(tid), track_id_remap)
                cur = canonical.get(int(canon))
                if cur is None:
                    canonical[int(canon)] = {**info, "track_id": int(canon)}
                    continue
                try:
                    if float(info.get("confidence", 0.0)) > float(cur.get("confidence", 0.0)):
                        canonical[int(canon)] = {**info, "track_id": int(canon)}
                except Exception:
                    pass
            jersey_by_track = canonical

        try:
            enriched = str(Path(out_dir) / f"tracking_{run_id}_with_jersey.csv")
            out_csv_path = _write_tracks_csv_with_jersey(
                in_csv_path=save_txt,
                out_csv_path=enriched,
                jersey_by_track=jersey_by_track,
                track_id_remap=track_id_remap,
                player_cls_id=int(getattr(cfg, "player_cls_id", 0)),
            )
        except Exception:
            out_csv_path = save_txt

    elif cfg is not None and bool(getattr(cfg, "run_jersey_number_recognition", False)):
        try:
            jersey_by_track = infer_jersey_numbers_from_tracking(
                video_path=video_path,
                tracks_csv_path=save_txt,
                out_dir=out_dir,
                cfg=cfg,
                progress_cb=progress_cb,
            )
        except Exception:
            jersey_by_track = {}

        try:
            track_id_remap = _build_track_id_remap_from_jerseys(jersey_by_track=jersey_by_track, cfg=cfg)
        except Exception:
            track_id_remap = {}

        try:
            enriched = str(Path(out_dir) / f"tracking_{run_id}_with_jersey.csv")
            out_csv_path = _write_tracks_csv_with_jersey(
                in_csv_path=save_txt,
                out_csv_path=enriched,
                jersey_by_track=jersey_by_track,
                track_id_remap=track_id_remap,
                player_cls_id=int(getattr(cfg, "player_cls_id", 0)),
            )
        except Exception:
            out_csv_path = save_txt

    return {
        "tracking_video_path": save_video,
        "tracks_csv_path": out_csv_path,
        "tracks_csv_raw_path": save_txt,
        "jersey_json_path": jersey_json_path if os.path.isfile(jersey_json_path) else None,
        "jersey_by_track": jersey_by_track,
        "track_id_remap": track_id_remap,
        "tracking_log_path": log_path,
        **({"tracking_retry_log_path": tracking_retry_log_path} if tracking_retry_log_path else {}),
        "tracking_seconds": float(time.time() - start_ts),
    }

def _normalize_tdeed_label(label: str) -> str:
    mapping: Dict[str, str] = {
        "PASS": "Pass",
        "DRIVE": "Drive",
        "HEADER": "Header",
        "HIGH PASS": "High pass",
        "FREE KICK": "Free-kick",
        "SHOT": "Shot",
        "BALL PLAYER BLOCK": "Ball player block",
        "BALL OUT OF PLAY": "Ball out of play",
    }
    return mapping.get(str(label).strip(), str(label).strip())

def _resolve_tdeed_checkpoint_path(
    *,
    tdeed_dir: Path,
    model_name: str,
    explicit_checkpoint_path: Optional[str],
) -> str:
    if explicit_checkpoint_path and os.path.isfile(explicit_checkpoint_path):
        return str(Path(explicit_checkpoint_path).resolve())

    dataset_prefix = model_name.split("_")[0]
    candidates = [
        tdeed_dir / "checkpoints" / dataset_prefix / model_name / "checkpoint_best.pt",
        tdeed_dir / "checkpoints" / "SoccerNet" / model_name / "checkpoint_best.pt",
    ]
    for c in candidates:
        if c.exists():
            return str(c.resolve())

    raise FileNotFoundError(
        f"T-DEED checkpoint not found for model '{model_name}'. Searched: "
        + ", ".join(str(c) for c in candidates)
    )

def run_action_spotting_tdeed(
    *,
    video_path: str,
    out_dir: str,
    run_id: str,
    model_name: str,
    threshold: float,
    nms_window_sec: float,
    frame_width: int,
    frame_height: int,
    checkpoint_path: Optional[str] = None,
    tdeed_repo_dir: Optional[str] = None,
    variant: str = "primary",
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    tdeed_root = tdeed_repo_dir or _default_tdeed_repo_dir()
    if not tdeed_root:
        raise FileNotFoundError("T-DEED repository not found")

    tdeed_dir = Path(str(tdeed_root)).resolve()
    if not tdeed_dir.exists():
        raise FileNotFoundError(str(tdeed_dir))

    script_path = tdeed_dir / "inference.py"
    if not script_path.exists():
        raise FileNotFoundError(str(script_path))

    model_name = str(model_name or "").strip()
    if not model_name:
        raise ValueError("T-DEED model_name is empty")

    config_path = tdeed_dir / "config" / model_name.split("_")[0] / f"{model_name}.json"
    if not config_path.exists():
        raise FileNotFoundError(str(config_path))

    ckpt_path = _resolve_tdeed_checkpoint_path(
        tdeed_dir=tdeed_dir,
        model_name=model_name,
        explicit_checkpoint_path=checkpoint_path,
    )

    os.makedirs(out_dir, exist_ok=True)
    out_json_name = f"tdeed_{_normalize_model_name_token(model_name)}_{run_id}.json"
    out_json_path = str((Path(out_dir) / out_json_name).resolve())

    def _invoke_tdeed(width: int, height: int) -> Tuple[subprocess.CompletedProcess, float, List[str]]:
        cmd: List[str] = [
            sys.executable,
            "-u",
            str(script_path),
            "--model",
            model_name,
            "--video_path",
            str(Path(video_path).resolve()),
            "--frame_width",
            str(int(width)),
            "--frame_height",
            str(int(height)),
            "--inference_threshold",
            str(float(threshold)),
            "--checkpoint_path",
            str(ckpt_path),
            "--output_dir",
            str(Path(out_dir).resolve()),
            "--output_filename",
            out_json_name,
        ]
        started = time.perf_counter()
        env = os.environ.copy()
        env.setdefault("PYTHONUNBUFFERED", "1")
        # Allow CUDA to schedule alongside calibration process (WDDM time-slicing).
        # pytorch_cuda_alloc_conf limits fragmentation when two contexts share VRAM.
        env.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
        proc = subprocess.run(
            cmd,
            cwd=str(tdeed_dir),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
            env=env,
        )
        elapsed = time.perf_counter() - started
        return proc, elapsed, cmd

    requested_w = max(64, int(frame_width))
    requested_h = max(64, int(frame_height))
    used_w, used_h = requested_w, requested_h
    used_fallback_resolution = False

    proc, elapsed_sec, cmd = _invoke_tdeed(requested_w, requested_h)
    stdout_text = str(proc.stdout or "")
    if proc.returncode != 0:
        lowered = stdout_text.lower()
        if "outofmemory" in lowered or "cuda out of memory" in lowered:
            fallback_w = min(requested_w, 398)
            fallback_h = min(requested_h, 224)
            if fallback_w != requested_w or fallback_h != requested_h:
                used_w, used_h = fallback_w, fallback_h
                used_fallback_resolution = True
                proc, elapsed_sec, cmd = _invoke_tdeed(used_w, used_h)
                stdout_text = str(proc.stdout or "")

    if proc.returncode != 0:
        tail = "\n".join(stdout_text.splitlines()[-40:])
        raise RuntimeError(f"T-DEED inference failed for {model_name}:\n{tail}")

    if not os.path.isfile(out_json_path):
        raise FileNotFoundError(out_json_path)

    with open(out_json_path, "r", encoding="utf-8") as f:
        pred_payload = json.load(f)

    predictions = pred_payload.get("predictions") if isinstance(pred_payload, dict) else []
    predictions = predictions if isinstance(predictions, list) else []

    cap = cv2.VideoCapture(str(Path(video_path).resolve()))
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 25.0)
    cap.release()
    if fps <= 1e-6:
        fps = 25.0

    dataset_name = model_name.split("_")[0]
    out_events: List[Dict[str, Any]] = []
    labels_found: List[str] = []
    for p in predictions:
        try:
            frame = int(float(p.get("frame", 0)))
            raw_label = str(p.get("label", "")).strip()
            score = float(p.get("confidence", 0.0))
        except Exception:
            continue
        if not raw_label:
            continue
        norm_label = _normalize_tdeed_label(raw_label)
        t = max(0.0, float(frame) / float(fps))
        out_events.append(
            {
                "type": "soccer_event",
                "source": "action_spotting",
                "label": norm_label,
                **({"raw_label": raw_label} if raw_label != norm_label else {}),
                "t": t,
                "timecode": _timecode(t),
                "confidence": score,
                "description_tr": _event_desc_tr(norm_label),
                "spotting_model": model_name,
                "spotting_variant": str(variant),
                "spotting_dataset": dataset_name,
            }
        )
        labels_found.append(norm_label)

    meta: Dict[str, Any] = {
        "model_name": model_name,
        "variant": variant,
        "dataset": dataset_name,
        "config_path": str(config_path),
        "checkpoint_path": str(ckpt_path),
        "threshold": float(threshold),
        "nms_window_sec": float(nms_window_sec),
        "frame_width": used_w,
        "frame_height": used_h,
        "used_fallback_resolution": used_fallback_resolution,
        "output_json_path": out_json_path,
        "prediction_count": len(predictions),
        "labels": sorted(set(labels_found)),
        "inference_seconds": round(elapsed_sec, 3),
        "command": cmd,
    }

    return out_events, meta

def run_action_spotting_spotting_v2(
    *,
    features_path: str,
    checkpoint_path: Optional[str],
    threshold: float,
    nms_window_sec: float,
) -> List[Dict[str, Any]]:
    repo = _repo_root()
    module_dir = repo / "model-training" / "action_spotting" / "spotting_v2"
    if not module_dir.exists():
        raise FileNotFoundError(str(module_dir))

    features_path = str(Path(features_path).resolve())
    if not os.path.isfile(features_path):
        raise FileNotFoundError(features_path)

    checkpoint_path = checkpoint_path or _default_action_checkpoint()
    if not checkpoint_path:
        raise FileNotFoundError("Action spotting checkpoint not found")
    checkpoint_path = str(Path(checkpoint_path).resolve())

    if str(module_dir) not in sys.path:
        sys.path.insert(0, str(module_dir))

    try:
        import inference as spotting_inference  # type: ignore
    except Exception as e:
        raise RuntimeError(f"Failed to import spotting_v2 inference.py: {e}")

    model = spotting_inference.load_model(checkpoint_path)
    preds = spotting_inference.predict(model, features_path, threshold=threshold)
    final_preds = spotting_inference.nms(preds, window_sec=nms_window_sec)

    out: List[Dict[str, Any]] = []
    for p in final_preds:
        t = float(p.get("time", 0.0))
        label = str(p.get("label", ""))
        score = float(p.get("score", 0.0))
        out.append(
            {
                "type": "soccer_event",
                "source": "action_spotting",
                "label": label,
                "t": t,
                "timecode": _timecode(t),
                "confidence": score,
                "description_tr": _event_desc_tr(label),
            }
        )
    return out

def _parse_tracks_csv(csv_path: str) -> Dict[int, List[Dict[str, Any]]]:
    by_frame: Dict[int, List[Dict[str, Any]]] = {}
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                frame_id = int(float(row["frame_id"]))
            except Exception:
                continue
            by_frame.setdefault(frame_id, []).append(row)
    return by_frame

def _xyxy_center(row: Dict[str, Any]) -> Tuple[float, float]:
    x1 = float(row["x1"])
    y1 = float(row["y1"])
    x2 = float(row["x2"])
    y2 = float(row["y2"])
    return ((x1 + x2) / 2.0, (y1 + y2) / 2.0)

def derive_possession_events_from_tracks(
    *,
    tracks_csv_path: str,
    fps: float,
    width: int,
    height: int,
    cfg: FullPipelineConfig,
) -> List[Dict[str, Any]]:
    by_frame = _parse_tracks_csv(tracks_csv_path)
    diag = float(np.sqrt(width * width + height * height)) + 1e-8

    frame_ids = sorted(by_frame.keys())
    owners: List[Tuple[float, Optional[int]]] = []

    stride = max(1, int(cfg.possession_stride_frames))
    ids = frame_ids[::stride]
    for fid in tqdm(ids, total=len(ids) if ids else None, desc="Possession", unit="frame"):
        rows = by_frame.get(fid, [])

        ball_rows = [r for r in rows if int(float(r.get("cls_id", -1))) == int(cfg.ball_cls_id)]
        if not ball_rows:
            owners.append((fid / fps, None))
            continue

        ball_row = max(ball_rows, key=lambda r: float(r.get("conf", 0.0)))
        bx, by = _xyxy_center(ball_row)

        player_rows = [r for r in rows if int(float(r.get("cls_id", -1))) == int(cfg.player_cls_id)]
        best_id: Optional[int] = None
        best_dist = 1e9
        for pr in player_rows:
            px, py = _xyxy_center(pr)
            d = float(np.sqrt((bx - px) ** 2 + (by - py) ** 2)) / diag
            if d < best_dist:
                best_dist = d
                try:
                    best_id = int(float(pr.get("track_id", 0)))
                except Exception:
                    best_id = None

        if best_id is not None and best_dist <= float(cfg.possession_dist_norm):
            owners.append((fid / fps, best_id))
        else:
            owners.append((fid / fps, None))

    stable_n = max(1, int(cfg.possession_stable_frames))
    window: List[Optional[int]] = []
    cur_stable: Optional[int] = None
    prev_emitted: Optional[int] = None
    events: List[Dict[str, Any]] = []

    def emit(t: float, new_owner: Optional[int], prev_owner: Optional[int]):
        if new_owner == prev_owner:
            return
        if new_owner is None:
            etype = "ball_uncontrolled"
            desc = "Top kontrolsüz kaldı."
        elif prev_owner is None:
            etype = "possession_start"
            desc = f"Top kontrolü #{new_owner} oyuncusuna geçti."
        else:
            etype = "possession_change"
            desc = f"Top kontrolü #{prev_owner} -> #{new_owner} değişti."

        events.append(
            {
                "type": etype,
                "source": "tracking",
                "t": float(t),
                "timecode": _timecode(float(t)),
                "player_track_id": int(new_owner) if new_owner is not None else None,
                "from_player_track_id": int(prev_owner) if prev_owner is not None else None,
                "confidence": 0.55,
                "description_tr": desc,
            }
        )

    for t, owner in owners:
        window.append(owner)
        if len(window) > stable_n:
            window.pop(0)
        if len(window) < stable_n:
            continue
        if all(o == window[0] for o in window):
            stable_owner = window[0]
            if stable_owner != cur_stable:
                cur_stable = stable_owner
                if cur_stable != prev_emitted:
                    emit(t, cur_stable, prev_emitted)
                    prev_emitted = cur_stable

    return events

def overlay_events_on_video(
    *,
    base_video_path: str,
    out_path: str,
    events: List[Dict[str, Any]],
    window_sec: float,
    tracks_csv_path: Optional[str] = None,
    jersey_by_track: Optional[Dict[int, Dict[str, Any]]] = None,
    track_id_remap: Optional[Dict[int, int]] = None,
    progress_cb: Optional[Callable[[str, int, int, str], None]] = None,
) -> str:
    out_path = str(Path(out_path).resolve())
    tmp_path = str(Path(out_path).with_suffix(".tmp.mp4"))

    cap = cv2.VideoCapture(base_video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open base video: {base_video_path}")

    fps = float(cap.get(cv2.CAP_PROP_FPS) or 25.0)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 640)
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 480)

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(tmp_path, fourcc, fps, (w, h))
    if not writer.isOpened():
        cap.release()
        raise RuntimeError(f"Failed to open writer: {tmp_path}")

    intervals: List[Tuple[float, float, str]] = []
    for e in events:
        try:
            t = float(e.get("t", 0.0))
        except Exception:
            continue
        label = str(e.get("label") or e.get("type") or "event")
        score = e.get("confidence")
        text = f"{_timecode(t)}  {label}" + (f"  ({float(score):.2f})" if isinstance(score, (float, int)) else "")
        intervals.append((t, t + float(window_sec), text))

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    frame_idx = 0
    pbar = tqdm(total=total_frames if total_frames > 0 else None, desc="Overlay", unit="frame")
    csv_f = None
    csv_reader = None
    next_row: Optional[Dict[str, Any]] = None
    if tracks_csv_path and os.path.isfile(tracks_csv_path):
        try:
            csv_f = open(tracks_csv_path, "r", encoding="utf-8", errors="ignore")
            csv_reader = csv.DictReader(csv_f)
            next_row = next(csv_reader, None)
        except Exception:
            csv_reader = None
            next_row = None

    jersey_by_track = jersey_by_track or {}
    track_id_remap = track_id_remap or {}

    while True:
        ret, frame = cap.read()
        if not ret or frame is None:
            break
        t = frame_idx / fps

        active = [txt for (t0, t1, txt) in intervals if t0 <= t <= t1]
        draw = frame
        y = 30
        for txt in active[:3]:
            cv2.putText(draw, txt, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 3)
            cv2.putText(draw, txt, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 1)
            y += 30

        if csv_reader is not None:
            rows: List[Dict[str, Any]] = []
            try:
                while next_row is not None:
                    fid = int(float(next_row.get("frame_id", -1)))
                    if fid < frame_idx:
                        next_row = next(csv_reader, None)
                        continue
                    if fid > frame_idx:
                        break
                    rows.append(next_row)
                    next_row = next(csv_reader, None)
            except Exception:
                rows = []

            for r in rows:
                try:
                    cls_id = int(float(r.get("cls_id", -1)))
                    if cls_id != 0:
                        continue
                    track_id_raw = int(float(r.get("track_id", 0)))
                    if track_id_raw <= 0:
                        continue
                    track_id = _resolve_track_id_remap(int(track_id_raw), track_id_remap)
                    info = jersey_by_track.get(int(track_id))
                    jersey_val = (info or {}).get("jersey_number")
                    jersey = str(jersey_val).strip() if jersey_val is not None else "-1"

                    x1 = int(float(r.get("x1", 0)))
                    y1 = int(float(r.get("y1", 0)))
                    x2 = int(float(r.get("x2", 0)))
                    y2 = int(float(r.get("y2", 0)))
                except Exception:
                    continue

                x1 = _clamp_int(x1, 0, draw.shape[1] - 1)
                x2 = _clamp_int(x2, 0, draw.shape[1] - 1)
                y1 = _clamp_int(y1, 0, draw.shape[0] - 1)
                y2 = _clamp_int(y2, 0, draw.shape[0] - 1)
                if x2 <= x1 or y2 <= y1:
                    continue

                cv2.rectangle(draw, (x1, y1), (x2, y2), (0, 255, 0), 2)

                label = f"#{track_id}  {jersey}"
                y_text = max(14, y1 - 6)
                cv2.putText(draw, label, (x1, y_text), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 3)
                cv2.putText(draw, label, (x1, y_text), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

        writer.write(draw)
        frame_idx += 1
        try:
            pbar.update(1)
        except Exception:
            pass
        if progress_cb is not None and total_frames > 0:
            try:
                progress_cb("overlay", frame_idx, total_frames, "Overlay yazılıyor")
            except Exception:
                pass

    cap.release()
    writer.release()
    try:
        if csv_f is not None:
            csv_f.close()
    except Exception:
        pass
    try:
        pbar.close()
    except Exception:
        pass

    ffmpeg_bin = _ffmpeg_exe()
    if ffmpeg_bin and os.path.isfile(tmp_path):
        try:
            cmd = [
                ffmpeg_bin,
                "-y",
                "-i",
                tmp_path,
                "-c:v",
                "libx264",
                "-pix_fmt",
                "yuv420p",
                "-preset",
                "fast",
                "-crf",
                "20",
                "-an",
                "-movflags",
                "+faststart",
                out_path,
            ]
            subprocess.run(cmd, capture_output=True, text=True, check=True)
            try:
                os.remove(tmp_path)
            except Exception:
                pass
            return out_path
        except Exception:
            pass

    try:
        if os.path.isfile(out_path):
            os.remove(out_path)
    except Exception:
        pass
    try:
        os.replace(tmp_path, out_path)
    except Exception:
        out_path = tmp_path
    return out_path

def run_full_pipeline(
    *,
    video_path: str,
    out_dir: str,
    cfg: FullPipelineConfig,
    progress_cb: Optional[Callable[[str, int, int, str], None]] = None,
) -> Dict[str, Any]:
    out_dir = str(Path(out_dir).resolve())
    os.makedirs(out_dir, exist_ok=True)

    run_id = datetime.utcnow().strftime("%Y%m%d%H%M%S") + f"_{int(time.time()*1000)%100000:05d}"
    stage_timings_sec: Dict[str, float] = {}
    runtime_config: Dict[str, Any] = {
        "start_seconds": float(cfg.start_seconds or 0.0),
        "duration_seconds": float(cfg.duration_seconds) if cfg.duration_seconds is not None else None,
        "run_calibration": bool(cfg.run_calibration),
        "run_tracking": bool(cfg.run_tracking),
        "run_action_spotting": bool(cfg.run_action_spotting),
        "run_jersey_number_recognition": bool(cfg.run_jersey_number_recognition),
        "run_commentary": bool(getattr(cfg, "run_commentary", True)),
        "calibration": {
            "conf_thres": float(cfg.calibration_conf_thres),
            "write_frames_jsonl": bool(cfg.calibration_write_frames_jsonl or bool(getattr(cfg, "run_commentary", True))),
            "frames_stride": int(cfg.calibration_frames_stride),
            "yolo_frame_window": int(cfg.calibration_yolo_frame_window),
            "yolo_selection_mode": str(cfg.calibration_yolo_selection_mode),
            "interpolation_mode": str(cfg.calibration_interpolation_mode),
        },
        "commentary": {
            "llm_backend": str(getattr(cfg, "commentary_llm_backend", "ollama") or "ollama"),
            "llm_url": str(getattr(cfg, "commentary_llm_url", "http://localhost:11434/") or "http://localhost:11434/"),
            "llm_model": str(getattr(cfg, "commentary_llm_model", "qwen3.5:9b") or "qwen3.5:9b"),
            "flush_gpu_before_llm": bool(getattr(cfg, "commentary_flush_gpu_before_llm", True)),
            "context_window_sec": float(getattr(cfg, "commentary_context_window_sec", 12.0) or 12.0),
            "context_stride_sec": float(getattr(cfg, "commentary_context_stride_sec", 1.0) or 1.0),
            "context_max_samples": int(getattr(cfg, "commentary_context_max_samples", 9) or 9),
        },
        "qwen_vl": {
            "manage_container": bool(getattr(cfg, "qwen_vl_manage_container", False)),
            "container_id": str(getattr(cfg, "qwen_vl_container_id", "") or ""),
            "stop_before_commentary": bool(getattr(cfg, "qwen_vl_stop_before_commentary", True)),
        },
        "action_spotting": {
            "model_name": str(getattr(cfg, "action_model_name", "SoccerNet_big") or "SoccerNet_big"),
            "frame_width": int(getattr(cfg, "action_frame_width", 398) or 398),
            "frame_height": int(getattr(cfg, "action_frame_height", 224) or 224),
            "threshold": float(getattr(cfg, "action_threshold", 0.50)),
            "run_ball": bool(getattr(cfg, "run_ball_action_spotting", True)),
            "ball_model_name": str(getattr(cfg, "ball_model_name", "SoccerNetBall_challenge2") or "SoccerNetBall_challenge2"),
            "ball_threshold": float(getattr(cfg, "ball_action_threshold", 0.20)),
        },
    }

    def emit(stage: str, cur: int, total: int, msg: str) -> None:
        if progress_cb is None:
            return
        try:
            progress_cb(stage, int(cur), int(total), str(msg))
        except Exception:
            pass

    segment_path = str(Path(out_dir) / f"segment_{run_id}.mp4")
    detection_cache_path = str(Path(out_dir) / f"det_cache_{run_id}.json")
    emit("segment", 0, 1, "Video segment hazırlanıyor")
    stage_start = time.perf_counter()
    extract_segment_to_mp4(
        src_path=video_path,
        out_path=segment_path,
        start_sec=float(cfg.start_seconds or 0.0),
        duration_sec=cfg.duration_seconds,
    )
    stage_timings_sec["segment"] = round(time.perf_counter() - stage_start, 3)
    emit("segment", 1, 1, "Video segment hazır")

    calibration_map_video_path: Optional[str] = None
    calibration_events_json_path: Optional[str] = None
    calibration_frames_jsonl_path: Optional[str] = None
    calibration_events: List[Dict[str, Any]] = []
    calibration_frames: List[Dict[str, Any]] = []
    calibration_frame_times: List[float] = []
    calibration_runtime_config: Optional[Dict[str, Any]] = None

    if bool(getattr(cfg, "run_calibration", True)):
        emit("calibration", 0, 1, "Calibration başlıyor")
        stage_start = time.perf_counter()
        try:
            out_map = str(Path(out_dir) / f"map_{run_id}.mp4")
            out_events = str(Path(out_dir) / f"calibration_events_{run_id}.json")
            out_frames = None
            if bool(getattr(cfg, "calibration_write_frames_jsonl", False)) or bool(getattr(cfg, "run_commentary", True)):
                out_frames = str(Path(out_dir) / f"calibration_frames_{run_id}.jsonl")
            calib_res = run_calibration_pipeline(
                video_path=segment_path,
                out_map=out_map,
                out_events=out_events,
                out_frames=out_frames,
                cfg=cfg,
                progress_cb=progress_cb,
                detection_cache_path=detection_cache_path,
            )
            calibration_runtime_config = calib_res.get("config") if isinstance(calib_res, dict) else None

            calibration_map_video_path = str(calib_res.get("map_video_path") or out_map)
            calibration_events_json_path = str(calib_res.get("events_json_path") or out_events)
            try:
                calibration_frames_jsonl_path = str(calib_res.get("frames_jsonl_path") or "")
            except Exception:
                calibration_frames_jsonl_path = None
            if calibration_frames_jsonl_path and not os.path.isfile(str(calibration_frames_jsonl_path)):
                calibration_frames_jsonl_path = None
            if calibration_frames_jsonl_path and bool(getattr(cfg, "run_commentary", True)):
                try:
                    calibration_frames, calibration_frame_times = _load_calibration_frames_jsonl(calibration_frames_jsonl_path)
                except Exception:
                    calibration_frames, calibration_frame_times = [], []

            if calibration_events_json_path and os.path.isfile(calibration_events_json_path):
                try:
                    with open(calibration_events_json_path, "r", encoding="utf-8") as f:
                        ce = json.load(f) or {}
                    evs = ce.get("events") if isinstance(ce, dict) else None
                    if isinstance(evs, list):
                        for e in evs:
                            if isinstance(e, dict):
                                calibration_events.append(e)
                except Exception:
                    pass

            try:
                ffmpeg_bin = _ffmpeg_exe()
                if ffmpeg_bin and calibration_map_video_path and os.path.isfile(calibration_map_video_path):
                    src = str(calibration_map_video_path)
                    tmp = str(Path(out_dir) / f"map_{run_id}.tmp_h264.mp4")
                    cmd = [
                        ffmpeg_bin,
                        "-y",
                        "-i",
                        src,
                        "-c:v",
                        "libx264",
                        "-pix_fmt",
                        "yuv420p",
                        "-preset",
                        "fast",
                        "-crf",
                        "20",
                        "-an",
                        "-movflags",
                        "+faststart",
                        tmp,
                    ]
                    subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", errors="replace", check=True)
                    try:
                        os.replace(tmp, src)
                    except Exception:
                        calibration_map_video_path = tmp
            except Exception:
                pass
        except Exception:
            calibration_map_video_path = None
            calibration_events_json_path = None
            calibration_frames_jsonl_path = None
            calibration_events = []
        stage_timings_sec["calibration"] = round(time.perf_counter() - stage_start, 3)
        emit("calibration", 1, 1, "Calibration tamam")

    tracking_video_path: Optional[str] = None
    tracks_csv_path: Optional[str] = None
    tracks_csv_raw_path: Optional[str] = None
    tracking_res: Optional[Dict[str, Any]] = None
    qwen_vl_container_events: List[Dict[str, Any]] = []

    if bool(cfg.run_jersey_number_recognition):
        try:
            evt = _set_qwen_vl_container_state(cfg=cfg, action="start", progress_cb=progress_cb)
            qwen_vl_container_events.append(evt)
        except Exception:
            pass

    if cfg.run_tracking:
        emit("tracking", 0, 1, "Tracking başlıyor")
        stage_start = time.perf_counter()
        tracking_res = run_tracking_reid_osnet(
            video_path=segment_path,
            out_dir=out_dir,
            device=cfg.tracking_device,
            config_path=cfg.tracking_config_path,
            detector_weights=cfg.detector_weights,
            reid_weights=cfg.reid_weights,
            cfg=cfg,
            progress_cb=progress_cb,
            detection_cache_path=detection_cache_path,
        )
        tracking_video_path = tracking_res["tracking_video_path"]
        tracks_csv_path = tracking_res["tracks_csv_path"]
        tracks_csv_raw_path = tracking_res.get("tracks_csv_raw_path")
        stage_timings_sec["tracking"] = round(time.perf_counter() - stage_start, 3)
        emit("tracking", 1, 1, "Tracking tamam")

    events: List[Dict[str, Any]] = []
    if calibration_events:
        events.extend(calibration_events)

    jersey_by_track: Dict[int, Dict[str, Any]] = {}
    track_id_remap: Dict[int, int] = {}

    if tracks_csv_path and os.path.isfile(tracks_csv_path):
        cap = cv2.VideoCapture(segment_path)
        fps = float(cap.get(cv2.CAP_PROP_FPS) or 25.0)
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 640)
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 480)
        cap.release()
        try:
            events.extend(
                derive_possession_events_from_tracks(
                    tracks_csv_path=tracks_csv_path,
                    fps=fps,
                    width=w,
                    height=h,
                    cfg=cfg,
                )
            )
        except Exception:
            pass

        if isinstance(tracking_res, dict):
            try:
                jersey_by_track = tracking_res.get("jersey_by_track") or {}
                track_id_remap = tracking_res.get("track_id_remap") or {}
            except Exception:
                jersey_by_track = {}
                track_id_remap = {}

        if bool(cfg.run_jersey_number_recognition) and (not jersey_by_track) and tracks_csv_path and os.path.isfile(tracks_csv_path):
            emit("jersey", 0, 1, "Forma numarası okunuyor")
            stage_start = time.perf_counter()
            try:
                jersey_by_track = infer_jersey_numbers_from_tracking(
                    video_path=segment_path,
                    tracks_csv_path=tracks_csv_path,
                    out_dir=out_dir,
                    cfg=cfg,
                    progress_cb=progress_cb,
                )
            except Exception:
                jersey_by_track = {}
            stage_timings_sec["jersey"] = round(time.perf_counter() - stage_start, 3)
            emit("jersey", 1, 1, "Forma numarası hazır")

            try:
                track_id_remap = _build_track_id_remap_from_jerseys(jersey_by_track=jersey_by_track, cfg=cfg)
            except Exception:
                track_id_remap = {}

    action_spotting_metadata: Dict[str, Any] = {}
    action_spotting_metadata_path: Optional[str] = None

    # ── EventEngine ─────────────────────────────────────────────────────────
    # tracking CSV + calibration JSONL hazırsa EventEngine'i çalıştır.
    # Sonuçlar EventEngineMeta listesine çevrilir ve diske yazılır.
    engine_meta_events: List[EventEngineMeta] = []
    event_engine_output_path: Optional[str] = None

    if tracks_csv_path and os.path.isfile(tracks_csv_path) and calibration_frames_jsonl_path and os.path.isfile(str(calibration_frames_jsonl_path)):
        emit("event_engine", 0, 1, "EventEngine başlıyor")
        stage_start = time.perf_counter()
        try:
            from event_engine import EventEngine as _EventEngine  # type: ignore

            cap_ee = cv2.VideoCapture(segment_path)
            fps_ee = float(cap_ee.get(cv2.CAP_PROP_FPS) or 25.0)
            cap_ee.release()

            event_engine_output_path = str(Path(out_dir) / f"event_engine_{run_id}.json")
            _ee = _EventEngine(
                tracking_csv_path=str(tracks_csv_path),
                calibration_jsonl_path=str(calibration_frames_jsonl_path),
                fps=fps_ee,
                event_log_path=str(Path(out_dir) / f"event_engine_{run_id}.jsonl"),
            )
            raw_engine_events = _ee.run(verbose=False)

            # Event → EventEngineMeta dönüşümü
            for ev in raw_engine_events:
                try:
                    em = EventEngineMeta(
                        frame=int(ev.frame_id),
                        priority=int(ev.priority),
                        event_text=str(ev.message),
                        track_id=ev.track_id if ev.track_id is not None else None,
                    )
                    engine_meta_events.append(em)
                except Exception:
                    pass

            # Diske yaz
            with open(event_engine_output_path, "w", encoding="utf-8") as _ef:
                json.dump(
                    {
                        "schema_version": "1.0",
                        "run_id": run_id,
                        "created_utc": datetime.utcnow().isoformat() + "Z",
                        "fps": fps_ee,
                        "total_events": len(engine_meta_events),
                        "events": [
                            {
                                "frame": em.frame,
                                "priority": em.priority,
                                "event_text": em.event_text,
                                **({"track_id": em.track_id} if em.track_id is not None else {}),
                            }
                            for em in engine_meta_events
                        ],
                    },
                    _ef,
                    ensure_ascii=False,
                    indent=2,
                )
        except Exception as _ee_err:
            engine_meta_events = []
            event_engine_output_path = None
        stage_timings_sec["event_engine"] = round(time.perf_counter() - stage_start, 3)
        emit("event_engine", 1, 1, f"EventEngine tamam ({len(engine_meta_events)} olay)")

    if cfg.run_action_spotting:
        emit("action_spotting", 0, 1, "Action spotting başlıyor (T-DEED)")
        stage_start = time.perf_counter()
        emit("action_spotting", 0, 1, "T-DEED SoccerNet inference çalışıyor")
        primary_events, primary_meta = run_action_spotting_tdeed(
            video_path=segment_path,
            out_dir=out_dir,
            run_id=run_id,
            model_name=str(getattr(cfg, "action_model_name", "SoccerNet_big") or "SoccerNet_big"),
            threshold=float(getattr(cfg, "action_threshold", 0.50)),
            nms_window_sec=float(getattr(cfg, "action_nms_window_sec", 10.0)),
            frame_width=int(getattr(cfg, "action_frame_width", 398) or 398),
            frame_height=int(getattr(cfg, "action_frame_height", 224) or 224),
            checkpoint_path=getattr(cfg, "checkpoint_path", None),
            tdeed_repo_dir=getattr(cfg, "tdeed_repo_dir", None),
            variant="primary",
        )
        events.extend(primary_events)
        action_spotting_metadata["primary"] = primary_meta

        if bool(getattr(cfg, "run_ball_action_spotting", True)):
            emit("action_spotting", 0, 1, "T-DEED SoccerNetBall inference çalışıyor")
            ball_events, ball_meta = run_action_spotting_tdeed(
                video_path=segment_path,
                out_dir=out_dir,
                run_id=run_id,
                model_name=str(getattr(cfg, "ball_model_name", "SoccerNetBall_challenge2") or "SoccerNetBall_challenge2"),
                threshold=float(getattr(cfg, "ball_action_threshold", 0.20)),
                nms_window_sec=float(getattr(cfg, "action_nms_window_sec", 10.0)),
                frame_width=int(getattr(cfg, "action_frame_width", 398) or 398),
                frame_height=int(getattr(cfg, "action_frame_height", 224) or 224),
                checkpoint_path=getattr(cfg, "ball_checkpoint_path", None),
                tdeed_repo_dir=getattr(cfg, "tdeed_repo_dir", None),
                variant="ball",
            )
            events.extend(ball_events)
            action_spotting_metadata["ball"] = ball_meta

        if action_spotting_metadata:
            action_spotting_metadata_path = str(Path(out_dir) / f"action_spotting_meta_{run_id}.json")
            with open(action_spotting_metadata_path, "w", encoding="utf-8") as _mf:
                json.dump(action_spotting_metadata, _mf, ensure_ascii=False, indent=2)

        stage_timings_sec["action_spotting"] = round(time.perf_counter() - stage_start, 3)
        emit("action_spotting", 1, 1, "Action spotting tamam")

    events.sort(key=lambda e: float(e.get("t", 0.0)))

    if track_id_remap:
        for e in events:
            for k in ("player_track_id", "from_player_track_id"):
                try:
                    v = e.get(k)
                    if v is None:
                        continue
                    e[k] = _resolve_track_id_remap(int(v), track_id_remap)
                except Exception:
                    continue

    if jersey_by_track:
        for e in events:
            try:
                pid = e.get("player_track_id")
                if pid is not None:
                    info = jersey_by_track.get(int(pid))
                    if info and info.get("jersey_number") is not None:
                        e["jersey_number"] = info.get("jersey_number")
                fpid = e.get("from_player_track_id")
                if fpid is not None:
                    info2 = jersey_by_track.get(int(fpid))
                    if info2 and info2.get("jersey_number") is not None:
                        e["from_jersey_number"] = info2.get("jersey_number")
            except Exception:
                continue

    commentary_input_path: Optional[str] = None
    commentary_output_path: Optional[str] = None
    commentary_audio_manifest_path: Optional[str] = None
    commentary_video_path: Optional[str] = None
    commentary_gpu_cleanup: Optional[Dict[str, Any]] = None
    commentary_error: Optional[str] = None
    product_video_path: str = segment_path

    overlay_events = [e for e in events if str(e.get("source")) == "action_spotting"]
    if not overlay_events:
        overlay_events = events

    base_video = segment_path
    final_overlay_path = str(Path(out_dir) / f"overlay_{run_id}.mp4")
    emit("overlay", 0, 1, "Overlay başlıyor")
    stage_start = time.perf_counter()
    overlay_events_on_video(
        base_video_path=base_video,
        out_path=final_overlay_path,
        events=overlay_events,
        window_sec=float(cfg.overlay_event_window_sec),
        tracks_csv_path=tracks_csv_path,
        jersey_by_track=jersey_by_track,
        track_id_remap=track_id_remap,
        progress_cb=progress_cb,
    )
    stage_timings_sec["overlay"] = round(time.perf_counter() - stage_start, 3)
    emit("overlay", 1, 1, "Overlay tamam")

    try:
        if bool(getattr(cfg, "run_commentary", True)):
            stage_start = time.perf_counter()
            action_events = [e for e in events if str(e.get("source")) == "action_spotting"]

            if action_events or engine_meta_events:
                if bool(getattr(cfg, "qwen_vl_stop_before_commentary", True)):
                    evt = _set_qwen_vl_container_state(cfg=cfg, action="stop", progress_cb=progress_cb)
                    qwen_vl_container_events.append(evt)
                    # Only block if container management is enabled AND stop failed.
                    # If management is disabled (skipped), proceed normally.
                    if not evt.get("skipped"):
                        state_after = str(evt.get("state_after") or "").strip().lower()
                        if evt.get("error") or state_after not in ("exited", "dead", "created"):
                            raise RuntimeError(
                                "Qwen-VL container stop could not be confirmed before commentary generation"
                            )

                if bool(getattr(cfg, "commentary_flush_gpu_before_llm", True)):
                    emit("commentary_cleanup", 0, 1, "GPU belleği temizleniyor")
                    commentary_gpu_cleanup = _flush_gpu_vram()
                    emit("commentary_cleanup", 1, 1, "GPU belleği temizlendi")

                _build_fps: float = 25.0
                try:
                    _cap_fps_build = cv2.VideoCapture(segment_path)
                    _build_fps = float(_cap_fps_build.get(cv2.CAP_PROP_FPS) or 25.0)
                    _cap_fps_build.release()
                except Exception:
                    pass

                items_in = _build_commentary_items(
                    events=events,
                    action_events=action_events,
                    calibration_frames=calibration_frames,
                    calibration_frame_times=calibration_frame_times,
                    jersey_by_track=jersey_by_track,
                    track_id_remap=track_id_remap,
                    cfg=cfg,
                    engine_meta_events=engine_meta_events,
                    video_fps=_build_fps,
                )

                # ── EventEngine bağlamını commentary itemlarına göm ────────
                # Her T-DEED olayını frame bazında yakın EventEngine metadatasiyla zenginleştir.
                if engine_meta_events:
                    cap_fps_tmp = cv2.VideoCapture(segment_path)
                    _fps_for_merge = float(cap_fps_tmp.get(cv2.CAP_PROP_FPS) or 25.0)
                    cap_fps_tmp.release()
                    for _item in items_in:
                        try:
                            _tdeed_ev = {
                                "label": str(_item.get("event_label") or ""),
                                "t": float(_item.get("t", 0.0)),
                                "timecode": str(_item.get("event_timecode") or ""),
                                "confidence": float(_item.get("event_confidence") or 1.0),
                                "source": "tdeed",
                            }
                            _merged = merge_tdeed_with_event_engine(
                                tdeed_event=_tdeed_ev,
                                engine_events=engine_meta_events,
                                fps=_fps_for_merge,
                                frame_window=int(_fps_for_merge * 3),  # ±3 saniye
                            )
                            if _merged.get("context"):
                                _item["event_engine_context"] = _merged["context"]
                        except Exception:
                            pass

                commentary_input_path = str(Path(out_dir) / f"commentary_input_{run_id}.json")
                _match_ctx = _build_match_context(
                    roster_json_str=getattr(cfg, "player_roster_json", None),
                    jersey_by_track=jersey_by_track,
                )

                # ── event_engine_context: "Takım A/B" → gerçek takım adı ──
                if _match_ctx:
                    _team_names_map = _match_ctx.get("team_names") or {}
                    _t0 = _team_names_map.get("0", "Takım 0")
                    _t1 = _team_names_map.get("1", "Takım 1")
                    _roster_names_map: Dict[str, str] = {}
                    # name → team_id (roster cross-validation için)
                    # NOT: rosters key'leri takım ismi ("Galatasaray"/"Juventus"), team_names ise "0"/"1"
                    _team_name_to_id_enrich: Dict[str, str] = {v: k for k, v in _team_names_map.items()}
                    _name_to_roster_team: Dict[str, str] = {}
                    for _rtname, _tp in (_match_ctx.get("rosters") or {}).items():
                        _rtid = _team_name_to_id_enrich.get(str(_rtname), str(_rtname))
                        for _rp in (_tp if isinstance(_tp, list) else []):
                            if isinstance(_rp, dict) and _rp.get("name") and _rp.get("number") is not None:
                                try:
                                    _roster_names_map[str(int(_rp["number"]))] = str(_rp["name"])
                                except Exception:
                                    pass
                            if isinstance(_rp, dict) and _rp.get("name"):
                                _name_to_roster_team[str(_rp["name"])] = _rtid
                    for _item in items_in:
                        for _ec in (_item.get("event_engine_context") or []):
                            _raw = str(_ec.get("text") or "")
                            _raw = re.sub(r"Tak[iı]m\s+A\b", _t0, _raw, flags=re.IGNORECASE)
                            _raw = re.sub(r"Tak[iı]m\s+B\b", _t1, _raw, flags=re.IGNORECASE)
                            # #N → PlayerName eğer roster'da varsa;
                            # yoksa: track_id'den team_id al → takım adı + "bir oyuncusu"
                            _last_found_name: List[str] = []
                            def _repl_jersey(m: re.Match) -> str:
                                _n = m.group(1)
                                _nm = _roster_names_map.get(_n)
                                if _nm:
                                    _last_found_name.append(_nm)
                                    return f"{_nm}"
                                # Roster'da yok — track_id üzerinden team adını dene
                                _ec_tid = _ec.get("track_id")
                                if _ec_tid is not None:
                                    _ec_jinfo = jersey_by_track.get(int(_ec_tid), {})
                                    _ec_team_id = _ec_jinfo.get("team_id")
                                    if _ec_team_id is not None:
                                        _ec_tname = _team_names_map.get(str(_ec_team_id), "")
                                        if _ec_tname:
                                            return f"{_ec_tname} oyuncusu"
                                return "bir oyuncu"
                            _raw = re.sub(r"#(\d+)", _repl_jersey, _raw)
                            # ec'ye team_id yaz (yoksa) — _build_commentary_item_prompt'ta _tn() kullanır
                            if _ec.get("team_id") is None:
                                _ec_tid2 = _ec.get("track_id")
                                if _ec_tid2 is not None:
                                    _ec_jinfo2 = jersey_by_track.get(int(_ec_tid2), {})
                                    _ec_team_id2 = _ec_jinfo2.get("team_id")
                                    if _ec_team_id2 is not None:
                                        _ec["team_id"] = _ec_team_id2
                            _ec["text"] = _raw
                            # player_name field'ını da doldur (track_id → jersey → roster önceliği)
                            if not _ec.get("player_name"):
                                _tid = _ec.get("track_id")
                                _pname_resolved: Optional[str] = None
                                if _tid is not None:
                                    _jinfo = jersey_by_track.get(int(_tid), {})
                                    _jnum = str(_jinfo.get("jersey_number") or "-1")
                                    _pname_resolved = _jinfo.get("player_name") or _roster_names_map.get(_jnum)
                                if not _pname_resolved and _last_found_name:
                                    _pname_resolved = _last_found_name[0]
                                if _pname_resolved:
                                    _ec["player_name"] = _pname_resolved
                            # Roster cross-validation: event_engine yanlış takım atamış olabilir
                            # → player_name roster'da farklı takımdaysa hem team_id hem text'i düzelt
                            _pname_chk = _ec.get("player_name")
                            if _pname_chk and _pname_chk in _name_to_roster_team:
                                _correct_tid_str = _name_to_roster_team[_pname_chk]
                                _current_tid_str = str(_ec.get("team_id", ""))
                                if _current_tid_str != _correct_tid_str:
                                    _wrong_tname = _team_names_map.get(_current_tid_str, "")
                                    _right_tname = _team_names_map.get(_correct_tid_str, "")
                                    try:
                                        _ec["team_id"] = int(_correct_tid_str)
                                    except Exception:
                                        _ec["team_id"] = _correct_tid_str
                                    if _wrong_tname and _right_tname:
                                        _ec["text"] = str(_ec.get("text", "")).replace(_wrong_tname, _right_tname)
                with open(commentary_input_path, "w", encoding="utf-8") as f:
                    json.dump(
                        {
                            "schema_version": "1.0",
                            "run_id": run_id,
                            "created_utc": datetime.utcnow().isoformat() + "Z",
                            **({"match_context": _match_ctx} if _match_ctx else {}),
                            "items": items_in,
                        },
                        f,
                        ensure_ascii=False,
                        indent=2,
                    )

                llm_backend = str(getattr(cfg, "commentary_llm_backend", "vllm") or "vllm").strip()
                llm_url = str(getattr(cfg, "commentary_llm_url", "http://localhost:8001/") or "http://localhost:8001/").strip()
                llm_model = str(getattr(cfg, "commentary_llm_model", "nvidia/Qwen3-8B-NVFP4") or "nvidia/Qwen3-8B-NVFP4").strip()
                llm_batch_size = int(getattr(cfg, "commentary_vllm_batch_size", 4) or 4)
                llm_enable_thinking = bool(getattr(cfg, "commentary_vllm_enable_thinking", True))
                llm_max_tokens = int(getattr(cfg, "commentary_vllm_max_tokens", 1800) or 1800)
                items_out: List[Dict[str, Any]] = []
                item_errors: List[str] = []
                recent_texts: List[str] = []
                timeout_sec = float(getattr(cfg, "commentary_llm_timeout_sec", 180.0) or 180.0)
                total_items = max(1, len(items_in))

                use_batch = _normalize_commentary_backend(llm_backend) == "vllm" and llm_batch_size > 1

                if use_batch:
                    all_prompts: List[str] = []
                    name_only_map: Dict[int, str] = {}  # idx → direct text (skip LLM)
                    _recent_tmp: List[str] = []
                    for i, it in enumerate(items_in):
                        direct = _try_name_only_text(it)
                        if direct is not None:
                            name_only_map[i] = direct
                            all_prompts.append("")  # placeholder; won't be sent
                        else:
                            all_prompts.append(_build_commentary_item_prompt(it, _recent_tmp, match_context=_match_ctx))
                        _recent_tmp.append("")

                    # Only send non-name-only prompts to LLM
                    llm_indices = [i for i in range(len(items_in)) if i not in name_only_map]
                    llm_items = [items_in[i] for i in llm_indices]
                    llm_prompts = [all_prompts[i] for i in llm_indices]

                    emit("commentary_llm", 0, total_items, f"Batch yorum üretiliyor (batch_size={llm_batch_size})")
                    if llm_prompts:
                        batch_results_llm = _request_commentary_batch(
                            prompts=llm_prompts,
                            base_url=llm_url,
                            model=llm_model,
                            backend=llm_backend,
                            timeout_sec=timeout_sec,
                            enable_thinking=llm_enable_thinking,
                            max_tokens=llm_max_tokens,
                            batch_size=llm_batch_size,
                            emit_cb=emit,
                            timecodes=[str(it.get("timecode") or "") for it in llm_items],
                        )
                        llm_result_map: Dict[int, tuple] = {idx: res for idx, res in zip(llm_indices, batch_results_llm)}
                    else:
                        llm_result_map = {}

                    for idx, it in enumerate(items_in, start=1):
                        emit("commentary_llm", idx, total_items, f"Yorum işleniyor ({idx}/{total_items})")
                        i = idx - 1
                        if i in name_only_map:
                            final_text = name_only_map[i]
                            err = None
                            raw = ""
                        else:
                            raw, err = llm_result_map.get(i, ("", "LLM result missing"))
                            txt = _extract_commentary_text_best_effort(str(raw or "")) if raw else None
                            final_text = _sanitize_commentary_text(str(txt or ""), it, recent_texts)
                        if err:
                            item_errors.append(f"{it.get('timecode')}: {err}")
                        recent_texts.append(final_text)
                        items_out.append(
                            {
                                **it,
                                "text": final_text,
                                "commentary_text": final_text,
                                "llm_raw": str(raw or "")[:1000],
                                "llm_item_error": err,
                            }
                        )
                else:
                    for idx, it in enumerate(items_in, start=1):
                        emit("commentary_llm", idx - 1, total_items, f"Yorum üretiliyor ({idx}/{total_items})")
                        direct = _try_name_only_text(it)
                        if direct is not None:
                            final_text = direct
                            raw, err = "", None
                        else:
                            prompt = _build_commentary_item_prompt(it, recent_texts, match_context=_match_ctx)
                            raw, err = _request_commentary_text(
                                base_url=llm_url,
                                model=llm_model,
                                prompt=prompt,
                                backend=llm_backend,
                                timeout_sec=timeout_sec,
                                enable_thinking=llm_enable_thinking,
                                max_tokens=llm_max_tokens,
                                timecode=str(it.get("timecode") or ""),
                                emit_cb=emit,
                            )
                            txt = _extract_commentary_text_best_effort(str(raw or "")) if raw else None
                            final_text = _sanitize_commentary_text(str(txt or ""), it, recent_texts)
                        if err:
                            item_errors.append(f"{it.get('timecode')}: {err}")
                        recent_texts.append(final_text)
                        items_out.append(
                            {
                                **it,
                                "text": final_text,
                                "commentary_text": final_text,
                                "llm_raw": str(raw or "")[:1000],
                                "llm_item_error": err,
                            }
                        )
                emit("commentary_llm", total_items, total_items, "Yorumlar hazır")

                err = "; ".join(item_errors[:10]) if item_errors else None

                commentary_output_path = str(Path(out_dir) / f"commentary_llm_{run_id}.json")
                with open(commentary_output_path, "w", encoding="utf-8") as f:
                    json.dump(
                        {
                            "schema_version": "1.0",
                            "run_id": run_id,
                            "created_utc": datetime.utcnow().isoformat() + "Z",
                            "llm_backend": llm_backend,
                            "llm_url": llm_url,
                            "llm_model": llm_model,
                            "llm_error": err,
                            **({"gpu_cleanup": commentary_gpu_cleanup} if commentary_gpu_cleanup else {}),
                            "items": items_out,
                        },
                        f,
                        ensure_ascii=False,
                        indent=2,
                    )

                if bool(getattr(cfg, "commentary_enable_tts", True)):
                    try:
                        from commentary_engine import CommentaryEngine  # type: ignore

                        tts_backend = str(getattr(cfg, "commentary_tts_backend", "xttsv2") or "xttsv2")
                        speaker_wav = getattr(cfg, "commentary_speaker_wav", None)
                        ce = CommentaryEngine(
                            output_dir=str(Path(out_dir) / f"commentary_{run_id}"),
                            enable_llm=False,
                            tts_backend=tts_backend,
                            speaker_wav=speaker_wav,
                        )

                        audio_manifest: List[Dict[str, Any]] = []
                        clips: List[Tuple[float, str]] = []
                        # Kesin çakışma önleme: her klibin (başlangıç, bitiş, cooldown) tuple'ını tut
                        # Üçüncü eleman: bu klip bittikten sonra kaç saniye sessiz kalınmalı (source-bazlı)
                        _scheduled_intervals: List[Tuple[float, float, float]] = []
                        min_audio_gap = max(0.0, float(getattr(cfg, "commentary_min_audio_gap_sec", 0.5) or 0.5))
                        _as_cooldown = max(4.0, float(getattr(cfg, "commentary_action_spotting_cooldown_sec", 8.0) or 8.0))
                        _filler_cooldown = max(2.0, float(getattr(cfg, "commentary_min_clip_cooldown_sec", 2.0) or 2.0))

                        def _find_free_start(desired_t: float, dur: float, win_start: float, gap: float) -> float:
                            """
                            desired_t'de başlamak istiyoruz; mevcut kliplerle çakışıyorsa
                            ilk uygun noktayı bul. Her klibin kendi cooldown süresi var.
                            """
                            candidate = max(float(win_start), float(desired_t))
                            for _ in range(80):
                                overlap = False
                                for _s, _e, _cd in _scheduled_intervals:
                                    # Yeni klibin başlangıcı → _e + _cd'den önce olamaz
                                    # Mevcut klibin başlangıcı → candidate + dur + gap'ten önce olamaz
                                    if candidate < _e + _cd and candidate + dur + gap > _s:
                                        overlap = True
                                        candidate = _e + _cd
                                        break
                                if not overlap:
                                    return candidate
                            return candidate

                        for it in items_out:
                            window = it.get("window") if isinstance(it, dict) else None
                            window = window if isinstance(window, dict) else {}
                            window_start = float(window.get("start_t", it.get("t", 0.0)) or 0.0)
                            window_end = float(window.get("end_t", window_start + _commentary_item_period_sec(it)) or (window_start + _commentary_item_period_sec(it)))
                            tt = float(it.get("speech_t", it.get("t", 0.0)) or 0.0)
                            txt = str(it.get("commentary_text") or "").strip()
                            if not txt:
                                continue
                            # Kaynak bazlı cooldown: action_spotting > pause_detector > filler
                            _src_it = str(it.get("event_source") or "").strip().lower()
                            _clip_cooldown = _as_cooldown if _src_it == "action_spotting" else _filler_cooldown
                            # Emotional goal prefix — XTTS reads caps+exclamation louder
                            _lbl_it = str(it.get("event_label", "")).strip().lower()
                            if _lbl_it in ("goal", "own goal", "penalty - goal") and "GOL" not in txt[:8].upper():
                                txt = "GOOOL! " + txt
                            r = ce.synthesize_commentary(text=txt, t_seconds=tt)
                            ap = r.get("audio_path")
                            if ap:
                                dur = _audio_duration_sec(str(ap))
                                _WIDE_BUDGET = {"goal", "own goal", "penalty - goal", "red card", "penalty"}
                                allowed_dur = (
                                    max(15.0, float(window_end) - float(window_start))
                                    if _lbl_it in _WIDE_BUDGET
                                    else max(3.0, float(window_end) - max(float(window_start), float(tt)) - float(min_audio_gap))
                                )
                                final_txt = txt
                                if dur > allowed_dur and final_txt:
                                    shorter = _trim_commentary_text(
                                        final_txt,
                                        max_sentences=1,
                                        max_words=_commentary_word_budget(max(4.0, allowed_dur)),
                                    )
                                    if shorter and shorter != final_txt:
                                        r_retry = ce.synthesize_commentary(text=shorter, t_seconds=tt)
                                        ap_retry = r_retry.get("audio_path")
                                        if ap_retry:
                                            r = r_retry
                                            ap = ap_retry
                                            dur = _audio_duration_sec(str(ap_retry))
                                            final_txt = shorter

                                # Çakışmasız başlangıç noktası bul (source cooldown ile)
                                scheduled_t = _find_free_start(
                                    desired_t=tt,
                                    dur=dur,
                                    win_start=window_start,
                                    gap=min_audio_gap,
                                )
                                _scheduled_intervals.append((float(scheduled_t), float(scheduled_t) + float(dur), _clip_cooldown))
                                _scheduled_intervals.sort(key=lambda x: x[0])
                                clips.append((scheduled_t, str(ap)))
                                audio_manifest.append(
                                    {
                                        **{**it, "commentary_text": final_txt, "text": final_txt},
                                        **r,
                                        "scheduled_t": float(scheduled_t),
                                        "scheduled_timecode": _timecode_mmss(float(scheduled_t)),
                                        "audio_duration_sec": float(dur),
                                        "clip_cooldown_sec": _clip_cooldown,
                                        "window_fit_ok": bool((float(scheduled_t) + float(dur)) <= (float(window_end) + 1e-6)),
                                        "overlap_free": True,
                                    }
                                )
                            else:
                                audio_manifest.append({**it, **r})

                        # --- Filler commentary: fill long silences between main clips ---
                        filler_gap_sec = max(3.0, float(getattr(cfg, "commentary_filler_gap_sec", 5.0) or 5.0))

                        _GENERIC_FILLERS = [
                            "Oyun devam ediyor, takımlar rakip yarı sahada gedik arıyor.",
                            "Top el değiştiriyor; orta saha mücadelesi sürüyor.",
                            "Her iki takım da tempo tutturmaya çalışıyor.",
                            "Savunma hatları sıkı, hücum için alan bulmak zorlaşıyor.",
                            "Oyun akıyor; dikkatler dağılmadan sahaya odaklanmak şart.",
                            "Tempo henüz oturmuş değil, iki takım da ritim arıyor.",
                            "Maç sakin bir seyir izliyor; kritik an her an gelebilir.",
                            "Sahada temkinli bir oyun var; her pas hesaplı atılıyor.",
                        ]
                        _PAUSED_FILLERS = [
                            "Sahada kısa bir duraklama var.",
                            "Oyun kesildi, oyuncular pozisyon alıyor.",
                            "Maçta kısa bir ara.",
                        ]

                        poss_events_all = [
                            e for e in events
                            if str(e.get("type") or "") in ("possession_change", "possession_start")
                            and str(e.get("jersey_number") or "").strip() not in ("", "-1")
                        ]

                        # Gol event zamanları — filler bu pencerelerde çalmasın
                        _goal_event_times: List[float] = []
                        for _m_it in audio_manifest:
                            _m_lbl = str(_m_it.get("event_label", "")).strip().lower()
                            _m_txt = str(_m_it.get("commentary_text") or _m_it.get("text") or "").strip().upper()
                            if (_m_lbl in {"goal", "own goal", "penalty - goal"}
                                    or _m_txt.startswith("GOOOOL") or _m_txt.startswith("GOOOL")
                                    or "GOOOOL" in _m_txt[:20]):
                                _goal_event_times.append(
                                    float(_m_it.get("scheduled_t") or _m_it.get("event_t") or _m_it.get("t") or 0.0)
                                )
                        _POST_GOAL_SILENCE_SEC = 25.0  # gol sonrası filler yasak penceresi

                        def _slot_free_filler(t: float, dur_est: float = 2.0) -> bool:
                            """True if the [t, t+dur_est] window doesn't clash with any scheduled clip."""
                            for _cs, _ce, _cd in _scheduled_intervals:
                                if t < _ce + _cd and t + dur_est > _cs:
                                    return False
                            return True

                        # Find silence gaps larger than filler_gap_sec
                        gaps_filler: List[Tuple[float, float]] = []
                        _prev_end_g = 0.0
                        for _occ_s, _occ_e, _occ_cd in _scheduled_intervals:
                            if _occ_s - _prev_end_g > filler_gap_sec:
                                gaps_filler.append((_prev_end_g, _occ_s))
                            _prev_end_g = max(_prev_end_g, _occ_e + _occ_cd)

                        _generic_idx = 0
                        for gap_s, gap_e in gaps_filler:
                            if gap_e - gap_s < filler_gap_sec:
                                continue
                            # Gol sonrası penceresindeyse filler koyma
                            _near_goal = any(
                                gap_s >= gt - 2.0 and gap_s <= gt + _POST_GOAL_SILENCE_SEC
                                for gt in _goal_event_times
                            )
                            if _near_goal:
                                continue
                            # Possession-based fillers within this gap
                            for pe in poss_events_all:
                                pe_t = float(pe.get("t", 0.0))
                                if pe_t < gap_s + 1.5 or pe_t > gap_e - 1.5:
                                    continue
                                if not _slot_free_filler(pe_t):
                                    continue
                                jersey_f = str(pe.get("jersey_number") or "").strip()
                                from_jersey_f = str(pe.get("from_jersey_number") or "").strip()
                                has_from = from_jersey_f not in ("", "-1")
                                has_to = jersey_f not in ("", "-1")
                                if not has_to:
                                    continue
                                filler_txt = f"Forma {from_jersey_f if has_from else jersey_f} topa sahip."
                                try:
                                    fr_r = ce.synthesize_commentary(text=filler_txt, t_seconds=pe_t)
                                    fap = fr_r.get("audio_path")
                                    if fap and os.path.isfile(str(fap)):
                                        fdur = _audio_duration_sec(str(fap))
                                        _scheduled_intervals.append((float(pe_t), float(pe_t) + max(0.5, fdur), _filler_cooldown))
                                        _scheduled_intervals.sort(key=lambda x: x[0])
                                        clips.append((float(pe_t), str(fap)))
                                except Exception:
                                    pass
                            # Walk through the gap with stride=filler_gap_sec.
                            _gt = gap_s + filler_gap_sec
                            while _gt < gap_e - 0.5:
                                if _slot_free_filler(_gt):
                                    _nearby_paused = any(
                                        "duraksadı" in str(((_it_p.get("match_state") or {}).get("state_summary") or {}).get("ball_progression", ""))
                                        or (len(list((_it_p.get("match_state") or {}).get("frame_samples") or [])) > 0
                                            and sum(1 for _fp in list((_it_p.get("match_state") or {}).get("frame_samples") or []) if _fp.get("ball") is not None)
                                            / max(1, len(list((_it_p.get("match_state") or {}).get("frame_samples") or []))) < 0.25)
                                        for _it_p in items_out
                                        if abs(float(_it_p.get("event_t", _it_p.get("t", 0.0)) or 0.0) - _gt) < 15.0
                                    )
                                    if _nearby_paused:
                                        gen_txt = _PAUSED_FILLERS[_generic_idx % len(_PAUSED_FILLERS)]
                                    else:
                                        gen_txt = _GENERIC_FILLERS[_generic_idx % len(_GENERIC_FILLERS)]
                                    _generic_idx += 1
                                    try:
                                        gen_r = ce.synthesize_commentary(text=gen_txt, t_seconds=_gt)
                                        gen_ap = gen_r.get("audio_path")
                                        if gen_ap and os.path.isfile(str(gen_ap)):
                                            gen_dur = _audio_duration_sec(str(gen_ap))
                                            _scheduled_intervals.append((float(_gt), float(_gt) + max(0.5, gen_dur), _filler_cooldown))
                                            _scheduled_intervals.sort(key=lambda x: x[0])
                                            clips.append((float(_gt), str(gen_ap)))
                                    except Exception:
                                        pass
                                _gt += filler_gap_sec

                        # Collect goal event timestamps for crowd cheer + golsesi SFX
                        _GOAL_SFX_PRE_SEC = 0.5   # golsesi event_t'den 0.5s önce gelsin
                        goal_labels = {"goal", "own goal", "penalty - goal"}
                        goal_sfx_path  = str(getattr(cfg, "commentary_goal_sfx_audio", "") or "")
                        goal_voice_path = str(getattr(cfg, "commentary_goal_voice_audio", "") or "")
                        goal_sfx_clips: List[Tuple] = []   # (t, path, volume)
                        _sfx_have_cheer = bool(goal_sfx_path and os.path.isfile(goal_sfx_path))
                        _sfx_have_voice = bool(goal_voice_path and os.path.isfile(goal_voice_path))
                        if _sfx_have_cheer or _sfx_have_voice:
                            for m_it in audio_manifest:
                                m_lbl = str(m_it.get("event_label", "")).strip().lower()
                                m_txt = str(m_it.get("commentary_text") or m_it.get("text") or "").strip().upper()
                                # event_label "goal" değilse bile LLM "GOOOOL"/"GOOOL" ürettiyse çal
                                _is_goal_item = (
                                    m_lbl in goal_labels
                                    or m_txt.startswith("GOOOOL")
                                    or m_txt.startswith("GOOOL")
                                    or m_txt.startswith("GOL!")
                                    or m_txt.startswith("GOOOL!")
                                    or "GOOOOL" in m_txt[:20]
                                )
                                if _is_goal_item:
                                    # SFX: event_t'den _GOAL_SFX_PRE_SEC önce başlasın
                                    _ev_t = float(m_it.get("event_t") or m_it.get("scheduled_t") or m_it.get("t") or 0.0)
                                    _sfx_t = max(0.0, _ev_t - _GOAL_SFX_PRE_SEC)
                                    if _sfx_have_voice:
                                        goal_sfx_clips.append((_sfx_t, goal_voice_path, 1.0))   # spiker GOL sesi — tam ses
                                    if _sfx_have_cheer:
                                        goal_sfx_clips.append((_sfx_t, goal_sfx_path, 0.65))    # kalabalık tezahürat

                        commentary_audio_manifest_path = str(Path(out_dir) / f"commentary_audio_{run_id}.json")
                        with open(commentary_audio_manifest_path, "w", encoding="utf-8") as f:
                            json.dump(
                                {
                                    "schema_version": "1.0",
                                    "run_id": run_id,
                                    "created_utc": datetime.utcnow().isoformat() + "Z",
                                    "tts_backend": tts_backend,
                                    "items": audio_manifest,
                                },
                                f,
                                ensure_ascii=False,
                                indent=2,
                            )

                        # ── Okunabilir transcript dosyası ──────────────────────────────────
                        try:
                            _tx_path = str(Path(out_dir) / f"commentary_transcript_{run_id}.txt")
                            _tx_lines: List[str] = [
                                f"Commentary Transcript — run {run_id}",
                                "=" * 60,
                                "",
                            ]
                            for _tx_it in sorted(audio_manifest, key=lambda x: float(x.get("scheduled_t") or x.get("event_t") or x.get("t") or 0)):
                                _tx_t = float(_tx_it.get("scheduled_t") or _tx_it.get("event_t") or _tx_it.get("t") or 0)
                                _tx_mm = int(_tx_t) // 60
                                _tx_ss = int(_tx_t) % 60
                                _tx_lbl = str(_tx_it.get("event_label") or "").strip()
                                _tx_txt = str(_tx_it.get("commentary_text") or _tx_it.get("text") or "").strip()
                                _tx_dur = _tx_it.get("audio_duration_sec")
                                _dur_str = f"  ({_tx_dur:.1f}s)" if _tx_dur else ""
                                _tx_lines.append(f"[{_tx_mm:02d}:{_tx_ss:02d}] {_tx_lbl:<20s}{_dur_str}")
                                if _tx_txt:
                                    _tx_lines.append(f"         {_tx_txt}")
                                _tx_lines.append("")
                            with open(_tx_path, "w", encoding="utf-8") as _txf:
                                _txf.write("\n".join(_tx_lines))
                        except Exception:
                            pass

                        mixed_path = str(Path(out_dir) / f"product_{run_id}_commentary.mp4")
                        ambient_path_cfg = str(getattr(cfg, "commentary_ambient_audio", "") or "")
                        try:
                            mixed = _mix_commentary_audio_into_video(
                                base_video_path=segment_path,
                                out_path=mixed_path,
                                clips=clips,
                                ambient_path=ambient_path_cfg if os.path.isfile(ambient_path_cfg) else None,
                                sfx_clips=goal_sfx_clips if goal_sfx_clips else None,
                            )
                            if mixed:
                                commentary_video_path = mixed
                                product_video_path = mixed
                        except Exception as _mix_exc:
                            commentary_error = f"mix_error: {_mix_exc}"
                    except Exception as _tts_exc:
                        commentary_error = f"tts_error: {_tts_exc}"
    except Exception as e:
        commentary_error = str(e)

    # --- Container cleanup: ensure Qwen-VL container is stopped after pipeline ---
    # This guarantees a clean state for the next run regardless of what happened above.
    if bool(getattr(cfg, "qwen_vl_manage_container", False)):
        try:
            _stop_evt = _set_qwen_vl_container_state(cfg=cfg, action="stop", progress_cb=None)
            if _stop_evt and not _stop_evt.get("skipped"):
                qwen_vl_container_events.append(_stop_evt)
        except Exception:
            pass

    stage_timings_sec["total"] = round(sum(float(v) for v in stage_timings_sec.values()), 3)

    # --- Match statistics (analyze_match.py) ---
    match_stats_path: Optional[str] = None
    try:
        _big_json = (action_spotting_metadata.get("primary") or {}).get("output_json_path")
        _ball_json = (action_spotting_metadata.get("ball") or {}).get("output_json_path")
        print(
            f"[analyze_match] pre-check: "
            f"tracks_csv={tracks_csv_path!r} exists={os.path.isfile(tracks_csv_path) if tracks_csv_path else False} | "
            f"calib_jsonl={calibration_frames_jsonl_path!r} exists={os.path.isfile(calibration_frames_jsonl_path) if calibration_frames_jsonl_path else False} | "
            f"big_json={_big_json!r} exists={os.path.isfile(_big_json) if _big_json else False} | "
            f"ball_json={_ball_json!r} exists={os.path.isfile(_ball_json) if _ball_json else False}",
            flush=True,
        )
        if (
            tracks_csv_path and os.path.isfile(tracks_csv_path)
            and calibration_frames_jsonl_path and os.path.isfile(calibration_frames_jsonl_path)
            and _big_json and os.path.isfile(_big_json)
            and _ball_json and os.path.isfile(_ball_json)
        ):
            _cap = cv2.VideoCapture(segment_path)
            _seg_fps = float(_cap.get(cv2.CAP_PROP_FPS) or 25.0)
            _cap.release()
            if _seg_fps <= 1e-6:
                _seg_fps = 25.0
            _stats_out = str(Path(out_dir) / f"match_stats_{run_id}.json")
            emit("analyze_match", 0, 1, "Maç istatistikleri hesaplanıyor")
            _analyze_script = str(Path(__file__).resolve().parent.parent.parent / "web" / "backend" / "analyze_match.py")
            if not os.path.isfile(_analyze_script):
                _analyze_script = str(Path(__file__).resolve().parent / "analyze_match.py")
            if os.path.isfile(_analyze_script):
                import subprocess as _sub
                _am_env = os.environ.copy()
                _am_env["PYTHONIOENCODING"] = "utf-8"
                _am_env["PYTHONUNBUFFERED"] = "1"
                _am_cmd = [
                    sys.executable, _analyze_script,
                    "--tracking", tracks_csv_path,
                    "--calibration", calibration_frames_jsonl_path,
                    "--big_model", _big_json,
                    "--ball_model", _ball_json,
                    "--fps", str(_seg_fps),
                    "--output", _stats_out,
                    "--session_id", run_id,
                ]
                _roster_json_str = getattr(cfg, "player_roster_json", None) or None
                _roster_tmp_path: Optional[str] = None
                if _roster_json_str:
                    import tempfile as _tmpmod
                    try:
                        _roster_tmp = _tmpmod.NamedTemporaryFile(
                            mode="w", suffix=".json", delete=False, encoding="utf-8"
                        )
                        _roster_tmp.write(_roster_json_str)
                        _roster_tmp.close()
                        _roster_tmp_path = _roster_tmp.name
                        _am_cmd += ["--roster", _roster_tmp_path]
                    except Exception as _re:
                        print(f"[analyze_match] roster temp write failed: {_re}", flush=True)
                _proc = _sub.run(
                    _am_cmd,
                    capture_output=True, text=True, encoding="utf-8", errors="replace",
                    env=_am_env,
                )
                if _roster_tmp_path:
                    try:
                        os.unlink(_roster_tmp_path)
                    except Exception:
                        pass
                if _proc.returncode == 0 and os.path.isfile(_stats_out):
                    match_stats_path = _stats_out
                else:
                    print(
                        f"[analyze_match] returncode={_proc.returncode}\n"
                        f"STDOUT: {_proc.stdout[-3000:] if _proc.stdout else ''}\n"
                        f"STDERR: {_proc.stderr[-3000:] if _proc.stderr else ''}",
                        flush=True,
                    )
            else:
                print(f"[analyze_match] script not found: {_analyze_script}", flush=True)
            emit("analyze_match", 1, 1, "Maç istatistikleri hazır")
    except Exception as _ae:
        print(f"[analyze_match] exception: {_ae}", flush=True)

    events_json_path = str(Path(out_dir) / f"events_{run_id}.json")
    payload = {
        "schema_version": "1.0",
        "created_utc": datetime.utcnow().isoformat() + "Z",
        "source": {
            "input_video_path": str(Path(video_path).resolve()),
            "segment_video_path": segment_path,
        },
        "runtime_config": {
            **runtime_config,
            **({"resolved_calibration": calibration_runtime_config} if calibration_runtime_config else {}),
        },
        "stage_timings_sec": stage_timings_sec,
        "artifacts": {
            "tracking_video_path": tracking_video_path,
            "tracks_csv_path": tracks_csv_path,
            **({"tracks_csv_raw_path": tracks_csv_raw_path} if tracks_csv_raw_path else {}),
            **({"calibration_map_video_path": calibration_map_video_path} if calibration_map_video_path else {}),
            **({"calibration_events_json_path": calibration_events_json_path} if calibration_events_json_path else {}),
            **({"calibration_frames_jsonl_path": calibration_frames_jsonl_path} if calibration_frames_jsonl_path else {}),
            "overlay_video_path": final_overlay_path,
            "product_video_path": product_video_path,
            **({"commentary_input_path": commentary_input_path} if commentary_input_path else {}),
            **({"commentary_output_path": commentary_output_path} if commentary_output_path else {}),
            **({"commentary_audio_manifest_path": commentary_audio_manifest_path} if commentary_audio_manifest_path else {}),
            **({"commentary_video_path": commentary_video_path} if commentary_video_path else {}),
            **({"action_spotting_metadata_path": action_spotting_metadata_path} if action_spotting_metadata_path else {}),
            **({"event_engine_output_path": event_engine_output_path} if event_engine_output_path else {}),
        },
        **({"action_spotting": action_spotting_metadata} if action_spotting_metadata else {}),
        **({"commentary_error": commentary_error} if commentary_error else {}),
        **({"qwen_vl_container_events": qwen_vl_container_events} if qwen_vl_container_events else {}),
        **({"commentary_gpu_cleanup": commentary_gpu_cleanup} if commentary_gpu_cleanup else {}),
        **({"track_id_remap": track_id_remap} if track_id_remap else {}),
        "event_engine_event_count": len(engine_meta_events),
        "jersey_numbers": list(jersey_by_track.values()) if jersey_by_track else [],
        "events": events,
    }
    with open(events_json_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)

    emit("done", 1, 1, "Pipeline tamam")

    return {
        "run_id": run_id,
        "segment_path": segment_path,
        "stage_timings_sec": stage_timings_sec,
        "runtime_config": {
            **runtime_config,
            **({"resolved_calibration": calibration_runtime_config} if calibration_runtime_config else {}),
        },
        **({"calibration_map_video_path": calibration_map_video_path} if calibration_map_video_path else {}),
        **({"calibration_events_json_path": calibration_events_json_path} if calibration_events_json_path else {}),
        **({"calibration_frames_jsonl_path": calibration_frames_jsonl_path} if calibration_frames_jsonl_path else {}),
        "tracking_video_path": tracking_video_path,
        "tracks_csv_path": tracks_csv_path,
        "overlay_video_path": final_overlay_path,
        "product_video_path": product_video_path,
        **({"commentary_video_path": commentary_video_path} if commentary_video_path else {}),
        **({"commentary_input_path": commentary_input_path} if commentary_input_path else {}),
        **({"commentary_output_path": commentary_output_path} if commentary_output_path else {}),
        **({"commentary_audio_manifest_path": commentary_audio_manifest_path} if commentary_audio_manifest_path else {}),
        **({"commentary_error": commentary_error} if commentary_error else {}),
        **({"qwen_vl_container_events": qwen_vl_container_events} if qwen_vl_container_events else {}),
        **({"commentary_gpu_cleanup": commentary_gpu_cleanup} if commentary_gpu_cleanup else {}),
        **({"action_spotting_metadata_path": action_spotting_metadata_path} if action_spotting_metadata_path else {}),
        **({"event_engine_output_path": event_engine_output_path} if event_engine_output_path else {}),
        "event_engine_event_count": len(engine_meta_events),
        "events_json_path": events_json_path,
        "event_count": len(events),
        **({"match_stats_path": match_stats_path} if match_stats_path else {}),
    }
