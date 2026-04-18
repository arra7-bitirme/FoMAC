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
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

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

    run_commentary: bool = True
    commentary_max_events: int = 30
    commentary_possession_max_age_sec: float = 8.0
    commentary_llm_backend: str = "vllm"
    commentary_llm_url: str = "http://localhost:8001/"
    commentary_llm_model: str = "nvidia/Qwen3-8B-NVFP4"
    commentary_vllm_batch_size: int = 4
    commentary_vllm_enable_thinking: bool = False
    commentary_flush_gpu_before_llm: bool = True
    commentary_context_window_sec: float = 12.0
    commentary_context_stride_sec: float = 1.0
    commentary_context_max_samples: int = 9
    commentary_segment_sec: float = 30.0
    commentary_state_interval_sec: float = 10.0
    commentary_llm_timeout_sec: float = 90.0
    commentary_min_audio_gap_sec: float = 0.35
    commentary_enable_tts: bool = True
    commentary_tts_backend: str = "xttsv2"
    commentary_speaker_wav: Optional[str] = str(Path(__file__).resolve().parent / "ertem_sener.wav")

    possession_dist_norm: float = 0.08
    possession_stable_frames: int = 6
    possession_stride_frames: int = 5
    ball_cls_id: int = 1
    player_cls_id: int = 0

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
            yolo_frame_window = int(getattr(cfg, "calibration_yolo_frame_window", 8) or 8)
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

def _strip_think_blocks(text: str) -> str:
    s = re.sub(r"<think>[\s\S]*?</think>", "", str(text or ""), flags=re.IGNORECASE)
    return s.strip()

def _qwen_text_openai_compatible(
    *,
    base_url: str,
    model: str,
    prompt: str,
    timeout_sec: float = 60.0,
    enable_thinking: bool = True,
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
                "content": "You are a Turkish football commentator.",
            },
            {
                "role": "user",
                "content": str(prompt or ""),
            },
        ],
        "temperature": 0.7,
        "max_tokens": 300,
    }
    if not enable_thinking:
        payload["chat_template_kwargs"] = {"enable_thinking": False}

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

    if not enable_thinking:
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
    enable_thinking: bool = False,
    timecode: Optional[str] = None,
    emit_cb: Optional[Any] = None,
) -> Tuple[Optional[str], Optional[str]]:
    raw, err = _qwen_text_openai_compatible(
        base_url=base_url,
        model=model,
        prompt=prompt,
        timeout_sec=timeout_sec,
        enable_thinking=enable_thinking,
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
        if not enable_thinking:
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
    enable_thinking: bool = False,
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
            {"role": "system", "content": "You are a Turkish football commentator."},
            {"role": "user", "content": ""},
        ],
        "temperature": 0.7,
        "max_tokens": 300,
    }
    if not enable_thinking:
        payload_template["chat_template_kwargs"] = {"enable_thinking": False}

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

def _build_commentary_item_prompt(item: Dict[str, Any], recent_texts: List[str]) -> str:
    style = _commentary_style_for_item(item)
    window = item.get("window") if isinstance(item, dict) else None
    window = window if isinstance(window, dict) else {}
    period_sec = float(window.get("duration_sec", item.get("segment_duration_sec", 10.0)) or 10.0)
    max_sentences = _commentary_sentence_budget(period_sec)
    max_words = _commentary_word_budget(period_sec)
    return (
        "Sen Türkçe futbol anlatımı yapan canlı maç spikerisin.\n"
        "Aşağıda tek bir zaman penceresine ait maç durumu var.\n"
        f"Üslup hedefi: {style}.\n"
        "Görev: Bu pencere için kısa, kesin ve akıcı tek bir yorum üret.\n"
        "Kesin kurallar:\n"
        "- Sadece Türkçe yaz.\n"
        "- JSON object döndür: {\"text\": \"...\"}\n"
        f"- En fazla {max_sentences} cümle kullan.\n"
        f"- En fazla {max_words} kelime kullan.\n"
        f"- Bu yorum yaklaşık {int(round(period_sec))} saniyelik pencereye sığmalı; uzun paragraf yazma.\n"
        "- 'veya', 'ya da', 'olabilir', 'muhtemelen', 'belki', 'gibi görünüyor', 'top görünmüyor' gibi belirsizlik ifadeleri kullanma.\n"
        "- Görsel eksikliği veya veri eksikliği hakkında konuşma.\n"
        "- Eldeki bağlama dayan, ama kesin olmayan oyuncu kimliği uydurma.\n"
        "- Oyun yönü, baskı, kanat kullanımı, merkez bağlantısı, ceza sahası çevresi, savunmadan çıkış gibi futbol dili kullan.\n"
        "- Pası veren oyuncu açıkça destekleniyorsa söyle; emin değilsen isim ya da forma uydurma.\n\n"
        "- Son üretilen cümleleri tekrar etme; her satıra farklı bir giriş ve akış ver.\n\n"
        "Son yorumlar (tekrar etme):\n"
        + json.dumps(recent_texts[-4:], ensure_ascii=False)
        + "\n\n"
        "Pencere verisi JSON:\n"
        + json.dumps(item, ensure_ascii=False, indent=2)
    )

def _extract_commentary_text_best_effort(raw: str) -> Optional[str]:
    s = str(raw or "").strip()
    if not s:
        return None
    s = re.sub(r"<think>[\s\S]*?</think>", "", s, flags=re.IGNORECASE).strip()
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
        return _fallback_commentary_text(item, recent_texts)

    cleaned = raw.replace(" veya ", " ve ").replace(" ya da ", " ve ")
    cleaned = re.sub(r"\b(muhtemelen|belki|galiba|sanirim|sanırım|olabilir)\b", "", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"gibi görünüyor", "", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"top görünmüyor|top görünür değil|net izlenemiyor|görünmüyor", "", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"\s+", " ", cleaned).strip(" ,")

    banned = [" veya ", " ya da ", "olabilir", "muhtemelen", "belki", "görünmüyor", "net izlenemiyor"]
    lower = f" {cleaned.lower()} "
    if any(tok in lower for tok in banned) or not cleaned:
        return _fallback_commentary_text(item, recent_texts)

    period_sec = _commentary_item_period_sec(item)
    max_sentences = _commentary_sentence_budget(period_sec)
    max_words = _commentary_word_budget(period_sec)

    out = _trim_commentary_text(cleaned, max_sentences=max_sentences, max_words=max_words)
    if not out:
        return _fallback_commentary_text(item, recent_texts)
    if _is_repetitive_commentary(out, recent_texts):
        return _fallback_commentary_text(item, recent_texts)
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

def _commentary_sentence_budget(period_sec: float) -> int:
    return 1 if float(period_sec) <= 18.0 else 2

def _commentary_word_budget(period_sec: float) -> int:
    return max(7, min(22, int(round(float(period_sec) * 0.6))))

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

def _build_commentary_items(
    *,
    events: List[Dict[str, Any]],
    action_events: List[Dict[str, Any]],
    calibration_events: List[Dict[str, Any]],
    possession_events: List[Dict[str, Any]],
    calibration_frames: List[Dict[str, Any]],
    calibration_frame_times: List[float],
    jersey_by_track: Dict[int, Dict[str, Any]],
    track_id_remap: Dict[int, int],
    cfg: FullPipelineConfig,
) -> List[Dict[str, Any]]:
    max_events = int(getattr(cfg, "commentary_max_events", 30) or 30)
    segment_sec = max(10.0, float(getattr(cfg, "commentary_segment_sec", 30.0) or 30.0))

    _CRITICAL_LABELS: set = {
        "goal", "yellow card", "red card", "corner", "penalty",
        "own goal", "goal (handball)", "penalty - goal",
        "shot on target", "var",
    }

    anchors: List[Tuple[float, int, Dict[str, Any]]] = []

    for e in action_events:
        try:
            label = str(e.get("label") or e.get("type") or "").strip().lower()
            prio = -1 if label in _CRITICAL_LABELS else 0
            anchors.append((float(e.get("t", 0.0)), prio, e))
        except Exception:
            continue

    for e in calibration_events:
        try:
            anchors.append((float(e.get("t", 0.0)), 1, e))
        except Exception:
            continue

    last_pos_t = -1e9
    pos_gap_sec = min(8.0, max(3.0, segment_sec * 0.35))
    for e in possession_events:
        try:
            tt = float(e.get("t", 0.0))
        except Exception:
            continue
        if tt - last_pos_t >= pos_gap_sec:
            anchors.append((tt, 2, e))
            last_pos_t = tt

    timeline_end = 0.0
    if calibration_frame_times:
        timeline_end = float(calibration_frame_times[-1])
    elif events:
        try:
            timeline_end = max(float(e.get("t", 0.0) or 0.0) for e in events)
        except Exception:
            timeline_end = 0.0

    anchors.sort(key=lambda item: (item[0], item[1]))

    chosen: List[Tuple[float, float, float, int, Dict[str, Any]]] = []
    if timeline_end <= 0.0 and anchors:
        try:
            timeline_end = max(float(a[0]) for a in anchors)
        except Exception:
            timeline_end = 0.0

    if timeline_end <= 0.0:
        timeline_end = float(segment_sec)

    total_segments = max(1, int(np.ceil(float(timeline_end) / float(segment_sec))))
    for seg_idx in range(total_segments):
        seg_start = float(seg_idx) * float(segment_sec)
        seg_end = min(float(timeline_end), seg_start + float(segment_sec))
        if seg_end <= seg_start:
            seg_end = seg_start + float(segment_sec)

        candidates = [(tt, prio, e) for tt, prio, e in anchors if seg_start <= tt < seg_end]
        if candidates:
            best_t, best_prio, best_event = sorted(
                candidates,
                key=lambda item: (
                    int(item[1]),
                    -float(item[2].get("confidence", 0.0) or 0.0),
                    abs(float(item[0]) - ((seg_start + seg_end) * 0.5)),
                ),
            )[0]
        else:
            best_t = float(seg_start + ((seg_end - seg_start) * 0.5))
            best_prio = 3
            best_event = {
                "t": float(best_t),
                "source": "state_window",
                "label": "Match State",
                "description_tr": "Oyun akışı bu bölümde sürüyor.",
                "confidence": 0.5,
            }

        chosen.append((float(seg_start), float(seg_end), float(best_t), int(best_prio), best_event))

    chosen = chosen[: max(1, max_events)]

    items_in: List[Dict[str, Any]] = []
    for seg_start, seg_end, event_t, _prio, ae in chosen:
        actor_tid: Optional[int] = None
        if possession_events:
            actor_tid = _assign_actor_track_id_to_action(
                action_t=float(event_t),
                possession_events=possession_events,
                max_age_sec=float(getattr(cfg, "commentary_possession_max_age_sec", 8.0)),
            )

        actor_info = jersey_by_track.get(int(actor_tid), {}) if actor_tid is not None else {}
        actor_jersey = actor_info.get("jersey_number") if actor_info else None
        actor_team = actor_info.get("team_id") if actor_info else None

        items_in.append(
            {
                "t": float(seg_start),
                "speech_t": float(seg_start),
                "timecode": _timecode_mmss(float(seg_start)),
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
                "event_label": str(ae.get("label") or ae.get("type") or "Match State"),
                "event_confidence": float(ae.get("confidence", 0.0) or 0.0),
                "event_source": str(ae.get("source") or "unknown"),
                "description_tr": str(ae.get("description_tr") or ""),
                "actor_track_id": int(actor_tid) if actor_tid is not None else None,
                "actor_jersey_number": actor_jersey,
                "actor_team_id": actor_team,
                "match_state": _summarize_calibration_window(
                    action_t=float(event_t),
                    calibration_frames=calibration_frames,
                    calibration_times=calibration_frame_times,
                    calibration_events=calibration_events,
                    possession_events=possession_events,
                    jersey_by_track=jersey_by_track,
                    track_id_remap=track_id_remap,
                    window_sec=float(getattr(cfg, "commentary_context_window_sec", 12.0) or 12.0),
                    stride_sec=float(getattr(cfg, "commentary_context_stride_sec", 1.0) or 1.0),
                    max_samples=int(getattr(cfg, "commentary_context_max_samples", 9) or 9),
                ),
            }
        )

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
    calibration_events: List[Dict[str, Any]],
    possession_events: List[Dict[str, Any]],
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
            "calibration_events": [
                _compact_event_for_commentary(e)
                for e in calibration_events
                if start_t <= float(e.get("t", 0.0) or 0.0) <= end_t
            ][:4],
            "possession_events": [
                _compact_event_for_commentary(e)
                for e in possession_events
                if start_t <= float(e.get("t", 0.0) or 0.0) <= end_t
            ][:4],
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
                        if team_id is None and info:
                            team_id = info.get("team_id")
                        nearby.append(
                            (
                                dist,
                                {
                                    "track_id": int(track_id) if track_id > 0 else track_id_raw,
                                    "jersey_number": info.get("jersey_number") if info else None,
                                    "team_id": team_id,
                                    "distance_to_ball_m": round(dist, 2),
                                },
                            )
                        )
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
    window_possession = [
        _compact_event_for_commentary(e)
        for e in possession_events
        if start_t <= float(e.get("t", 0.0) or 0.0) <= end_t
    ][:4]
    if len(window_possession) >= 3:
        state_tags.append("topa sahip olma değişimleri")

    window_calibration = [
        _compact_event_for_commentary(e)
        for e in calibration_events
        if start_t <= float(e.get("t", 0.0) or 0.0) <= end_t
    ][:4]

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
        "calibration_events": window_calibration,
        "possession_events": window_possession,
    }

def _mix_commentary_audio_into_video(
    *,
    base_video_path: str,
    out_path: str,
    clips: List[Tuple[float, str]],
) -> Optional[str]:
    ffmpeg_bin = _ffmpeg_exe()
    if not ffmpeg_bin:
        return None
    if not clips:
        return None

    kept: List[Tuple[float, str]] = []
    for t, p in clips:
        try:
            if os.path.isfile(p) and os.path.getsize(p) > 0:
                kept.append((float(t), str(p)))
        except Exception:
            continue
    if not kept:
        return None

    inputs = ["-i", str(Path(base_video_path).resolve())]
    for _, ap in kept:
        inputs += ["-i", str(Path(ap).resolve())]

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

    parts: List[str] = []
    amix_inputs: List[str] = []

    use_silence = duration_sec is not None and duration_sec > 0.25
    if use_silence:
        dur = max(0.25, float(duration_sec))
        parts.append(
            "anullsrc=channel_layout=stereo:sample_rate=44100,atrim=0:{:.3f},asetpts=N/SR/TB[sil]".format(dur)
        )
        amix_inputs.append("[sil]")

    for i, (t, _ap) in enumerate(kept, start=1):
        delay_ms = max(0, int(round(float(t) * 1000.0)))
        tag = f"a{i}"
        parts.append(
            f"[{i}:a]aformat=sample_fmts=fltp:sample_rates=44100:channel_layouts=stereo,adelay={delay_ms}|{delay_ms}[{tag}]"
        )
        amix_inputs.append(f"[{tag}]")

    mix_duration = "first" if use_silence else "longest"
    parts.append(
        "".join(amix_inputs)
        + f"amix=inputs={len(amix_inputs)}:duration={mix_duration}:dropout_transition=0[outa]"
    )
    filter_complex = ";".join(parts)

    out_path = str(Path(out_path).resolve())
    os.makedirs(str(Path(out_path).parent), exist_ok=True)

    cmd = [
        ffmpeg_bin,
        "-y",
        *inputs,
        "-filter_complex",
        filter_complex,
        "-map",
        "0:v:0",
        "-map",
        "[outa]",
        "-c:v",
        "copy",
        "-c:a",
        "aac",
        "-b:a",
        "160k",
        "-movflags",
        "+faststart",
        out_path,
    ]

    try:
        subprocess.run(cmd, capture_output=True, text=True, check=True)
        return out_path
    except Exception:
        try:
            cmd2 = [
                ffmpeg_bin,
                "-y",
                *inputs,
                "-filter_complex",
                filter_complex,
                "-map",
                "0:v:0",
                "-map",
                "[outa]",
                "-c:v",
                "libx264",
                "-preset",
                "fast",
                "-crf",
                "20",
                "-pix_fmt",
                "yuv420p",
                "-c:a",
                "aac",
                "-b:a",
                "160k",
                "-movflags",
                "+faststart",
                out_path,
            ]
            subprocess.run(cmd2, capture_output=True, text=True, check=True)
            return out_path
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
                prompt=str(getattr(cfg, "jersey_prompt", "") or ""),
            )
            if raw:
                per_track_raw.setdefault(int(track_id), []).append(str(raw))

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
        out[int(tid)] = {
            "track_id": int(tid),
            "jersey_number": jersey_num,
            "confidence": float(jersey_conf.get(int(tid), 0.0 if jersey_num == "-1" else 1.0)),
            "votes": votes,
            "raw": raws[-3:],
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
        return None
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
            str(getattr(cfg, "jersey_prompt", "") or ""),
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

    def _run_tracking_subprocess(cmd_to_run: List[str], log_file: str) -> subprocess.Popen:
        os.makedirs(str(Path(log_file).resolve().parent), exist_ok=True)
        lf = open(log_file, "w", encoding="utf-8", errors="ignore")
        env = os.environ.copy()
        env.setdefault("PYTHONUNBUFFERED", "1")
        env.setdefault("PYTHONFAULTHANDLER", "1")
        env.setdefault("TORCH_SHOW_CPP_STACKTRACES", "1")
        p = subprocess.Popen(cmd_to_run, stdout=lf, stderr=subprocess.STDOUT, text=True, env=env)
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
    tick_pbar = tqdm(total=None, desc="Tracking", unit="s", bar_format="{desc}: {elapsed}")

    try:
        try:
            rf = open(log_path, "r", encoding="utf-8", errors="ignore")
        except Exception:
            rf = None

        while True:
            rc = p.poll()
            any_update = False

            if rf is not None:
                while True:
                    line = rf.readline()
                    if not line:
                        break
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

                        any_update = True

                if rc is not None:
                    break

                if not any_update:
                    time.sleep(1.0)
                    try:
                        tick_pbar.update(1)
                    except Exception:
                        pass
                    if progress_cb is not None:
                        try:
                            elapsed = int(time.time() - start_ts)
                            progress_cb("tracking", elapsed, 0, "Tracking çalışıyor")
                        except Exception:
                            pass
                else:
                    time.sleep(0.2)
    finally:
        try:
            if rf is not None:
                rf.close()
        except Exception:
            pass
        try:
            if pbar is not None:
                pbar.close()
        except Exception:
            pass
        try:
            tick_pbar.close()
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
            rf2 = None
            try:
                rf2 = open(tracking_retry_log_path, "r", encoding="utf-8", errors="ignore")
            except Exception:
                rf2 = None

            retry_last_frame = 0
            retry_total: Optional[int] = None
            while True:
                rc2 = p.poll()
                any_update2 = False

                if rf2 is not None:
                    while True:
                        line2 = rf2.readline()
                        if not line2:
                            break
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

                        any_update2 = True

                if rc2 is not None:
                    break

                if not any_update2:
                    time.sleep(1.0)
                    if progress_cb is not None:
                        try:
                            elapsed2 = int(time.time() - start_ts)
                            progress_cb("tracking", elapsed2, 0, "Tracking (retry) çalışıyor")
                        except Exception:
                            pass
                else:
                    time.sleep(0.2)

            try:
                if rf2 is not None:
                    rf2.close()
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
            if os.path.isfile(jersey_json_path):
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
            possession_events = [
                e
                for e in events
                if str(e.get("source")) == "tracking" and str(e.get("type")) in ("possession_start", "possession_change")
            ]

            if action_events or calibration_events or possession_events:
                if bool(getattr(cfg, "qwen_vl_stop_before_commentary", True)):
                    evt = _set_qwen_vl_container_state(cfg=cfg, action="stop", progress_cb=progress_cb)
                    qwen_vl_container_events.append(evt)
                    state_after = str(evt.get("state_after") or "").strip().lower()
                    if evt.get("error") or state_after not in ("exited", "dead", "created"):
                        raise RuntimeError(
                            "Qwen-VL container stop could not be confirmed before Ollama commentary generation"
                        )

                if bool(getattr(cfg, "commentary_flush_gpu_before_llm", True)):
                    emit("commentary_cleanup", 0, 1, "GPU belleği temizleniyor")
                    commentary_gpu_cleanup = _flush_gpu_vram()
                    emit("commentary_cleanup", 1, 1, "GPU belleği temizlendi")

                items_in = _build_commentary_items(
                    events=events,
                    action_events=action_events,
                    calibration_events=calibration_events,
                    possession_events=possession_events,
                    calibration_frames=calibration_frames,
                    calibration_frame_times=calibration_frame_times,
                    jersey_by_track=jersey_by_track,
                    track_id_remap=track_id_remap,
                    cfg=cfg,
                )

                commentary_input_path = str(Path(out_dir) / f"commentary_input_{run_id}.json")
                with open(commentary_input_path, "w", encoding="utf-8") as f:
                    json.dump(
                        {
                            "schema_version": "1.0",
                            "run_id": run_id,
                            "created_utc": datetime.utcnow().isoformat() + "Z",
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
                llm_enable_thinking = bool(getattr(cfg, "commentary_vllm_enable_thinking", False))
                items_out: List[Dict[str, Any]] = []
                item_errors: List[str] = []
                recent_texts: List[str] = []
                timeout_sec = float(getattr(cfg, "commentary_llm_timeout_sec", 90.0) or 90.0)
                total_items = max(1, len(items_in))

                use_batch = _normalize_commentary_backend(llm_backend) == "vllm" and llm_batch_size > 1

                if use_batch:
                    all_prompts: List[str] = []
                    _recent_tmp: List[str] = []
                    for it in items_in:
                        all_prompts.append(_build_commentary_item_prompt(it, _recent_tmp))
                        _recent_tmp.append("")  # placeholder; will be filled after batch

                    emit("commentary_llm", 0, total_items, f"Batch yorum üretiliyor (batch_size={llm_batch_size})")
                    batch_results = _request_commentary_batch(
                        prompts=all_prompts,
                        base_url=llm_url,
                        model=llm_model,
                        backend=llm_backend,
                        timeout_sec=timeout_sec,
                        enable_thinking=llm_enable_thinking,
                        batch_size=llm_batch_size,
                            emit_cb=emit,
                            timecodes=[str(it.get("timecode") or "") for it in items_in],
                    )
                    for idx, (it, (raw, err)) in enumerate(zip(items_in, batch_results), start=1):
                        emit("commentary_llm", idx, total_items, f"Yorum işleniyor ({idx}/{total_items})")
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
                        prompt = _build_commentary_item_prompt(it, recent_texts)
                        raw, err = _request_commentary_text(
                            base_url=llm_url,
                            model=llm_model,
                            prompt=prompt,
                            backend=llm_backend,
                            timeout_sec=timeout_sec,
                            enable_thinking=llm_enable_thinking,
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
                        min_audio_gap = max(0.0, float(getattr(cfg, "commentary_min_audio_gap_sec", 0.35) or 0.35))
                        next_free_t = 0.0
                        for it in items_out:
                            window = it.get("window") if isinstance(it, dict) else None
                            window = window if isinstance(window, dict) else {}
                            window_start = float(window.get("start_t", it.get("t", 0.0)) or 0.0)
                            window_end = float(window.get("end_t", window_start + _commentary_item_period_sec(it)) or (window_start + _commentary_item_period_sec(it)))
                            tt = float(it.get("speech_t", it.get("t", 0.0)) or 0.0)
                            txt = str(it.get("commentary_text") or "").strip()
                            if not txt:
                                continue
                            r = ce.synthesize_commentary(text=txt, t_seconds=tt)
                            ap = r.get("audio_path")
                            if ap:
                                dur = _audio_duration_sec(str(ap))
                                allowed_dur = max(3.0, float(window_end) - max(float(window_start), float(tt)) - float(min_audio_gap))
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
                                scheduled_t = max(float(window_start), max(float(tt), float(next_free_t)))
                                latest_start = max(float(window_start), float(window_end) - float(dur))
                                if scheduled_t > latest_start:
                                    scheduled_t = latest_start
                                next_free_t = float(scheduled_t) + float(dur) + float(min_audio_gap)
                                clips.append((scheduled_t, str(ap)))
                                audio_manifest.append(
                                    {
                                        **{**it, "commentary_text": final_txt, "text": final_txt},
                                        **r,
                                        "scheduled_t": float(scheduled_t),
                                        "scheduled_timecode": _timecode_mmss(float(scheduled_t)),
                                        "audio_duration_sec": float(dur),
                                        "window_fit_ok": bool((float(scheduled_t) + float(dur)) <= (float(window_end) + 1e-6)),
                                    }
                                )
                            else:
                                audio_manifest.append({**it, **r})

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

                        mixed_path = str(Path(out_dir) / f"product_{run_id}_commentary.mp4")
                        mixed = _mix_commentary_audio_into_video(
                            base_video_path=segment_path,
                            out_path=mixed_path,
                            clips=clips,
                        )
                        if mixed:
                            commentary_video_path = mixed
                            product_video_path = mixed
                    except Exception:
                        pass
    except Exception as e:
        commentary_error = str(e)

    stage_timings_sec["total"] = round(sum(float(v) for v in stage_timings_sec.values()), 3)

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
        },
        **({"action_spotting": action_spotting_metadata} if action_spotting_metadata else {}),
        **({"commentary_error": commentary_error} if commentary_error else {}),
        **({"qwen_vl_container_events": qwen_vl_container_events} if qwen_vl_container_events else {}),
        **({"commentary_gpu_cleanup": commentary_gpu_cleanup} if commentary_gpu_cleanup else {}),
        **({"track_id_remap": track_id_remap} if track_id_remap else {}),
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
        "events_json_path": events_json_path,
        "event_count": len(events),
    }
