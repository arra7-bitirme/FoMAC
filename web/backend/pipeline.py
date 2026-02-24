from __future__ import annotations

import csv
import json
import os
import subprocess
import sys
import time
import re
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import cv2
import numpy as np
from tqdm import tqdm


@dataclass
class FullPipelineConfig:
    # segment
    start_seconds: float = 0.0
    duration_seconds: Optional[float] = None

    # tracking
    run_tracking: bool = True
    tracking_device: Optional[str] = None
    tracking_config_path: Optional[str] = None
    detector_weights: Optional[str] = None
    reid_weights: Optional[str] = None

    # action spotting
    run_action_spotting: bool = True
    features_path: Optional[str] = None
    checkpoint_path: Optional[str] = None
    action_threshold: float = 0.50
    action_nms_window_sec: float = 10.0

    # overlay
    overlay_event_window_sec: float = 3.0

    # heuristics
    possession_dist_norm: float = 0.08
    possession_stable_frames: int = 6
    possession_stride_frames: int = 5
    ball_cls_id: int = 1
    player_cls_id: int = 0


def _repo_root() -> Path:
    # web/backend/pipeline.py -> parents: backend(0), web(1), repo(2)
    return Path(__file__).resolve().parents[2]


def _default_action_checkpoint() -> Optional[str]:
    p = _repo_root() / "model-training" / "action_spotting" / "spotting_v2" / "checkpoints" / "v3_cnn_0211_2000_best_map.pth"
    return str(p) if p.exists() else None


def _default_tracking_config() -> Optional[str]:
    p = _repo_root() / "model-training" / "tracking-reid-osnet" / "config.yaml"
    return str(p) if p.exists() else None


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
    }
    return mapping.get(label, f"Olay: {label}")


def extract_segment_to_mp4(*, src_path: str, out_path: str, start_sec: float, duration_sec: Optional[float]) -> str:
    """Extract/re-encode a segment using ffmpeg if available, otherwise OpenCV."""
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
    # avoid importing shutil in hot path; simple PATH scan
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
    """Return a usable ffmpeg executable path.

    Tries system PATH first, then falls back to `imageio-ffmpeg` (which can
    provide an ffmpeg binary on Windows without a global install).
    """
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


def run_tracking_reid_osnet(
    *,
    video_path: str,
    out_dir: str,
    device: Optional[str],
    config_path: Optional[str],
    detector_weights: Optional[str],
    reid_weights: Optional[str],
    progress_cb: Optional[Callable[[str, int, int, str], None]] = None,
) -> Dict[str, str]:
    repo = _repo_root()
    script = repo / "model-training" / "tracking-reid-osnet" / "run_botsort_team_reid.py"
    if not script.exists():
        raise FileNotFoundError(str(script))

    run_id = datetime.utcnow().strftime("%Y%m%d%H%M%S") + f"_{int(time.time()*1000)%100000:05d}"

    config_path = config_path or _default_tracking_config()
    if not config_path:
        raise FileNotFoundError("tracking config.yaml not found")

    # If OSNet is enabled in config but weights are missing, create a temp config disabling OSNet.
    # The tracking script does not expose osnet.enabled via CLI overrides.
    try:
        import yaml  # type: ignore

        cfg_obj = None
        with open(config_path, "r", encoding="utf-8") as f:
            cfg_obj = yaml.safe_load(f) or {}

        osnet = cfg_obj.get("osnet") if isinstance(cfg_obj, dict) else None
        if isinstance(osnet, dict) and bool(osnet.get("enabled", False)):
            w_rel = str(osnet.get("weights", "") or "").strip()
            # resolve weights relative to config file
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
        # Best-effort; if yaml not available or parsing fails, we'll just run with given config.
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

    if device:
        cmd += ["--device", device]
    if detector_weights:
        cmd += ["--detector_weights", detector_weights]
    if reid_weights:
        cmd += ["--reid_weights", reid_weights]

    start_ts = time.time()
    with open(log_path, "w", encoding="utf-8", errors="ignore") as lf:
        env = os.environ.copy()
        env.setdefault("PYTHONUNBUFFERED", "1")
        p = subprocess.Popen(cmd, stdout=lf, stderr=subprocess.STDOUT, text=True, env=env)

        # Prefer progress in frames/FPS (parsed from the tracking script logs).
        # Fallback: elapsed-seconds ticker (but without misleading rate info).
        progress_re = re.compile(
            r"\[(?P<cur>\d+)\/(?P<total>\d+)\]\s+(?P<pct>[0-9.]+)%\s+\|\s+(?P<fps>[0-9.]+)\s+FPS\s+\|\s+ETA\s+(?P<eta>\d\d:\d\d:\d\d)"
        )

        last_frame = 0
        inferred_total: Optional[int] = None
        pbar: Optional[tqdm] = None
        tick_pbar = tqdm(total=None, desc="Tracking", unit="s", bar_format="{desc}: {elapsed}")

        try:
            # Open the log for tailing.
            try:
                rf = open(log_path, "r", encoding="utf-8", errors="ignore")
            except Exception:
                rf = None

            while True:
                rc = p.poll()
                any_update = False

                # Tail log and parse progress lines.
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

                # If we didn't get a parsed progress line, still tick elapsed time.
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

    if p.returncode != 0:
        tail = ""
        try:
            with open(log_path, "r", encoding="utf-8", errors="ignore") as rf:
                tail = rf.read()[-4000:]
        except Exception:
            tail = ""
        raise RuntimeError(
            "tracking-reid-osnet failed\n"
            f"cmd: {' '.join(cmd)}\n"
            f"log: {log_path}\n"
            f"tail: {tail}"
        )

    if not os.path.isfile(save_video):
        raise RuntimeError(f"Tracking video not produced: {save_video}\nlog: {log_path}")
    if not os.path.isfile(save_txt):
        raise RuntimeError(f"Tracking CSV not produced: {save_txt}\nlog: {log_path}")

    return {"tracking_video_path": save_video, "tracks_csv_path": save_txt}


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

    # normalize output
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

    # iterate frames in order, with stride
    frame_ids = sorted(by_frame.keys())
    owners: List[Tuple[float, Optional[int]]] = []

    stride = max(1, int(cfg.possession_stride_frames))
    ids = frame_ids[::stride]
    for fid in tqdm(ids, total=len(ids) if ids else None, desc="Possession", unit="frame"):
        rows = by_frame.get(fid, [])

        # select ball
        ball_rows = [r for r in rows if int(float(r.get("cls_id", -1))) == int(cfg.ball_cls_id)]
        if not ball_rows:
            owners.append((fid / fps, None))
            continue

        ball_row = max(ball_rows, key=lambda r: float(r.get("conf", 0.0)))
        bx, by = _xyxy_center(ball_row)

        # players
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

    # stable segments -> events
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
                # emit immediately on stable confirmation
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

    # build event display intervals
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
        pbar.close()
    except Exception:
        pass

    # Ensure browser playback: re-encode to H.264 (yuv420p) and move moov atom to start.
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
            # If remux fails, fall back to the original file.
            pass

    # Fallback: no ffmpeg available or remux failed.
    try:
        if os.path.isfile(out_path):
            os.remove(out_path)
    except Exception:
        pass
    try:
        os.replace(tmp_path, out_path)
    except Exception:
        # last resort
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

    def emit(stage: str, cur: int, total: int, msg: str) -> None:
        if progress_cb is None:
            return
        try:
            progress_cb(stage, int(cur), int(total), str(msg))
        except Exception:
            pass

    segment_path = str(Path(out_dir) / f"segment_{run_id}.mp4")
    emit("segment", 0, 1, "Video segment hazırlanıyor")
    extract_segment_to_mp4(
        src_path=video_path,
        out_path=segment_path,
        start_sec=float(cfg.start_seconds or 0.0),
        duration_sec=cfg.duration_seconds,
    )
    emit("segment", 1, 1, "Video segment hazır")

    tracking_video_path: Optional[str] = None
    tracks_csv_path: Optional[str] = None

    if cfg.run_tracking:
        emit("tracking", 0, 1, "Tracking başlıyor")
        tracking_res = run_tracking_reid_osnet(
            video_path=segment_path,
            out_dir=out_dir,
            device=cfg.tracking_device,
            config_path=cfg.tracking_config_path,
            detector_weights=cfg.detector_weights,
            reid_weights=cfg.reid_weights,
            progress_cb=progress_cb,
        )
        tracking_video_path = tracking_res["tracking_video_path"]
        tracks_csv_path = tracking_res["tracks_csv_path"]
        emit("tracking", 1, 1, "Tracking tamam")

    events: List[Dict[str, Any]] = []

    # Derive possession events from tracking CSV (if available)
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
            # possession is best-effort
            pass

    # Action spotting events
    if cfg.run_action_spotting:
        emit("action_spotting", 0, 1, "Action spotting başlıyor")
        if not cfg.features_path:
            # Auto-extract PCA512 features (SoccerNet sn-spotting style) when not provided.
            # This produces a single .npy with shape (T, 512) at ~2 FPS.
            try:
                from pcaextractor import extract_resnet_tf2_pca512  # type: ignore
            except Exception as e:
                raise RuntimeError(f"Failed to import pcaextractor: {e}")

            auto_feat_path = str(Path(out_dir) / f"features_{run_id}_ResNET_TF2_PCA512.npy")
            cfg.features_path = extract_resnet_tf2_pca512(
                video_path=segment_path,
                out_features_npy=auto_feat_path,
                fps=2.0,
                transform="crop",
                overwrite=False,
                progress_cb=progress_cb,
            )
        emit("action_spotting", 0, 1, "Model inference çalışıyor")
        events.extend(
            run_action_spotting_spotting_v2(
                features_path=cfg.features_path,
                checkpoint_path=cfg.checkpoint_path,
                threshold=float(cfg.action_threshold),
                nms_window_sec=float(cfg.action_nms_window_sec),
            )
        )
        emit("action_spotting", 1, 1, "Action spotting tamam")

    # Sort events by time
    events.sort(key=lambda e: float(e.get("t", 0.0)))

    # The user-facing overlay should focus on action spotting events.
    overlay_events = [e for e in events if str(e.get("source")) == "action_spotting"]
    if not overlay_events:
        overlay_events = events

    base_video = tracking_video_path or segment_path
    final_overlay_path = str(Path(out_dir) / f"overlay_{run_id}.mp4")
    emit("overlay", 0, 1, "Overlay başlıyor")
    overlay_events_on_video(
        base_video_path=base_video,
        out_path=final_overlay_path,
        events=overlay_events,
        window_sec=float(cfg.overlay_event_window_sec),
        progress_cb=progress_cb,
    )
    emit("overlay", 1, 1, "Overlay tamam")

    events_json_path = str(Path(out_dir) / f"events_{run_id}.json")
    payload = {
        "schema_version": "1.0",
        "created_utc": datetime.utcnow().isoformat() + "Z",
        "source": {
            "input_video_path": str(Path(video_path).resolve()),
            "segment_video_path": segment_path,
        },
        "artifacts": {
            "tracking_video_path": tracking_video_path,
            "tracks_csv_path": tracks_csv_path,
            "overlay_video_path": final_overlay_path,
        },
        "events": events,
    }
    with open(events_json_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)

    emit("done", 1, 1, "Pipeline tamam")

    return {
        "run_id": run_id,
        "segment_path": segment_path,
        "tracking_video_path": tracking_video_path,
        "tracks_csv_path": tracks_csv_path,
        "overlay_video_path": final_overlay_path,
        "events_json_path": events_json_path,
        "event_count": len(events),
    }
