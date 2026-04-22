import argparse
import json
import math
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np


THIS_DIR = Path(__file__).resolve().parent
# Ensure local calibration modules are importable when invoked from repo root.
sys.path.insert(0, str(THIS_DIR))


def _ensure_trt_weights(weights_path: str, device: str, half: bool = True) -> str:
    """Export YOLO .pt → TensorRT .engine on first run; subsequent runs load the cached .engine."""
    p = Path(weights_path)
    if p.suffix == ".engine" or not device.startswith("cuda"):
        return weights_path
    engine_path = p.with_suffix(".engine")
    if engine_path.exists():
        return str(engine_path)
    try:
        from ultralytics import YOLO as _YOLO
        print(f"[TRT] Exporting {p.name} → TensorRT (half={half}) …", flush=True)
        model = _YOLO(str(p))
        model.export(format="engine", half=half, device=device, verbose=False)
        if engine_path.exists():
            print(f"[TRT] Export done → {engine_path.name}", flush=True)
            return str(engine_path)
    except Exception as _e:
        print(f"[TRT] Export failed, falling back to .pt: {_e}", flush=True)
    return weights_path


class _HRNetTRTRunner:
    """TRT 10.x inference wrapper — drop-in replacement for an HRNet nn.Module."""

    def __init__(self, engine_path: str, device: Any) -> None:
        import tensorrt as trt
        import torch

        logger = trt.Logger(trt.Logger.WARNING)
        runtime = trt.Runtime(logger)
        with open(engine_path, "rb") as f:
            engine = runtime.deserialize_cuda_engine(f.read())
        self._context = engine.create_execution_context()
        self._device = device

        _dtype_map: dict = {}
        try:
            _dtype_map = {
                trt.DataType.FLOAT: torch.float32,
                trt.DataType.HALF: torch.float16,
                trt.DataType.INT32: torch.int32,
            }
        except AttributeError:
            _dtype_map = {
                trt.float32: torch.float32,
                trt.float16: torch.float16,
                trt.int32: torch.int32,
            }

        self._in_name: str = ""
        self._in_buf: Any = None
        self._out_name: str = ""
        self._out_buf: Any = None

        for i in range(engine.num_io_tensors):
            name = engine.get_tensor_name(i)
            shape = tuple(engine.get_tensor_shape(name))
            dtype = engine.get_tensor_dtype(name)
            tdtype = _dtype_map.get(dtype, torch.float32)
            buf = torch.zeros(*shape, dtype=tdtype, device=device)
            if engine.get_tensor_mode(name) == trt.TensorIOMode.INPUT:
                self._in_name, self._in_buf = name, buf
            else:
                self._out_name, self._out_buf = name, buf

    def __call__(self, x: Any) -> Any:
        import torch
        self._in_buf.copy_(x.to(dtype=self._in_buf.dtype))
        self._context.set_tensor_address(self._in_name, self._in_buf.data_ptr())
        self._context.set_tensor_address(self._out_name, self._out_buf.data_ptr())
        stream = torch.cuda.current_stream(self._device)
        self._context.execute_async_v3(stream_handle=stream.cuda_stream)
        stream.synchronize()
        return self._out_buf.float()

    def eval(self) -> "_HRNetTRTRunner":
        return self


def _try_hrnet_trt(model: Any, weights_path: str, dev: Any,
                   input_shape: tuple = (1, 3, 540, 960)) -> Any:
    """ONNX-export HRNet then build a TRT 10.x FP16 engine. Returns _HRNetTRTRunner or None."""
    import torch
    if getattr(dev, "type", "") != "cuda":
        return None
    p = Path(weights_path)
    onnx_path = p.with_suffix(".onnx")
    engine_path = p.with_suffix(".trt.engine")

    if not onnx_path.exists():
        print(f"[TRT] {p.name} → ONNX export …", flush=True)
        try:
            dummy = torch.zeros(*input_shape, device=dev)
            with torch.no_grad():
                torch.onnx.export(
                    model, dummy, str(onnx_path),
                    opset_version=16,
                    input_names=["input"],
                    output_names=["output"],
                )
            print(f"[TRT] ONNX saved → {onnx_path.name}", flush=True)
        except Exception as _e:
            print(f"[TRT] ONNX export failed: {_e}", flush=True)
            return None

    if not engine_path.exists():
        print(f"[TRT] Building FP16 engine from {onnx_path.name} …", flush=True)
        try:
            import tensorrt as trt
            logger = trt.Logger(trt.Logger.WARNING)
            builder = trt.Builder(logger)
            network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
            parser = trt.OnnxParser(network, logger)
            config = builder.create_builder_config()
            if builder.platform_has_fast_fp16:
                config.set_flag(trt.BuilderFlag.FP16)
            with open(str(onnx_path), "rb") as f:
                if not parser.parse(f.read()):
                    errs = [str(parser.get_error(i)) for i in range(parser.num_errors)]
                    print(f"[TRT] ONNX parse errors: {errs}", flush=True)
                    return None
            serialized = builder.build_serialized_network(network, config)
            if serialized is None:
                print("[TRT] Engine build returned None", flush=True)
                return None
            with open(str(engine_path), "wb") as f:
                f.write(serialized)
            print(f"[TRT] Engine built → {engine_path.name}", flush=True)
        except Exception as _e:
            print(f"[TRT] TRT engine build failed: {_e}", flush=True)
            return None

    try:
        runner = _HRNetTRTRunner(str(engine_path), dev)
        print(f"[TRT] HRNet TRT runner ready ({engine_path.name})", flush=True)
        return runner
    except Exception as _e:
        print(f"[TRT] TRT runner init failed: {_e}", flush=True)
        return None


@dataclass
class CalibrationOutputs:
    map_video_path: str
    events_json_path: str
    frames_jsonl_path: str


def _safe_float(x: Any, default: float = 0.0) -> float:
    try:
        return float(x)
    except Exception:
        return float(default)


def _safe_int(x: Any, default: int = 0) -> int:
    try:
        return int(x)
    except Exception:
        return int(default)


def _load_hrnet_models(*, device: str, kp_weights: str, line_weights: str):
    import torch
    import torchvision.transforms as T

    from nbjw_calib.model.cls_hrnet import get_cls_net
    from nbjw_calib.model.cls_hrnet_l import get_cls_net as get_cls_net_l

    # Minimal configs matching demo.py
    cfg_kp = {
        "MODEL": {
            "IMAGE_SIZE": [960, 540],
            "NUM_JOINTS": 58,
            "PRETRAIN": "",
            "EXTRA": {
                "FINAL_CONV_KERNEL": 1,
                "STAGE1": {"NUM_MODULES": 1, "NUM_BRANCHES": 1, "BLOCK": "BOTTLENECK", "NUM_BLOCKS": [4], "NUM_CHANNELS": [64], "FUSE_METHOD": "SUM"},
                "STAGE2": {"NUM_MODULES": 1, "NUM_BRANCHES": 2, "BLOCK": "BASIC", "NUM_BLOCKS": [4, 4], "NUM_CHANNELS": [48, 96], "FUSE_METHOD": "SUM"},
                "STAGE3": {"NUM_MODULES": 4, "NUM_BRANCHES": 3, "BLOCK": "BASIC", "NUM_BLOCKS": [4, 4, 4], "NUM_CHANNELS": [48, 96, 192], "FUSE_METHOD": "SUM"},
                "STAGE4": {"NUM_MODULES": 3, "NUM_BRANCHES": 4, "BLOCK": "BASIC", "NUM_BLOCKS": [4, 4, 4, 4], "NUM_CHANNELS": [48, 96, 192, 384], "FUSE_METHOD": "SUM"},
            },
        }
    }
    cfg_lines = {
        "MODEL": {
            "IMAGE_SIZE": [960, 540],
            "NUM_JOINTS": 24,
            "PRETRAIN": "",
            "EXTRA": {
                "FINAL_CONV_KERNEL": 1,
                "STAGE1": {"NUM_MODULES": 1, "NUM_BRANCHES": 1, "BLOCK": "BOTTLENECK", "NUM_BLOCKS": [4], "NUM_CHANNELS": [64], "FUSE_METHOD": "SUM"},
                "STAGE2": {"NUM_MODULES": 1, "NUM_BRANCHES": 2, "BLOCK": "BASIC", "NUM_BLOCKS": [4, 4], "NUM_CHANNELS": [48, 96], "FUSE_METHOD": "SUM"},
                "STAGE3": {"NUM_MODULES": 4, "NUM_BRANCHES": 3, "BLOCK": "BASIC", "NUM_BLOCKS": [4, 4, 4], "NUM_CHANNELS": [48, 96, 192], "FUSE_METHOD": "SUM"},
                "STAGE4": {"NUM_MODULES": 3, "NUM_BRANCHES": 4, "BLOCK": "BASIC", "NUM_BLOCKS": [4, 4, 4, 4], "NUM_CHANNELS": [48, 96, 192, 384], "FUSE_METHOD": "SUM"},
            },
        }
    }

    dev = torch.device(device)

    model_kp = get_cls_net(cfg_kp)
    sd = torch.load(kp_weights, map_location=dev)
    if isinstance(sd, dict) and any(k.startswith("module.") for k in sd.keys()):
        sd = {k.replace("module.", ""): v for k, v in sd.items()}
    model_kp.load_state_dict(sd)
    model_kp.to(dev)
    model_kp.eval()

    model_lines = get_cls_net_l(cfg_lines)
    sd2 = torch.load(line_weights, map_location=dev)
    if isinstance(sd2, dict) and any(k.startswith("module.") for k in sd2.keys()):
        sd2 = {k.replace("module.", ""): v for k, v in sd2.items()}
    model_lines.load_state_dict(sd2)
    model_lines.to(dev)
    model_lines.eval()

    tfms_resize = T.Compose([T.Resize((540, 960)), T.ToTensor()])

    return model_kp, model_lines, tfms_resize, dev


def _pitch_background(*, scale: int = 8, margin: int = 50) -> np.ndarray:
    """Draw a football pitch background using OpenCV.

    Geometry matches utils_field.py (nbjw_calib) — all coordinates in metres,
    SoccerPitch convention: origin at centre, x: -52.5..+52.5, y: -34..+34.
    """
    pitch_length = 105   # metres
    pitch_width  = 68
    img_width  = int(pitch_length * scale + 2 * margin)
    img_height = int(pitch_width  * scale + 2 * margin)

    # Dark-green grass
    bg = np.ones((img_height, img_width, 3), dtype=np.uint8) * 50
    bg[:, :, 1] = 110

    lc = (210, 210, 210)   # line colour (light grey in BGR)
    lw = 2

    def w2p(wx: float, wy: float) -> Tuple[int, int]:
        """World metres → pixel (SoccerPitch centred convention)."""
        px = int((wx + pitch_length / 2.0) * scale + margin)
        py = int((wy + pitch_width  / 2.0) * scale + margin)
        return px, py

    # ── Boundary ──────────────────────────────────────────────────────────────
    cv2.line(bg, w2p(-52.5, -34), w2p(-52.5,  34), lc, lw)  # left
    cv2.line(bg, w2p( 52.5, -34), w2p( 52.5,  34), lc, lw)  # right
    cv2.line(bg, w2p(-52.5, -34), w2p( 52.5, -34), lc, lw)  # top
    cv2.line(bg, w2p(-52.5,  34), w2p( 52.5,  34), lc, lw)  # bottom

    # ── Centre line ───────────────────────────────────────────────────────────
    cv2.line(bg, w2p(0, -34), w2p(0, 34), lc, lw)

    # ── Left penalty area (16.5 m deep, 40.3 m wide) ─────────────────────────
    cv2.line(bg, w2p(-52.5, -20.15), w2p(-36.0, -20.15), lc, lw)
    cv2.line(bg, w2p(-52.5,  20.15), w2p(-36.0,  20.15), lc, lw)
    cv2.line(bg, w2p(-36.0, -20.15), w2p(-36.0,  20.15), lc, lw)

    # ── Left goal area (5.5 m deep, 18.3 m wide) ─────────────────────────────
    cv2.line(bg, w2p(-52.5, -9.15), w2p(-47.0, -9.15), lc, lw)
    cv2.line(bg, w2p(-52.5,  9.15), w2p(-47.0,  9.15), lc, lw)
    cv2.line(bg, w2p(-47.0, -9.15), w2p(-47.0,  9.15), lc, lw)

    # ── Right penalty area ────────────────────────────────────────────────────
    cv2.line(bg, w2p(52.5, -20.15), w2p(36.0, -20.15), lc, lw)
    cv2.line(bg, w2p(52.5,  20.15), w2p(36.0,  20.15), lc, lw)
    cv2.line(bg, w2p(36.0, -20.15), w2p(36.0,  20.15), lc, lw)

    # ── Right goal area ───────────────────────────────────────────────────────
    cv2.line(bg, w2p(52.5, -9.15), w2p(47.0, -9.15), lc, lw)
    cv2.line(bg, w2p(52.5,  9.15), w2p(47.0,  9.15), lc, lw)
    cv2.line(bg, w2p(47.0, -9.15), w2p(47.0,  9.15), lc, lw)

    # ── Centre circle (r = 9.15 m) + spot ────────────────────────────────────
    r_px = int(9.15 * scale)
    cv2.circle(bg, w2p(0.0, 0.0), r_px, lc, lw)
    cv2.circle(bg, w2p(0.0, 0.0), 3,    lc, -1)

    # ── Left penalty spot (11 m from goal line = -41.5 from centre) ──────────
    cv2.circle(bg, w2p(-41.5, 0.0), 3, lc, -1)
    # Penalty arc: portion outside left penalty area (penalty box edge at x=-36)
    # Distance from spot to box edge = 41.5-36 = 5.5 m; arc half-angle ≈ 53°
    cv2.ellipse(bg, w2p(-41.5, 0.0), (r_px, r_px), 0, -53, 53, lc, lw)

    # ── Right penalty spot (94 m from left = +41.5 from centre) ──────────────
    cv2.circle(bg, w2p(41.5, 0.0), 3, lc, -1)
    cv2.ellipse(bg, w2p(41.5, 0.0), (r_px, r_px), 0, 127, 233, lc, lw)

    # ── Try SoccerPitch for extra accuracy (optional, non-fatal) ─────────────
    try:
        from sn_calibration_baseline.soccerpitch import SoccerPitch
        field = SoccerPitch()

        def _w2p_sp(pt3d: np.ndarray) -> Tuple[int, int]:
            return w2p(float(pt3d[0]), float(pt3d[1]))

        for line in field.sample_field_points():
            pts = [_w2p_sp(np.asarray(p)) for p in line]
            cv2.polylines(bg, [np.array(pts, dtype=np.int32)], False, (230, 230, 230), 1)
    except Exception:
        pass

    return bg


def _draw_map_frame(
    *,
    background: np.ndarray,
    camera: Any,
    persons: List[Tuple[np.ndarray, int, int]],
    balls: List[np.ndarray],
    scale: int = 8,
    margin: int = 50,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """Return (frame_bgr, meta) where meta contains projected world coords."""
    frame = background.copy()

    pitch_length = 105
    pitch_width = 68

    def world_to_minimap(pt3d: np.ndarray) -> Tuple[int, int]:
        mx = int((float(pt3d[0]) + pitch_length / 2) * scale + margin)
        my = int((float(pt3d[1]) + pitch_width / 2) * scale + margin)
        return mx, my

    players_out: List[Dict[str, Any]] = []

    # Players
    for bbox_xyxy, team_id, track_id in persons:
        try:
            x1, y1, x2, y2 = [int(v) for v in bbox_xyxy.tolist()]
        except Exception:
            continue
        foot_x = (x1 + x2) / 2.0
        foot_y = float(y2)
        try:
            ground_pt = camera.unproject_point_on_planeZ0(np.array([foot_x, foot_y], dtype=np.float32))
            ground_pt = np.asarray(ground_pt).reshape(-1)
        except Exception:
            continue

        # Sanity check: keep roughly inside pitch bounds
        if not (abs(float(ground_pt[0])) < 120 and abs(float(ground_pt[1])) < 90):
            continue

        mx, my = world_to_minimap(ground_pt)
        # Dot color by team
        dot_c = (200, 200, 200)
        if int(team_id) == 0:
            dot_c = (255, 100, 100)
        elif int(team_id) == 1:
            dot_c = (100, 100, 255)

        cv2.circle(frame, (mx, my), 6, (255, 255, 255), -1)
        cv2.circle(frame, (mx, my), 4, dot_c, -1)

        players_out.append(
            {
                "track_id": int(track_id),
                "team_id": int(team_id),
                "bbox_xyxy": [int(x1), int(y1), int(x2), int(y2)],
                "world_xy": [float(ground_pt[0]), float(ground_pt[1])],
            }
        )

    # Ball (use first ball)
    ball_out: Optional[Dict[str, Any]] = None
    if balls:
        bbox = balls[0]
        try:
            x1, y1, x2, y2 = [float(v) for v in bbox.tolist()]
            cx = (x1 + x2) / 2.0
            cy = (y1 + y2) / 2.0 + (y2 - y1) * 0.4
            ground_pt = camera.unproject_point_on_planeZ0(np.array([cx, cy], dtype=np.float32))
            ground_pt = np.asarray(ground_pt).reshape(-1)
            if abs(float(ground_pt[0])) < 120 and abs(float(ground_pt[1])) < 90:
                mx, my = world_to_minimap(ground_pt)
                cv2.circle(frame, (mx, my), 6, (0, 0, 0), -1)
                cv2.circle(frame, (mx, my), 4, (0, 165, 255), -1)
                ball_out = {
                    "bbox_xyxy": [float(x1), float(y1), float(x2), float(y2)],
                    "world_xy": [float(ground_pt[0]), float(ground_pt[1])],
                }
        except Exception:
            ball_out = None

    return frame, {"players": players_out, "ball": ball_out}


def _fuse_meta_frames(
    meta_frames: List[Dict[str, Any]],
    *,
    player_merge_dist_m: float = 1.5,
    ball_merge_dist_m: float = 2.5,
) -> Dict[str, Any]:
    clusters: List[Dict[str, Any]] = []

    for meta in meta_frames:
        for player in meta.get("players") or []:
            try:
                world_xy = np.asarray(player.get("world_xy"), dtype=np.float32)
                team_id = int(player.get("team_id", -1))
            except Exception:
                continue

            matched_cluster = None
            best_dist = 1e9
            for cluster in clusters:
                if int(cluster["team_id"]) != team_id:
                    continue
                dist = float(np.linalg.norm(world_xy - cluster["world_xy"]))
                if dist <= float(player_merge_dist_m) and dist < best_dist:
                    best_dist = dist
                    matched_cluster = cluster

            if matched_cluster is None:
                clusters.append(
                    {
                        "team_id": team_id,
                        "track_id": int(player.get("track_id", -1)),
                        "world_xy": world_xy,
                        "bbox_xyxy": np.asarray(player.get("bbox_xyxy") or [0, 0, 0, 0], dtype=np.float32),
                        "count": 1,
                    }
                )
            else:
                count = int(matched_cluster["count"]) + 1
                matched_cluster["world_xy"] = (matched_cluster["world_xy"] * matched_cluster["count"] + world_xy) / float(count)
                matched_cluster["bbox_xyxy"] = (
                    matched_cluster["bbox_xyxy"] * matched_cluster["count"]
                    + np.asarray(player.get("bbox_xyxy") or [0, 0, 0, 0], dtype=np.float32)
                ) / float(count)
                matched_cluster["count"] = count
                if int(matched_cluster.get("track_id", -1)) == -1 and int(player.get("track_id", -1)) != -1:
                    matched_cluster["track_id"] = int(player.get("track_id", -1))

    fused_players: List[Dict[str, Any]] = []
    for cluster in clusters:
        fused_players.append(
            {
                "track_id": int(cluster.get("track_id", -1)),
                "team_id": int(cluster.get("team_id", -1)),
                "bbox_xyxy": [int(round(v)) for v in cluster["bbox_xyxy"].tolist()],
                "world_xy": [float(cluster["world_xy"][0]), float(cluster["world_xy"][1])],
            }
        )

    ball_candidates: List[np.ndarray] = []
    ball_bboxes: List[np.ndarray] = []
    for meta in meta_frames:
        ball = meta.get("ball")
        if not isinstance(ball, dict):
            continue
        try:
            ball_candidates.append(np.asarray(ball.get("world_xy"), dtype=np.float32))
            ball_bboxes.append(np.asarray(ball.get("bbox_xyxy") or [0.0, 0.0, 0.0, 0.0], dtype=np.float32))
        except Exception:
            continue

    fused_ball: Optional[Dict[str, Any]] = None
    if ball_candidates:
        anchor = ball_candidates[0]
        close_points = []
        close_bboxes = []
        for point, bbox in zip(ball_candidates, ball_bboxes):
            if float(np.linalg.norm(point - anchor)) <= float(ball_merge_dist_m):
                close_points.append(point)
                close_bboxes.append(bbox)
        if not close_points:
            close_points = ball_candidates
            close_bboxes = ball_bboxes
        fused_ball = {
            "bbox_xyxy": np.mean(np.stack(close_bboxes, axis=0), axis=0).astype(float).tolist(),
            "world_xy": np.mean(np.stack(close_points, axis=0), axis=0).astype(float).tolist(),
        }

    return {"players": fused_players, "ball": fused_ball}


def _parse_yolo_boxes(result: Any, conf_thres: float) -> Tuple[List[np.ndarray], List[np.ndarray]]:
    person_dets: List[np.ndarray] = []
    ball_boxes: List[np.ndarray] = []

    if result is None or getattr(result, "boxes", None) is None:
        return person_dets, ball_boxes

    try:
        boxes = result.boxes
        xyxy = boxes.xyxy.detach().cpu().numpy() if getattr(boxes, "xyxy", None) is not None else np.zeros((0, 4), dtype=np.float32)
        confs = boxes.conf.detach().cpu().numpy() if getattr(boxes, "conf", None) is not None else np.zeros((xyxy.shape[0],), dtype=np.float32)
        clss = boxes.cls.detach().cpu().numpy() if getattr(boxes, "cls", None) is not None else np.zeros((xyxy.shape[0],), dtype=np.float32)
    except Exception:
        return person_dets, ball_boxes

    for i in range(int(xyxy.shape[0])):
        try:
            cls_id = int(clss[i])
            conf = float(confs[i])
            if conf < float(conf_thres):
                continue
            x1, y1, x2, y2 = [float(v) for v in xyxy[i].tolist()]
            if cls_id == 0:
                person_dets.append(np.array([x1, y1, x2, y2, conf, float(cls_id)], dtype=np.float32))
            elif cls_id == 1:
                ball_boxes.append(np.array([x1, y1, x2, y2], dtype=np.float32))
        except Exception:
            continue

    return person_dets, ball_boxes


def _render_meta_frame(
    *,
    background: np.ndarray,
    meta_frame: Dict[str, Any],
    scale: int = 8,
    margin: int = 50,
) -> np.ndarray:
    frame = background.copy()

    pitch_length = 105
    pitch_width = 68

    def world_to_minimap(pt2d: List[float]) -> Tuple[int, int]:
        mx = int((float(pt2d[0]) + pitch_length / 2) * scale + margin)
        my = int((float(pt2d[1]) + pitch_width / 2) * scale + margin)
        return mx, my

    for p in meta_frame.get("players") or []:
        try:
            world_xy = p.get("world_xy")
            if world_xy is None:
                continue
            mx, my = world_to_minimap(world_xy)
            team_id = int(p.get("team_id", -1))
            dot_c = (200, 200, 200)
            if team_id == 0:
                dot_c = (255, 100, 100)
            elif team_id == 1:
                dot_c = (100, 100, 255)
            cv2.circle(frame, (mx, my), 6, (255, 255, 255), -1)
            cv2.circle(frame, (mx, my), 4, dot_c, -1)
        except Exception:
            continue

    ball = meta_frame.get("ball")
    if isinstance(ball, dict) and ball.get("world_xy") is not None:
        try:
            mx, my = world_to_minimap(ball["world_xy"])
            cv2.circle(frame, (mx, my), 6, (0, 0, 0), -1)
            cv2.circle(frame, (mx, my), 4, (0, 165, 255), -1)
        except Exception:
            pass

    return frame


def _lerp_values(a: List[float], b: List[float], alpha: float) -> List[float]:
    if len(a) != len(b):
        return list(a if alpha < 0.5 else b)
    return [float((1.0 - alpha) * float(av) + alpha * float(bv)) for av, bv in zip(a, b)]


def _apply_interpolation_curve(alpha: float, interpolation_mode: str) -> float:
    alpha = max(0.0, min(1.0, float(alpha)))
    if interpolation_mode == "smoothstep":
        return float(alpha * alpha * (3.0 - 2.0 * alpha))
    return alpha


def _interpolate_meta_frames(
    prev_meta: Dict[str, Any],
    next_meta: Dict[str, Any],
    alpha: float,
    interpolation_mode: str = "linear",
) -> Dict[str, Any]:
    alpha = max(0.0, min(1.0, float(alpha)))
    interp_alpha = _apply_interpolation_curve(alpha, interpolation_mode)

    prev_players = {
        int(p.get("track_id", -1)): p
        for p in (prev_meta.get("players") or [])
        if isinstance(p, dict) and int(p.get("track_id", -1)) != -1
    }
    next_players = {
        int(p.get("track_id", -1)): p
        for p in (next_meta.get("players") or [])
        if isinstance(p, dict) and int(p.get("track_id", -1)) != -1
    }

    players_out: List[Dict[str, Any]] = []
    for track_id in sorted(set(prev_players.keys()) | set(next_players.keys())):
        prev_player = prev_players.get(track_id)
        next_player = next_players.get(track_id)
        if prev_player is not None and next_player is not None:
            prev_xy = prev_player.get("world_xy")
            next_xy = next_player.get("world_xy")
            if prev_xy is None or next_xy is None:
                base_player = prev_player if interp_alpha < 0.5 else next_player
            else:
                base_player = dict(prev_player if interp_alpha < 0.5 else next_player)
                base_player["world_xy"] = _lerp_values(list(prev_xy), list(next_xy), interp_alpha)
                prev_bbox = prev_player.get("bbox_xyxy")
                next_bbox = next_player.get("bbox_xyxy")
                if prev_bbox is not None and next_bbox is not None:
                    base_player["bbox_xyxy"] = [int(round(v)) for v in _lerp_values(list(prev_bbox), list(next_bbox), interp_alpha)]
            players_out.append(base_player)
        elif prev_player is not None and interp_alpha < 0.5:
            players_out.append(dict(prev_player))
        elif next_player is not None and interp_alpha >= 0.5:
            players_out.append(dict(next_player))

    out_ball: Optional[Dict[str, Any]] = None
    prev_ball = prev_meta.get("ball") if isinstance(prev_meta.get("ball"), dict) else None
    next_ball = next_meta.get("ball") if isinstance(next_meta.get("ball"), dict) else None
    if prev_ball is not None and next_ball is not None:
        out_ball = dict(prev_ball if interp_alpha < 0.5 else next_ball)
        prev_xy = prev_ball.get("world_xy")
        next_xy = next_ball.get("world_xy")
        if prev_xy is not None and next_xy is not None:
            out_ball["world_xy"] = _lerp_values(list(prev_xy), list(next_xy), interp_alpha)
        prev_bbox = prev_ball.get("bbox_xyxy")
        next_bbox = next_ball.get("bbox_xyxy")
        if prev_bbox is not None and next_bbox is not None:
            out_ball["bbox_xyxy"] = _lerp_values(list(prev_bbox), list(next_bbox), interp_alpha)
    elif prev_ball is not None and interp_alpha < 0.5:
        out_ball = dict(prev_ball)
    elif next_ball is not None and interp_alpha >= 0.5:
        out_ball = dict(next_ball)

    return {"players": players_out, "ball": out_ball}


def _write_frame_record(
    *,
    vw: Any,
    f_frames: Any,
    frame_bgr: np.ndarray,
    frame_idx: int,
    fps: float,
    frames_stride: int,
    calib_res: bool,
    rep_err: Optional[float],
    meta_frame: Dict[str, Any],
    sampled_frame_idx: int,
    sampled_frame_t: float,
    interpolation_alpha: Optional[float] = None,
    interpolation_from_idx: Optional[int] = None,
    interpolation_to_idx: Optional[int] = None,
) -> None:
    vw.write(frame_bgr)

    if f_frames is not None and (frame_idx % frames_stride == 0):
        rec: Dict[str, Any] = {
            "frame_idx": int(frame_idx),
            "t": float(frame_idx) / float(max(1e-6, fps)),
            "fps": float(fps),
            "calibration_ok": bool(calib_res),
            "rep_err": rep_err,
            "data": meta_frame,
            "sampled_frame_idx": int(sampled_frame_idx),
            "sampled_frame_t": float(sampled_frame_t),
        }
        if interpolation_alpha is not None:
            rec["interpolation_alpha"] = float(interpolation_alpha)
        if interpolation_from_idx is not None:
            rec["interpolation_from_idx"] = int(interpolation_from_idx)
        if interpolation_to_idx is not None:
            rec["interpolation_to_idx"] = int(interpolation_to_idx)
        f_frames.write(json.dumps(rec, ensure_ascii=False) + "\n")


def run(
    *,
    source: str,
    out_map: str,
    out_events: str,
    out_frames: Optional[str],
    detector_weights: str,
    kp_weights: str,
    line_weights: str,
    conf_thres: float = 0.30,
    frames_stride: int = 1,
    progress_every: int = 25,
    max_frames: int = 0,
    yolo_frame_window: int = 8,
    yolo_selection_mode: str = "ball_priority",
    interpolation_mode: str = "linear",
    detection_cache_path: Optional[str] = None,
) -> CalibrationOutputs:
    import torch

    device = "cuda" if torch.cuda.is_available() else "cpu"

    if not os.path.isfile(kp_weights):
        raise FileNotFoundError(f"Missing keypoints weights: {kp_weights}")
    if not os.path.isfile(line_weights):
        raise FileNotFoundError(f"Missing lines weights: {line_weights}")
    if not os.path.isfile(detector_weights):
        raise FileNotFoundError(f"Missing detector weights: {detector_weights}")

    model_kp, model_lines, tfms_resize, dev = _load_hrnet_models(
        device=device, kp_weights=kp_weights, line_weights=line_weights
    )

    from nbjw_calib.utils.utils_heatmap import (
        complete_keypoints,
        coords_to_dict,
        get_keypoints_from_heatmap_batch_maxpool,
        get_keypoints_from_heatmap_batch_maxpool_l,
    )
    from nbjw_calib.utils.utils_calib import FramebyFrameCalib
    from sn_calibration_baseline.camera import Camera

    # Detector
    from ultralytics import YOLO

    det = YOLO(detector_weights)

    # Detection cache (optional — written here, read by tracking subprocess)
    det_cache = None
    if detection_cache_path:
        try:
            import sys as _sys
            import os as _os
            _cache_dir = _os.path.dirname(_os.path.abspath(__file__))
            _backend = _os.path.join(_os.path.dirname(_os.path.dirname(_cache_dir)), "web", "backend")
            if _backend not in _sys.path:
                _sys.path.insert(0, _backend)
            from detection_cache import DetectionCache, boxes_to_cache
            det_cache = DetectionCache(detection_cache_path)
        except Exception:
            det_cache = None

    # Optional tracker/team classifier (best-effort)
    # Prefer boxmot ByteTrack if available; otherwise use Ultralytics built-in tracker via YOLO.track(persist=True).
    tracker = None
    tracking_backend = "none"
    try:
        from boxmot import ByteTrack  # type: ignore

        tracker = ByteTrack(
            reid_weights=None,
            device=("cuda:0" if device == "cuda" else "cpu"),
            half=(device == "cuda"),
            frame_rate=25,
        )
        tracking_backend = "boxmot_bytetrack"
    except Exception:
        tracker = None
        tracking_backend = "ultralytics_track"

    embedder = None
    clusterer = None
    try:
        from team_clasifier import AutoLabEmbedder, AutomaticTeamClusterer

        embedder = AutoLabEmbedder()
        clusterer = AutomaticTeamClusterer()
    except Exception:
        embedder = None
        clusterer = None

    cap = cv2.VideoCapture(str(source))
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video: {source}")

    fps = float(cap.get(cv2.CAP_PROP_FPS) or 25.0)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)

    # Stats for debugging/diagnosis
    stats = {
        "frames": 0,
        "calibration_ok_frames": 0,
        "frames_with_person_det": 0,
        "frames_with_ball_det": 0,
        "projected_player_points": 0,
        "projected_ball_points": 0,
        "events": 0,
        "tracking_backend": tracking_backend,
        "tracker_ok_frames": 0,
        "tracks_total": 0,
        "team_cluster_trained": False,
        "team_cluster_collected": 0,
        "yolo_frame_window": 1,
        "yolo_windows": 0,
        "yolo_sampled_frames": 0,
        "yolo_probe_frames": 0,
        "yolo_selection_mode": "ball_priority",
        "interpolation_mode": "linear",
        "interpolated_output_frames": 0,
    }

    try:
        progress_every = int(progress_every)
    except Exception:
        progress_every = 25
    if progress_every < 1:
        progress_every = 25

    try:
        max_frames = int(max_frames)
    except Exception:
        max_frames = 0
    if max_frames < 0:
        max_frames = 0

    try:
        yolo_frame_window = int(yolo_frame_window)
    except Exception:
        yolo_frame_window = 8
    if yolo_frame_window < 1:
        yolo_frame_window = 1
    stats["yolo_frame_window"] = int(yolo_frame_window)

    yolo_selection_mode = str(yolo_selection_mode or "ball_priority").strip().lower()
    if yolo_selection_mode not in {"ball_priority", "first", "fused_window"}:
        yolo_selection_mode = "ball_priority"
    stats["yolo_selection_mode"] = yolo_selection_mode

    interpolation_mode = str(interpolation_mode or "linear").strip().lower()
    if interpolation_mode not in {"linear", "smoothstep"}:
        interpolation_mode = "linear"
    stats["interpolation_mode"] = interpolation_mode

    run_config = {
        "source": str(Path(source).resolve()),
        "detector_weights": str(Path(detector_weights).resolve()),
        "kp_weights": str(Path(kp_weights).resolve()),
        "line_weights": str(Path(line_weights).resolve()),
        "conf_thres": float(conf_thres),
        "frames_stride": int(frames_stride),
        "progress_every": int(progress_every),
        "max_frames": int(max_frames),
        "yolo_frame_window": int(yolo_frame_window),
        "yolo_selection_mode": str(yolo_selection_mode),
        "interpolation_mode": str(interpolation_mode),
        "write_frames_jsonl": bool(out_frames and str(out_frames).strip()),
    }

    try:
        print("__CONFIG__ " + json.dumps(run_config, ensure_ascii=False), flush=True)
    except Exception:
        pass

    # Prepare map writer (fixed pitch canvas)
    scale = 8
    margin = 50
    bg = _pitch_background(scale=scale, margin=margin)
    map_h, map_w = int(bg.shape[0]), int(bg.shape[1])

    out_map_p = Path(out_map)
    out_map_p.parent.mkdir(parents=True, exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    vw = cv2.VideoWriter(str(out_map_p), fourcc, fps, (map_w, map_h))
    if not vw.isOpened():
        cap.release()
        raise RuntimeError(f"Failed to create map video writer: {out_map}")

    write_frames = bool(out_frames and str(out_frames).strip())
    out_frames_p: Optional[Path] = None
    if write_frames:
        out_frames_p = Path(str(out_frames)).resolve()
        out_frames_p.parent.mkdir(parents=True, exist_ok=True)

    try:
        frames_stride = int(frames_stride)
    except Exception:
        frames_stride = 1
    if frames_stride < 1:
        frames_stride = 1

    # Possession/pass event state
    events: List[Dict[str, Any]] = []
    last_possessor: Optional[Dict[str, Any]] = None
    potential_possessor: Optional[Dict[str, Any]] = None
    potential_frames = 0
    frames_since_loss = 0

    # Track -> team mapping (filled once clustering is trained)
    team_by_track: Dict[int, int] = {}

    frame_idx = 0
    f_frames = None
    pending_sample: Optional[Dict[str, Any]] = None
    try:
        if write_frames and out_frames_p is not None:
            f_frames = open(out_frames_p, "w", encoding="utf-8")

        while True:
            frame_batch: List[Tuple[int, np.ndarray]] = []

            while len(frame_batch) < yolo_frame_window:
                current_frame_idx = frame_idx + len(frame_batch)
                if max_frames > 0 and current_frame_idx >= max_frames:
                    break

                ret, next_frame_bgr = cap.read()
                if not ret or next_frame_bgr is None:
                    break

                stats["frames"] += 1

                if (current_frame_idx % progress_every) == 0:
                    try:
                        msg = "Calibration çalışıyor"
                        if total_frames > 0:
                            pct = int((float(current_frame_idx) * 100.0) / float(total_frames))
                            msg = f"Calibration: {pct}%"
                        print(
                            "__PROGRESS__ "
                            + json.dumps(
                                {
                                    "stage": "calibration",
                                    "current": int(current_frame_idx),
                                    "total": int(total_frames) if total_frames > 0 else 0,
                                    "message": msg,
                                },
                                ensure_ascii=False,
                            ),
                            flush=True,
                        )
                    except Exception:
                        pass

                frame_batch.append((current_frame_idx, next_frame_bgr))

            if not frame_batch:
                break

            selected_batch_idx = len(frame_batch) - 1
            selected_frame_idx, frame_bgr = frame_batch[selected_batch_idx]
            selected_track_result = None
            selected_detect_result = None
            fused_window_detections: List[Tuple[np.ndarray, List[np.ndarray]]] = []

            if yolo_selection_mode == "first":
                selected_batch_idx = 0
                selected_frame_idx, frame_bgr = frame_batch[0]
                if tracking_backend == "ultralytics_track":
                    try:
                        stats["yolo_probe_frames"] += 1
                        selected_track_result = det.track(
                            frame_bgr,
                            verbose=False,
                            persist=True,
                            conf=float(conf_thres),
                            tracker="bytetrack.yaml",
                        )
                    except Exception:
                        selected_track_result = None
                    if det_cache is not None:
                        _r = selected_track_result[0] if selected_track_result else None
                        det_cache.set(selected_frame_idx, boxes_to_cache(_r))
                else:
                    try:
                        stats["yolo_probe_frames"] += 1
                        probe_results = det(frame_bgr, verbose=False, conf=float(conf_thres))
                        selected_detect_result = probe_results[0] if probe_results else None
                    except Exception:
                        selected_detect_result = None
                    if det_cache is not None:
                        det_cache.set(selected_frame_idx, boxes_to_cache(selected_detect_result))
            elif yolo_selection_mode == "fused_window":
                selected_batch_idx = 0
                selected_frame_idx, frame_bgr = frame_batch[0]
                for _fw_idx, candidate_frame_bgr in frame_batch:
                    try:
                        stats["yolo_probe_frames"] += 1
                        probe_results = det(candidate_frame_bgr, verbose=False, conf=float(conf_thres))
                        probe_result = probe_results[0] if probe_results else None
                    except Exception:
                        probe_result = None
                    if det_cache is not None:
                        det_cache.set(_fw_idx, boxes_to_cache(probe_result))
                    candidate_person_dets, candidate_ball_boxes = _parse_yolo_boxes(probe_result, conf_thres)
                    fused_window_detections.append((np.asarray(candidate_person_dets), list(candidate_ball_boxes)))
            else:
                # ball_priority: run YOLO on ALL frames in window to populate cache,
                # but select the first frame where a ball is detected (or last if none found).
                selected_batch_idx = len(frame_batch) - 1
                selected_frame_idx = frame_batch[-1][0]
                frame_bgr = frame_batch[-1][1]
                for batch_idx, (candidate_frame_idx, candidate_frame_bgr) in enumerate(frame_batch):
                    if tracking_backend == "ultralytics_track":
                        try:
                            stats["yolo_probe_frames"] += 1
                            _track_res = det.track(
                                candidate_frame_bgr,
                                verbose=False,
                                persist=True,
                                conf=float(conf_thres),
                                tracker="bytetrack.yaml",
                            )
                        except Exception:
                            _track_res = None

                        probe_result = _track_res[0] if _track_res else None
                        if det_cache is not None:
                            det_cache.set(candidate_frame_idx, boxes_to_cache(probe_result))
                    else:
                        try:
                            stats["yolo_probe_frames"] += 1
                            probe_results = det(candidate_frame_bgr, verbose=False, conf=float(conf_thres))
                            _detect_result = probe_results[0] if probe_results else None
                        except Exception:
                            _detect_result = None

                        probe_result = _detect_result
                        if det_cache is not None:
                            det_cache.set(candidate_frame_idx, boxes_to_cache(probe_result))

                    _, candidate_ball_boxes = _parse_yolo_boxes(probe_result, conf_thres)
                    if candidate_ball_boxes and selected_batch_idx == len(frame_batch) - 1:
                        # First frame with a ball found — use it as selected
                        selected_batch_idx = batch_idx
                        selected_frame_idx = candidate_frame_idx
                        frame_bgr = candidate_frame_bgr
                        if tracking_backend == "ultralytics_track":
                            selected_track_result = _track_res
                        else:
                            selected_detect_result = _detect_result
                    elif batch_idx == len(frame_batch) - 1 and selected_batch_idx == len(frame_batch) - 1:
                        # No ball found; store last frame result as fallback
                        if tracking_backend == "ultralytics_track":
                            selected_track_result = _track_res
                        else:
                            selected_detect_result = _detect_result

            t_sec = float(selected_frame_idx) / float(max(1e-6, fps))

            stats["yolo_windows"] += 1
            stats["yolo_sampled_frames"] += 1

            person_dets: List[np.ndarray] = []
            ball_boxes: List[np.ndarray] = []
            window_meta_frames: List[Dict[str, Any]] = []

            # Detection + tracking
            if tracking_backend == "ultralytics_track":
                tr_res = selected_track_result

                if tr_res and tr_res[0] is not None and getattr(tr_res[0], "boxes", None) is not None:
                    b = tr_res[0].boxes
                    try:
                        xyxy = b.xyxy.detach().cpu().numpy() if getattr(b, "xyxy", None) is not None else np.zeros((0, 4), dtype=np.float32)
                        confs = b.conf.detach().cpu().numpy() if getattr(b, "conf", None) is not None else np.zeros((xyxy.shape[0],), dtype=np.float32)
                        clss = b.cls.detach().cpu().numpy() if getattr(b, "cls", None) is not None else np.zeros((xyxy.shape[0],), dtype=np.float32)
                        ids = None
                        try:
                            ids = b.id.detach().cpu().numpy() if getattr(b, "id", None) is not None else None
                        except Exception:
                            ids = None
                    except Exception:
                        xyxy = np.zeros((0, 4), dtype=np.float32)
                        confs = np.zeros((0,), dtype=np.float32)
                        clss = np.zeros((0,), dtype=np.float32)
                        ids = None

                    # Build YOLO-like det rows for persons so we can reuse the ByteTrack path.
                    for i in range(int(xyxy.shape[0])):
                        try:
                            cls_id = int(clss[i])
                            c = float(confs[i])
                            if c < float(conf_thres):
                                continue
                            x1, y1, x2, y2 = [float(v) for v in xyxy[i].tolist()]
                            if cls_id == 0:
                                tid = -1
                                if ids is not None and i < len(ids):
                                    try:
                                        tid = int(ids[i])
                                    except Exception:
                                        tid = -1
                                person_dets.append(np.array([x1, y1, x2, y2, c, float(cls_id), float(tid)], dtype=np.float32))
                            elif cls_id == 1:
                                ball_boxes.append(np.array([x1, y1, x2, y2], dtype=np.float32))
                        except Exception:
                            continue
            else:
                # Plain YOLO detect (and optional boxmot tracking)
                person_dets, ball_boxes = _parse_yolo_boxes(selected_detect_result, conf_thres)

            if len(person_dets) > 0:
                stats["frames_with_person_det"] += 1
            if len(ball_boxes) > 0:
                stats["frames_with_ball_det"] += 1

            if yolo_selection_mode == "fused_window":
                window_has_person = any(int(det_pack.shape[0]) > 0 for det_pack, _ in fused_window_detections)
                window_has_ball = any(len(ball_pack) > 0 for _, ball_pack in fused_window_detections)
                if window_has_person:
                    stats["frames_with_person_det"] += 1
                if window_has_ball:
                    stats["frames_with_ball_det"] += 1

            # Tracking + team id
            persons: List[Tuple[np.ndarray, int, int]] = []
            if tracking_backend == "ultralytics_track" and len(person_dets) > 0:
                stats["tracker_ok_frames"] += 1
                # person_dets rows are [x1,y1,x2,y2,conf,cls,track_id]
                for d in person_dets:
                    try:
                        x1, y1, x2, y2 = [int(v) for v in d[:4].tolist()]
                        tid = int(float(d[6])) if len(d) >= 7 else -1
                    except Exception:
                        continue

                    x1, y1 = max(0, x1), max(0, y1)
                    x2, y2 = min(frame_bgr.shape[1], x2), min(frame_bgr.shape[0], y2)

                    team_id = -1
                    if embedder is not None and clusterer is not None and (y2 - y1) > 5 and (x2 - x1) > 5:
                        crop = frame_bgr[y1:y2, x1:x2]
                        feat = None
                        try:
                            feat = embedder.get_features(crop)
                        except Exception:
                            feat = None

                        if feat is not None:
                            try:
                                if not clusterer.trained:
                                    # Collect a bit faster by sampling every 3 frames.
                                    if (selected_frame_idx % 3) == 0:
                                        clusterer.collect(feat)
                                        stats["team_cluster_collected"] = int(stats.get("team_cluster_collected", 0)) + 1
                                        if len(getattr(clusterer, "data_bank", [])) >= 50:
                                            clusterer.train()
                                if clusterer.trained:
                                    team_id = int(clusterer.predict(feat))
                            except Exception:
                                team_id = -1

                    persons.append((np.array([x1, y1, x2, y2], dtype=np.int32), int(team_id), int(tid)))
                    if int(tid) != -1 and int(team_id) != -1:
                        team_by_track[int(tid)] = int(team_id)
                    try:
                        stats["tracks_total"] += 1
                    except Exception:
                        pass

            elif tracker is not None and len(person_dets) > 0:
                try:
                    tracks = tracker.update(np.array(person_dets), frame_bgr)
                except Exception:
                    tracks = None
                if tracks is not None:
                    stats["tracker_ok_frames"] += 1
                    for tr in tracks:
                        bbox = tr[:4].astype(int)
                        tid = int(tr[4])
                        x1, y1, x2, y2 = [int(v) for v in bbox.tolist()]
                        x1, y1 = max(0, x1), max(0, y1)
                        x2, y2 = min(frame_bgr.shape[1], x2), min(frame_bgr.shape[0], y2)
                        team_id = -1
                        if embedder is not None and clusterer is not None and (y2 - y1) > 5 and (x2 - x1) > 5:
                            crop = frame_bgr[y1:y2, x1:x2]
                            try:
                                feat = embedder.get_features(crop)
                            except Exception:
                                feat = None
                            if feat is not None:
                                try:
                                    if not clusterer.trained:
                                        clusterer.collect(feat)
                                        if len(getattr(clusterer, "data_bank", [])) >= 50:
                                            clusterer.train()
                                    if clusterer.trained:
                                        team_id = int(clusterer.predict(feat))
                                except Exception:
                                    team_id = -1
                        persons.append((np.array([x1, y1, x2, y2], dtype=np.int32), int(team_id), int(tid)))
                        if int(tid) != -1 and int(team_id) != -1:
                            team_by_track[int(tid)] = int(team_id)
                        try:
                            stats["tracks_total"] += 1
                        except Exception:
                            pass
            else:
                for d in person_dets:
                    bb = d[:4].astype(int)
                    persons.append((bb, -1, -1))

            try:
                if clusterer is not None:
                    stats["team_cluster_trained"] = bool(getattr(clusterer, "trained", False))
            except Exception:
                pass

            # HRNet keypoints/lines -> camera params
            try:
                import torch

                img_pil = ImageFromBGR(frame_bgr)
                img_tensor = tfms_resize(img_pil).unsqueeze(0).to(dev)
                with torch.no_grad():
                    heatmaps = model_kp(img_tensor)
                    heatmaps_l = model_lines(img_tensor)

                kp_coords = get_keypoints_from_heatmap_batch_maxpool(heatmaps[:, :-1, :, :])
                line_coords = get_keypoints_from_heatmap_batch_maxpool_l(heatmaps_l[:, :-1, :, :])

                kp_dict = coords_to_dict(kp_coords, threshold=0.05)
                lines_dict = coords_to_dict(line_coords, threshold=0.05)
                final_dict = complete_keypoints(kp_dict, lines_dict, w=960, h=540, normalize=True)
                keypoints_prediction = final_dict[0]

                cam = FramebyFrameCalib(iwidth=width, iheight=height, denormalize=True)
                cam.update(keypoints_prediction)
                calib_res = cam.heuristic_voting()

                if calib_res and math.isnan(_safe_float(calib_res.get("rep_err", 0.0), 0.0)):
                    calib_res = False
            except Exception:
                calib_res = False

            map_frame = bg.copy()
            meta_frame: Dict[str, Any] = {"players": [], "ball": None}
            rep_err = None

            if calib_res:
                try:
                    params = calib_res["cam_params"]
                    rep_err = _safe_float(calib_res.get("rep_err", None), None) if calib_res.get("rep_err") is not None else None

                    camera = Camera(iwidth=width, iheight=height)
                    camera.from_json_parameters(params)

                    if yolo_selection_mode == "fused_window":
                        for fused_person_dets, fused_ball_boxes in fused_window_detections:
                            fused_persons: List[Tuple[np.ndarray, int, int]] = []
                            for det_row in fused_person_dets:
                                try:
                                    fused_persons.append((det_row[:4].astype(np.int32), -1, -1))
                                except Exception:
                                    continue
                            _, meta_candidate = _draw_map_frame(
                                background=bg,
                                camera=camera,
                                persons=fused_persons,
                                balls=fused_ball_boxes,
                                scale=scale,
                                margin=margin,
                            )
                            window_meta_frames.append(meta_candidate)
                        meta_frame = _fuse_meta_frames(window_meta_frames) if window_meta_frames else {"players": [], "ball": None}
                        map_frame = _render_meta_frame(background=bg, meta_frame=meta_frame, scale=scale, margin=margin)
                    else:
                        map_frame, meta_frame = _draw_map_frame(
                            background=bg, camera=camera, persons=persons, balls=ball_boxes, scale=scale, margin=margin
                        )

                    try:
                        stats["calibration_ok_frames"] += 1
                        stats["projected_player_points"] += int(len((meta_frame.get("players") or [])))
                        stats["projected_ball_points"] += int(1 if (meta_frame.get("ball") is not None) else 0)
                    except Exception:
                        pass

                    # Possession/pass events (best-effort)
                    ball_pos = None
                    if meta_frame.get("ball") and meta_frame["ball"].get("world_xy"):
                        try:
                            ball_pos = np.array(meta_frame["ball"]["world_xy"], dtype=np.float32)
                        except Exception:
                            ball_pos = None

                    current_possessor = None
                    if ball_pos is not None:
                        min_dist = 1e9
                        for p in meta_frame.get("players") or []:
                            try:
                                team_id = int(p.get("team_id", -1))
                                track_id = int(p.get("track_id", -1))
                                wxy = p.get("world_xy")
                                if wxy is None:
                                    continue
                                ppos = np.array(wxy, dtype=np.float32)
                                dist = float(np.linalg.norm(ppos - ball_pos))
                                # Allow team_id=-1 (unknown) as long as tracking id exists.
                                if dist < 2.5 and dist < min_dist and track_id != -1:
                                    min_dist = dist
                                    current_possessor = {"track_id": track_id, "team_id": team_id, "pos": ppos}
                            except Exception:
                                continue

                    if current_possessor is not None:
                        frames_since_loss = 0
                        if last_possessor is None:
                            last_possessor = current_possessor
                            events.append(
                                {
                                    "t": t_sec,
                                    "source": "calibration",
                                    "type": "possession_start",
                                    "team_id": int(current_possessor["team_id"]),
                                    "player_track_id": int(current_possessor["track_id"]),
                                }
                            )
                        else:
                            is_same_team = int(last_possessor["team_id"]) == int(current_possessor["team_id"])
                            is_different_id = int(last_possessor["track_id"]) != int(current_possessor["track_id"])
                            try:
                                dist_between = float(np.linalg.norm(last_possessor["pos"] - current_possessor["pos"]))
                            except Exception:
                                dist_between = 0.0
                            is_physically_different = dist_between > 3.0

                            if is_same_team:
                                if is_different_id and is_physically_different:
                                    if potential_possessor is not None and int(potential_possessor["track_id"]) == int(current_possessor["track_id"]):
                                        potential_frames += 1
                                    else:
                                        potential_possessor = current_possessor
                                        potential_frames = 1

                                    if potential_frames >= 6:
                                        # Confirmed pass
                                        from_id = int(last_possessor["track_id"])
                                        to_id = int(current_possessor["track_id"])
                                        pass_dist = float(np.linalg.norm(last_possessor["pos"] - current_possessor["pos"]))
                                        events.append(
                                            {
                                                "t": t_sec,
                                                "source": "calibration",
                                                "type": "pass",
                                                "team_id": int(current_possessor["team_id"]),
                                                "from_player_track_id": from_id,
                                                "player_track_id": to_id,
                                                "distance_m": float(pass_dist),
                                            }
                                        )
                                        try:
                                            stats["events"] += 1
                                        except Exception:
                                            pass
                                        last_possessor = current_possessor
                                        potential_possessor = None
                                        potential_frames = 0
                                else:
                                    # Same player; update position
                                    last_possessor["pos"] = current_possessor["pos"]
                                    potential_possessor = None
                                    potential_frames = 0
                            else:
                                # Turnover / possession change after 5 frames confirmation
                                if potential_possessor is not None and int(potential_possessor["track_id"]) == int(current_possessor["track_id"]):
                                    potential_frames += 1
                                else:
                                    potential_possessor = current_possessor
                                    potential_frames = 1

                                if potential_frames >= 5:
                                    events.append(
                                        {
                                            "t": t_sec,
                                            "source": "calibration",
                                            "type": "possession_change",
                                            "from_team_id": int(last_possessor.get("team_id", -1)),
                                            "team_id": int(current_possessor["team_id"]),
                                            "from_player_track_id": int(last_possessor.get("track_id", -1)),
                                            "player_track_id": int(current_possessor["track_id"]),
                                        }
                                    )
                                    try:
                                        stats["events"] += 1
                                    except Exception:
                                        pass
                                    last_possessor = current_possessor
                                    potential_possessor = None
                                    potential_frames = 0
                    else:
                        frames_since_loss += 1
                        if frames_since_loss > 15:
                            potential_possessor = None
                            potential_frames = 0

                except Exception:
                    # If any calibration-dependent step fails, fall back to empty map frame.
                    pass

            sample_packet = {
                "frame_idx": int(selected_frame_idx),
                "t": float(t_sec),
                "meta_frame": meta_frame,
                "map_frame": map_frame,
                "calibration_ok": bool(calib_res),
                "rep_err": rep_err,
            }

            if pending_sample is None:
                for lead_frame_idx in range(int(frame_batch[0][0]), int(selected_frame_idx)):
                    _write_frame_record(
                        vw=vw,
                        f_frames=f_frames,
                        frame_bgr=map_frame,
                        frame_idx=lead_frame_idx,
                        fps=fps,
                        frames_stride=frames_stride,
                        calib_res=bool(calib_res),
                        rep_err=rep_err,
                        meta_frame=meta_frame,
                        sampled_frame_idx=int(selected_frame_idx),
                        sampled_frame_t=float(t_sec),
                    )
            else:
                prev_frame_idx = int(pending_sample["frame_idx"])
                next_frame_idx = int(selected_frame_idx)
                frame_gap = max(1, next_frame_idx - prev_frame_idx)
                for out_frame_idx in range(prev_frame_idx, next_frame_idx):
                    alpha = float(out_frame_idx - prev_frame_idx) / float(frame_gap)
                    interp_meta = _interpolate_meta_frames(
                        pending_sample["meta_frame"],
                        meta_frame,
                        alpha,
                        interpolation_mode,
                    )
                    interp_frame = _render_meta_frame(
                        background=bg,
                        meta_frame=interp_meta,
                        scale=scale,
                        margin=margin,
                    )
                    interp_calib_ok = bool(pending_sample["calibration_ok"] if alpha < 0.5 else calib_res)
                    if pending_sample["rep_err"] is not None and rep_err is not None:
                        interp_rep_err = float((1.0 - alpha) * float(pending_sample["rep_err"]) + alpha * float(rep_err))
                    else:
                        interp_rep_err = pending_sample["rep_err"] if alpha < 0.5 else rep_err
                    _write_frame_record(
                        vw=vw,
                        f_frames=f_frames,
                        frame_bgr=interp_frame,
                        frame_idx=out_frame_idx,
                        fps=fps,
                        frames_stride=frames_stride,
                        calib_res=interp_calib_ok,
                        rep_err=interp_rep_err,
                        meta_frame=interp_meta,
                        sampled_frame_idx=int(selected_frame_idx if alpha >= 0.5 else prev_frame_idx),
                        sampled_frame_t=float(t_sec if alpha >= 0.5 else pending_sample["t"]),
                        interpolation_alpha=alpha,
                        interpolation_from_idx=prev_frame_idx,
                        interpolation_to_idx=next_frame_idx,
                    )
                    if alpha > 0.0:
                        stats["interpolated_output_frames"] += 1

            pending_sample = sample_packet

            frame_idx += len(frame_batch)

        if pending_sample is not None:
            final_start_idx = int(pending_sample["frame_idx"])
            final_end_idx = int(frame_idx)
            for out_frame_idx in range(final_start_idx, final_end_idx):
                _write_frame_record(
                    vw=vw,
                    f_frames=f_frames,
                    frame_bgr=pending_sample["map_frame"],
                    frame_idx=out_frame_idx,
                    fps=fps,
                    frames_stride=frames_stride,
                    calib_res=bool(pending_sample["calibration_ok"]),
                    rep_err=pending_sample["rep_err"],
                    meta_frame=pending_sample["meta_frame"],
                    sampled_frame_idx=int(pending_sample["frame_idx"]),
                    sampled_frame_t=float(pending_sample["t"]),
                )
    finally:
        try:
            if f_frames is not None:
                f_frames.close()
        except Exception:
            pass

        cap.release()
        vw.release()

        # Backfill team ids in events if they were unknown (-1) at event time.
        try:
            for e in events:
                if not isinstance(e, dict):
                    continue
                et = str(e.get("type") or "")

                if et in ("possession_start", "possession_change"):
                    try:
                        pid = e.get("player_track_id")
                        if (e.get("team_id") is None) or int(e.get("team_id", -1)) == -1:
                            if pid is not None and int(pid) in team_by_track:
                                e["team_id"] = int(team_by_track[int(pid)])
                    except Exception:
                        pass
                    try:
                        fpid = e.get("from_player_track_id")
                        if (e.get("from_team_id") is None) or int(e.get("from_team_id", -1)) == -1:
                            if fpid is not None and int(fpid) in team_by_track:
                                e["from_team_id"] = int(team_by_track[int(fpid)])
                    except Exception:
                        pass

                if et == "pass":
                    try:
                        pid = e.get("player_track_id")
                        if (e.get("team_id") is None) or int(e.get("team_id", -1)) == -1:
                            if pid is not None and int(pid) in team_by_track:
                                e["team_id"] = int(team_by_track[int(pid)])
                    except Exception:
                        pass
        except Exception:
            pass

        try:
            stats["team_by_track_size"] = int(len(team_by_track))
        except Exception:
            pass

        # Final progress tick
        try:
            print(
                "__PROGRESS__ "
                + json.dumps(
                    {
                        "stage": "calibration",
                        "current": int(total_frames if total_frames > 0 else frame_idx),
                        "total": int(total_frames) if total_frames > 0 else 0,
                        "message": "Calibration tamam",
                    },
                    ensure_ascii=False,
                ),
                flush=True,
            )
        except Exception:
            pass

    out_events_p = Path(out_events)
    out_events_p.parent.mkdir(parents=True, exist_ok=True)
    with open(out_events_p, "w", encoding="utf-8") as f_ev:
        json.dump(
            {
                "schema_version": "1.0",
                "created_utc": None,
                "source": {"video_path": str(Path(source).resolve())},
                "run_config": run_config,
                "artifacts": {
                    "map_video_path": str(out_map_p.resolve()),
                    **({"frames_jsonl_path": str(out_frames_p.resolve())} if write_frames and out_frames_p is not None else {}),
                },
                "stats": stats,
                "events": events,
            },
            f_ev,
            ensure_ascii=False,
            indent=2,
        )

    if det_cache is not None:
        try:
            det_cache.flush()
        except Exception:
            pass

    return CalibrationOutputs(
        map_video_path=str(out_map_p.resolve()),
        events_json_path=str(out_events_p.resolve()),
        frames_jsonl_path=str(out_frames_p.resolve()) if write_frames and out_frames_p is not None else "",
    )


def ImageFromBGR(frame_bgr: np.ndarray):
    from PIL import Image

    return Image.fromarray(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", type=str, required=True, help="input video path")
    ap.add_argument("--out_map", type=str, required=True, help="output map video path (.mp4)")
    ap.add_argument("--out_events", type=str, required=True, help="output events json path")
    ap.add_argument("--out_frames", type=str, default="", help="output per-frame jsonl path (optional; can be huge)")
    ap.add_argument("--frames_stride", type=int, default=1, help="write every N frames into jsonl (only if out_frames is set)")
    ap.add_argument("--progress_every", type=int, default=25, help="emit progress every N frames")
    ap.add_argument("--max_frames", type=int, default=0, help="process at most N frames (0 = all)")
    ap.add_argument(
        "--yolo_frame_window",
        type=int,
        default=8,
        help="run YOLO on one selected frame per N-frame window, preferring the earliest frame with a ball",
    )
    ap.add_argument(
        "--yolo_selection_mode",
        type=str,
        default="ball_priority",
        choices=["ball_priority", "first", "fused_window"],
        help="choose earliest-ball probing, always-first sampling, or fused detections from all frames in each window",
    )
    ap.add_argument(
        "--interpolation_mode",
        type=str,
        default="linear",
        choices=["linear", "smoothstep"],
        help="choose how skipped output frames are interpolated between sampled map states",
    )
    ap.add_argument("--detector", type=str, default=str(THIS_DIR / "best.pt"), help="YOLO weights path")
    ap.add_argument("--kp_weights", type=str, default=str(THIS_DIR / "SV_kp.pth"), help="HRNet keypoints weights")
    ap.add_argument("--line_weights", type=str, default=str(THIS_DIR / "SV_lines.pth"), help="HRNet lines weights")
    ap.add_argument("--conf", type=float, default=0.30, help="detector confidence threshold")
    ap.add_argument("--detection_cache", type=str, default="", help="path to detection cache JSON (optional)")
    ap.add_argument("--torch_compile", action="store_true", default=False, help="enable torch.compile for HRNet models")
    ap.add_argument("--torch_compile_mode", type=str, default="reduce-overhead", help="torch.compile mode")
    args = ap.parse_args()

    outs = run(
        source=str(args.source),
        out_map=str(args.out_map),
        out_events=str(args.out_events),
        out_frames=str(args.out_frames) if str(args.out_frames or "").strip() else None,
        detector_weights=str(args.detector),
        kp_weights=str(args.kp_weights),
        line_weights=str(args.line_weights),
        conf_thres=float(args.conf),
        frames_stride=int(args.frames_stride),
        progress_every=int(args.progress_every),
        max_frames=int(args.max_frames),
        yolo_frame_window=int(args.yolo_frame_window),
        yolo_selection_mode=str(args.yolo_selection_mode),
        interpolation_mode=str(args.interpolation_mode),
        detection_cache_path=str(args.detection_cache) if str(args.detection_cache or "").strip() else None,
    )

    # Print a short machine-readable summary for the caller.
    print(
        json.dumps(
            {
                "map_video_path": outs.map_video_path,
                "events_json_path": outs.events_json_path,
                **({"frames_jsonl_path": outs.frames_jsonl_path} if outs.frames_jsonl_path else {}),
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
