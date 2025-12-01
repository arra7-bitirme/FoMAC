from __future__ import annotations

from dataclasses import dataclass
from typing import Generator, Optional

import numpy as np
from ultralytics import YOLO


@dataclass
class FrameDetections:
    """Container for a single frame's detections and image."""

    frame_idx: int
    frame: Optional[np.ndarray]
    result: object


def iter_detections(
    model: YOLO,
    source: str,
    imgsz: int,
    device: str,
    conf: float,
    iou: float,
    half: bool = False,
    vid_stride: int = 1,
    limit_frames: Optional[int] = None,
) -> Generator[FrameDetections, None, None]:
    """Yield YOLO detections frame-by-frame for a directory of images.

    Args:
        model: Loaded Ultralytics YOLO model.
        source: Path to image directory / video / glob.
        imgsz: Inference resolution.
        device: Device string ("cpu", "cuda:0", etc.).
        conf: Detector confidence threshold.
        iou: NMS IoU threshold.
        half: Whether to use half-precision (if supported).
        vid_stride: Frame skipping stride for video inputs.
        limit_frames: Optional hard stop for number of frames.
    """

    stream = model.predict(
        source=source,
        imgsz=imgsz,
        device=device,
        conf=conf,
        iou=iou,
        half=half,
        stream=True,
        verbose=False,
        vid_stride=vid_stride,
        save=False,
    )

    for frame_idx, result in enumerate(stream, start=1):
        if limit_frames is not None and frame_idx > limit_frames:
            break
        frame = getattr(result, "orig_img", None)
        yield FrameDetections(frame_idx=frame_idx, frame=frame, result=result)
