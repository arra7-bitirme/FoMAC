from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterable, Optional

import cv2
import numpy as np

from .class_aware_tracker import Tracklet


def draw_tracks(
    frame: np.ndarray,
    tracklets: Iterable[Tracklet],
    class_colors: Dict[int, tuple],
    thickness: int = 2,
    team_assignments: Optional[Dict[int, int]] = None,
    team_colors: Optional[Dict[int, tuple]] = None,
) -> np.ndarray:
    """Annotate frame with bounding boxes + IDs."""

    if frame is None:
        return frame

    for track in tracklets:
        team_id = None
        if team_assignments:
            team_id = team_assignments.get(track.track_id)
        if team_id is not None and team_id >= 0 and team_colors:
            color = team_colors.get(team_id, class_colors.get(track.class_id, (0, 255, 255)))
        else:
            color = class_colors.get(track.class_id, (0, 255, 255))
        x1, y1, x2, y2 = track.bbox.astype(int)
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)
        label = f"ID {track.track_id}"
        if team_id is not None and team_id >= 0:
            label += f" T{team_id}"
        cv2.putText(
            frame,
            label,
            (x1, max(0, y1 - 6)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            color,
            1,
            lineType=cv2.LINE_AA,
        )
    return frame


def save_visual_frame(output_dir: Path, frame_idx: int, frame: np.ndarray) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{frame_idx:06d}.jpg"
    cv2.imwrite(str(output_path), frame)
    return output_path
