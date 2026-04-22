"""
Global Detection Cache — frame-level YOLO detection results shared between pipeline subprocesses.

Format: JSON file keyed by 0-indexed frame_id → [[x1, y1, x2, y2, conf, cls_id], ...]

Subprocesses use 0-indexed keys:
  - run_pipeline_calibration.py  : selected_frame_idx  (already 0-indexed)
  - run_botsort_team_reid.py     : frame_id - 1        (its loop is 1-indexed)
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Dict, List, Optional


class DetectionCache:
    def __init__(self, path: str) -> None:
        self._path = Path(path)
        self._data: Dict[str, List] = {}
        self._dirty = False
        if self._path.exists():
            try:
                with open(self._path, "r", encoding="utf-8") as f:
                    self._data = json.load(f)
            except Exception:
                self._data = {}

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def get(self, frame_idx: int) -> Optional[List[List[float]]]:
        """Return cached detections for *frame_idx*, or None on cache miss."""
        return self._data.get(str(frame_idx))

    def set(self, frame_idx: int, dets: List[List[float]]) -> None:
        """Store detections for *frame_idx* (rows: [x1, y1, x2, y2, conf, cls_id])."""
        self._data[str(frame_idx)] = dets
        self._dirty = True

    def flush(self) -> None:
        """Atomically persist cache to disk (no-op if unchanged)."""
        if not self._dirty:
            return
        self._path.parent.mkdir(parents=True, exist_ok=True)
        tmp = str(self._path) + ".tmp"
        try:
            with open(tmp, "w", encoding="utf-8") as f:
                json.dump(self._data, f)
            os.replace(tmp, str(self._path))
            self._dirty = False
        except Exception:
            try:
                os.remove(tmp)
            except Exception:
                pass

    def __len__(self) -> int:
        return len(self._data)

    def __contains__(self, frame_idx: object) -> bool:
        return str(frame_idx) in self._data


# ---------------------------------------------------------------------------
# Helpers used by subprocess scripts (no YOLO dependency here)
# ---------------------------------------------------------------------------

def boxes_to_cache(result: object) -> List[List[float]]:
    """
    Extract raw detection rows from a YOLO result object for caching.
    Returns [[x1, y1, x2, y2, conf, cls_id], ...].
    Track IDs (7th column) are intentionally dropped.
    """
    try:
        if result is None:
            return []
        boxes = getattr(result, "boxes", None)
        if boxes is None:
            return []
        data = getattr(boxes, "data", None)
        if data is None:
            return []
        rows = data.detach().cpu().numpy().tolist()
        return [[float(r[i]) for i in range(6)] for r in rows if len(r) >= 6]
    except Exception:
        return []


def cached_to_dets_np(cached: List[List[float]]):
    """Convert a cached entry back to a float32 numpy array of shape (N, 6)."""
    import numpy as np

    if not cached:
        return np.empty((0, 6), dtype=np.float32)
    return np.array(cached, dtype=np.float32)
