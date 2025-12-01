from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set
try:
    from typing import Protocol
except ImportError:  # pragma: no cover
    Protocol = object  # type: ignore[misc, assignment]

import cv2
import numpy as np

try:  # Optional dependency loaded lazily
    from sklearn.cluster import KMeans
except ImportError:  # pragma: no cover
    KMeans = None  # type: ignore[assignment]


class ProgressReporter(Protocol):
    def start_task(self, label: str, total: Optional[int] = None, *, color: str = "magenta") -> Any:
        ...

    def advance_task(self, task_id: Any, advance: float = 1.0) -> None:
        ...

    def close_task(self, task_id: Any) -> None:
        ...


@dataclass
class TeamClassificationSettings:
    enabled: bool = False
    method: str = "color"
    samples_per_track: int = 12
    min_track_hits: int = 5
    color_space: str = "lab"
    blur_kernel: int = 5
    clusters: int = 2
    focus_ratio: float = 0.6
    model_weights: Optional[str] = None

    @classmethod
    def from_dict(cls, data: Optional[Dict[str, object]]) -> "TeamClassificationSettings":
        if not data:
            return cls()
        return cls(
            enabled=bool(data.get("enabled", False)),
            method=str(data.get("method", "color")),
            samples_per_track=int(data.get("samples_per_track", 12)),
            min_track_hits=int(data.get("min_track_hits", 5)),
            color_space=str(data.get("color_space", "lab")),
            blur_kernel=int(data.get("blur_kernel", 5)),
            clusters=int(data.get("clusters", 2)),
            focus_ratio=float(data.get("focus_ratio", 0.6)),
            model_weights=data.get("model_weights"),
        )


def assign_team_ids(
    sequence,
    records: List[Dict[str, float]],
    settings: TeamClassificationSettings,
    allowed_classes: Set[int],
    progress: Optional[ProgressReporter] = None,
) -> Dict[int, int]:
    if not settings.enabled:
        return {}
    if settings.method == "color":
        return _assign_color_clusters(sequence, records, settings, allowed_classes, progress)
    if settings.method == "model":
        raise NotImplementedError("Learned team classifier integration pending")
    raise ValueError(f"Unknown team classification method: {settings.method}")


def _assign_color_clusters(
    sequence,
    records: List[Dict[str, float]],
    settings: TeamClassificationSettings,
    allowed_classes: Set[int],
    progress: Optional[ProgressReporter] = None,
) -> Dict[int, int]:
    if KMeans is None:
        raise ImportError(
            "scikit-learn is required for color-based team classification. Install with `pip install scikit-learn`."
        )

    tracks: Dict[int, List[Dict[str, float]]] = defaultdict(list)
    for record in records:
        if int(record["class_id"]) not in allowed_classes:
            continue
        tracks[int(record["track_id"])].append(record)

    if not tracks:
        return {}

    eligible_tracks = {
        track_id: tr
        for track_id, tr in tracks.items()
        if len(tr) >= settings.min_track_hits
    }

    if not eligible_tracks:
        return {}

    sample_total = sum(min(len(tr), settings.samples_per_track) for tr in eligible_tracks.values())
    sample_task = None
    if progress is not None and sample_total > 0:
        sample_task = progress.start_task("Team patch extraction", total=sample_total)

    features: List[np.ndarray] = []
    owners: List[int] = []

    for track_id, track_records in eligible_tracks.items():
        samples = _sample_records(track_records, settings.samples_per_track)
        patches = _load_patches(sequence, samples, settings.focus_ratio, progress)
        for patch in patches:
            feature = _patch_feature(patch, settings)
            if feature is not None:
                features.append(feature)
                owners.append(track_id)
        if progress is not None:
            progress.advance_task(sample_task, len(patches))

    if progress is not None:
        progress.close_task(sample_task)

    if len(features) < settings.clusters:
        return {}

    feature_matrix = np.stack(features, axis=0)
    kmeans = KMeans(n_clusters=settings.clusters, n_init=10, random_state=0)
    cluster_task = None
    if progress is not None:
        cluster_task = progress.start_task("Team clustering", total=1, color="green")
    labels = kmeans.fit_predict(feature_matrix)
    if progress is not None:
        progress.advance_task(cluster_task)
        progress.close_task(cluster_task)

    votes: Dict[int, List[int]] = defaultdict(list)
    for track_id, label in zip(owners, labels):
        votes[track_id].append(int(label))

    assignments: Dict[int, int] = {}
    for track_id, voter_labels in votes.items():
        choice, _ = Counter(voter_labels).most_common(1)[0]
        assignments[track_id] = choice

    return assignments


def _sample_records(records: Sequence[Dict[str, float]], desired: int) -> List[Dict[str, float]]:
    if len(records) <= desired:
        return list(records)
    idxs = np.linspace(0, len(records) - 1, desired, dtype=int)
    return [records[i] for i in idxs]


def _load_patches(
    sequence,
    samples: Iterable[Dict[str, float]],
    focus_ratio: float,
    progress: Optional[ProgressReporter] = None,
) -> List[np.ndarray]:
    patches: List[np.ndarray] = []
    samples_by_frame: Dict[int, List[Dict[str, float]]] = defaultdict(list)
    for record in samples:
        samples_by_frame[int(record["frame"])].append(record)

    if not samples_by_frame:
        return patches

    img_dir = sequence.img_dir
    frame_task = None
    total_frames = 0
    if img_dir.is_file():
        total_frames = max(samples_by_frame)
    else:
        total_frames = len(samples_by_frame)
    if progress is not None and total_frames > 0:
        frame_task = progress.start_task(
            "Team frame extraction",
            total=total_frames,
            color="blue",
        )
    if img_dir.is_file():
        cap = cv2.VideoCapture(str(img_dir))
        if not cap.isOpened():
            if progress is not None:
                progress.close_task(frame_task)
            return patches
        current_idx = 0
        max_needed = max(samples_by_frame)
        while current_idx < max_needed:
            ret, frame = cap.read()
            if not ret:
                break
            current_idx += 1
            if progress is not None:
                progress.advance_task(frame_task, 1)
            if current_idx not in samples_by_frame:
                continue
            patches.extend(
                _extract_patches_from_frame(frame, samples_by_frame[current_idx], focus_ratio)
            )
        cap.release()
        if progress is not None:
            progress.close_task(frame_task)
        return patches

    img_ext = sequence.seqinfo.get("imext") or sequence.seqinfo.get("imExt") or ".jpg"
    img_ext = img_ext if img_ext.startswith(".") else f".{img_ext}"
    for frame_idx, records in samples_by_frame.items():
        frame_path = img_dir / f"{frame_idx:06d}{img_ext}"
        if not frame_path.exists():
            continue
        frame = cv2.imread(str(frame_path))
        if frame is None:
            continue
        patches.extend(_extract_patches_from_frame(frame, records, focus_ratio))
        if progress is not None:
            progress.advance_task(frame_task, 1)
    if progress is not None:
        progress.close_task(frame_task)
    return patches


def _extract_patches_from_frame(frame: np.ndarray, records: Sequence[Dict[str, float]], focus_ratio: float) -> List[np.ndarray]:
    patches: List[np.ndarray] = []
    for record in records:
        x = int(record["x"])
        y = int(record["y"])
        w = max(2, int(record["w"]))
        h = max(2, int(record["h"]))
        x2 = min(frame.shape[1], x + w)
        y2 = min(frame.shape[0], y + h)
        y_focus = int(y + h * focus_ratio)
        patch = frame[y:y_focus, x:x2]
        if patch.size == 0:
            continue
        patches.append(patch)
    return patches


def _patch_feature(patch: np.ndarray, settings: TeamClassificationSettings) -> Optional[np.ndarray]:
    kernel = settings.blur_kernel
    if kernel > 1 and kernel % 2 == 1:
        patch = cv2.GaussianBlur(patch, (kernel, kernel), 0)

    if settings.color_space.lower() == "lab":
        patch = cv2.cvtColor(patch, cv2.COLOR_BGR2LAB)
    elif settings.color_space.lower() == "hsv":
        patch = cv2.cvtColor(patch, cv2.COLOR_BGR2HSV)
    elif settings.color_space.lower() == "rgb":
        patch = cv2.cvtColor(patch, cv2.COLOR_BGR2RGB)

    pixels = patch.reshape(-1, patch.shape[-1]).astype(np.float32)
    if pixels.size == 0:
        return None
    mean = pixels.mean(axis=0)
    std = pixels.std(axis=0)
    return np.concatenate([mean, std], axis=0)
