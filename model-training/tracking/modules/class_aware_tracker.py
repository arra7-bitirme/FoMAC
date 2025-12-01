from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
from scipy.optimize import linear_sum_assignment


Detection = Dict[str, Any]


@dataclass
class Tracklet:
    """Lightweight track representation used across the tracking pipeline."""

    track_id: int
    class_id: int
    bbox: np.ndarray  # xyxy format
    score: float
    frame_idx: int
    hits: int

    @property
    def xywh(self) -> np.ndarray:
        x1, y1, x2, y2 = self.bbox
        return np.array([x1, y1, x2 - x1, y2 - y1], dtype=np.float32)


@dataclass
class _TrackState:
    bbox: np.ndarray
    score: float
    class_id: int
    last_frame: int
    hits: int = 1
    time_since_update: int = 0
    embedding: Optional[np.ndarray] = None


class IoUTracker:
    """Simple IoU + Hungarian tracker for class groups."""

    def __init__(
        self,
        class_ids: Sequence[int],
        max_age: int = 30,
        min_hits: int = 3,
        iou_threshold: float = 0.3,
        start_id: int = 1,
        association: Optional[Dict[str, float]] = None,
    ) -> None:
        self.class_ids = set(class_ids)
        self.max_age = max_age
        self.min_hits = min_hits
        self.iou_threshold = iou_threshold
        self.next_id = start_id
        self.tracks: Dict[int, _TrackState] = {}
        assoc = association or {}
        self.alpha = float(assoc.get("iou_weight", 1.0))
        self.beta = float(assoc.get("appearance_weight", 0.0))
        self.max_embed_distance = float(assoc.get("max_distance", 0.7))
        self.use_embeddings = self.beta > 0

    def update(
        self,
        detections: Sequence[Detection],
        frame_idx: int,
    ) -> List[Tracklet]:
        if not detections and not self.tracks:
            return []

        det_bboxes = [det["bbox"] for det in detections]
        det_scores = [det["score"] for det in detections]
        det_classes = [det["class_id"] for det in detections]
        det_embeddings = [det.get("embedding") for det in detections]

        track_ids = list(self.tracks.keys())
        matches, unmatched_tracks, unmatched_dets = self._associate(det_bboxes, det_embeddings, track_ids)

        # Update matched tracks
        for track_idx, det_idx in matches:
            track_id = track_ids[track_idx]
            bbox = det_bboxes[det_idx]
            score = det_scores[det_idx]
            cls = det_classes[det_idx]
            embedding = det_embeddings[det_idx]
            state = self.tracks[track_id]
            state.bbox = bbox
            state.score = score
            state.class_id = cls
            state.last_frame = frame_idx
            state.hits += 1
            state.time_since_update = 0
            if embedding is not None:
                state.embedding = embedding

        # Age unmatched tracks
        for track_idx in unmatched_tracks:
            track_id = track_ids[track_idx]
            state = self.tracks[track_id]
            state.time_since_update += 1

        # Create new tracks for unmatched detections
        for det_idx in unmatched_dets:
            bbox = det_bboxes[det_idx]
            score = det_scores[det_idx]
            cls = det_classes[det_idx]
            embedding = det_embeddings[det_idx]
            if cls not in self.class_ids:
                continue
            self.tracks[self.next_id] = _TrackState(
                bbox=bbox,
                score=score,
                class_id=cls,
                last_frame=frame_idx,
                embedding=embedding,
            )
            self.next_id += 1

        # Drop stale tracks
        dropped = [tid for tid, state in self.tracks.items() if state.time_since_update > self.max_age]
        for tid in dropped:
            self.tracks.pop(tid)

        # Collect active tracklets
        active: List[Tracklet] = []
        for track_id, state in self.tracks.items():
            if state.hits < self.min_hits and state.time_since_update > 0:
                continue
            active.append(
                Tracklet(
                    track_id=track_id,
                    class_id=state.class_id,
                    bbox=state.bbox.copy(),
                    score=state.score,
                    frame_idx=frame_idx,
                    hits=state.hits,
                )
            )
        return active

    def _associate(
        self,
        det_bboxes: Sequence[np.ndarray],
        det_embeddings: Sequence[Optional[np.ndarray]],
        track_ids: Sequence[int],
    ) -> Tuple[List[Tuple[int, int]], List[int], List[int]]:
        if not det_bboxes or not track_ids:
            return [], list(range(len(track_ids))), list(range(len(det_bboxes)))

        track_bboxes = [self.tracks[tid].bbox for tid in track_ids]
        cost, iou_matrix = self._build_cost_matrix(track_ids, track_bboxes, det_bboxes, det_embeddings)
        row_idx, col_idx = linear_sum_assignment(cost)

        matches: List[Tuple[int, int]] = []
        unmatched_tracks = set(range(len(track_ids)))
        unmatched_dets = set(range(len(det_bboxes)))

        for r, c in zip(row_idx, col_idx):
            if iou_matrix[r, c] < self.iou_threshold:
                continue
            if cost[r, c] >= 1e3:
                continue
            matches.append((r, c))
            unmatched_tracks.discard(r)
            unmatched_dets.discard(c)

        return matches, sorted(unmatched_tracks), sorted(unmatched_dets)

    def _build_cost_matrix(
        self,
        track_ids: Sequence[int],
        track_bboxes: Sequence[np.ndarray],
        det_bboxes: Sequence[np.ndarray],
        det_embeddings: Sequence[Optional[np.ndarray]],
    ) -> Tuple[np.ndarray, np.ndarray]:
        iou_matrix = _pairwise_iou(track_bboxes, det_bboxes)
        if not self.use_embeddings:
            return 1.0 - iou_matrix, iou_matrix

        cost = np.empty_like(iou_matrix)
        for i, track_id in enumerate(track_ids):
            track_state = self.tracks[track_id]
            track_emb = track_state.embedding
            for j, det_emb in enumerate(det_embeddings):
                iou_term = 1.0 - iou_matrix[i, j]
                if track_emb is None or det_emb is None:
                    cost[i, j] = iou_term
                    continue
                dist = _cosine_distance(track_emb, det_emb)
                if dist > self.max_embed_distance:
                    cost[i, j] = 1e3  # force unmatched
                else:
                    cost[i, j] = self.alpha * iou_term + self.beta * dist
        return cost, iou_matrix


class BallTracker:
    """Single-object tracker dedicated to the ball."""

    def __init__(self, class_ids: Sequence[int], max_age: int = 12, start_id: int = 9000) -> None:
        self.class_ids = set(class_ids)
        self.max_age = max_age
        self.track_id = start_id
        self.state: Optional[_TrackState] = None

    def update(
        self,
        detections: Sequence[Detection],
        frame_idx: int,
    ) -> List[Tracklet]:
        ball_dets = [det for det in detections if det["class_id"] in self.class_ids]
        updated = False

        if ball_dets:
            best_det = max(ball_dets, key=lambda det: det["score"])
            bbox = best_det["bbox"]
            score = best_det["score"]
            cls = best_det["class_id"]
            if self.state is None:
                self.state = _TrackState(bbox=bbox, score=score, class_id=cls, last_frame=frame_idx)
            else:
                self.state.bbox = bbox
                self.state.score = score
                self.state.class_id = cls
                self.state.last_frame = frame_idx
            self.state.time_since_update = 0
            self.state.hits += 1
            updated = True
        else:
            if self.state is not None:
                self.state.time_since_update += 1
                if self.state.time_since_update > self.max_age:
                    self.state = None

        if not updated or self.state is None:
            return []

        return [
            Tracklet(
                track_id=self.track_id,
                class_id=self.state.class_id,
                bbox=self.state.bbox.copy(),
                score=self.state.score,
                frame_idx=frame_idx,
                hits=self.state.hits,
            )
        ]


def _pairwise_iou(
    tracks: Sequence[np.ndarray],
    detections: Sequence[np.ndarray],
) -> np.ndarray:
    if not tracks or not detections:
        return np.zeros((len(tracks), len(detections)), dtype=np.float32)

    tracks_arr = np.stack(tracks, axis=0)
    dets_arr = np.stack(detections, axis=0)

    tl = np.maximum(tracks_arr[:, None, :2], dets_arr[None, :, :2])
    br = np.minimum(tracks_arr[:, None, 2:], dets_arr[None, :, 2:])
    inter_wh = np.clip(br - tl, a_min=0.0, a_max=None)
    inter_area = inter_wh[..., 0] * inter_wh[..., 1]

    track_area = (tracks_arr[:, 2] - tracks_arr[:, 0]) * (tracks_arr[:, 3] - tracks_arr[:, 1])
    det_area = (dets_arr[:, 2] - dets_arr[:, 0]) * (dets_arr[:, 3] - dets_arr[:, 1])

    union = track_area[:, None] + det_area[None, :] - inter_area
    union = np.clip(union, a_min=1e-6, a_max=None)
    return inter_area / union


def _cosine_distance(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=np.float32)
    b = np.asarray(b, dtype=np.float32)
    denom = float(np.linalg.norm(a) * np.linalg.norm(b))
    if denom < 1e-6:
        return 1.0
    cosine = float(np.dot(a, b) / denom)
    cosine = max(min(cosine, 1.0), -1.0)
    return 1.0 - cosine


def xyxy_from_tensor(tensor) -> np.ndarray:
    """Utility to convert a tensor-like bbox to numpy array."""
    if hasattr(tensor, "cpu"):
        tensor = tensor.cpu().numpy()
    return np.asarray(tensor, dtype=np.float32)
