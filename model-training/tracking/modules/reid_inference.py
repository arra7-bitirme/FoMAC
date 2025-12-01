from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional, Sequence, Set

import numpy as np
import torch
from PIL import Image
from torchvision import transforms

from .reid_training import ReIDTrainerConfig, YOLOReIDModel, _resolve_device


@dataclass
class ReIDRuntimeConfig:
    weights: str
    device: str = "auto"
    image_size: Sequence[int] = (256, 128)
    target_classes: str = "players"


class ReIDEmbeddingExtractor:
    """Loads a trained ReID head and produces embeddings for detection crops."""

    def __init__(self, runtime_cfg: ReIDRuntimeConfig) -> None:
        checkpoint_path = Path(runtime_cfg.weights)
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"ReID checkpoint not found: {checkpoint_path}")

        ckpt = torch.load(checkpoint_path, map_location="cpu")
        cfg_dict = ckpt.get("config") or {}
        reid_cfg = ReIDTrainerConfig(**cfg_dict)
        if runtime_cfg.image_size:
            reid_cfg.image_size = tuple(runtime_cfg.image_size)
        device = _resolve_device(runtime_cfg.device)
        self.device = device
        num_classes = max(1, int(ckpt.get("num_classes", 1)))
        self.model = YOLOReIDModel(reid_cfg, num_classes, device).to(device)
        self.model.load_state_dict(ckpt["state_dict"], strict=False)
        self.model.eval()

        self.transform = transforms.Compose(
            [
                transforms.Resize(reid_cfg.image_size),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        )
        self.target_classes = runtime_cfg.target_classes

    @torch.no_grad()
    def extract(
        self,
        frame: np.ndarray,
        boxes: np.ndarray,
        class_ids: Sequence[int],
        allowed_classes: Set[int],
    ) -> List[Optional[np.ndarray]]:
        num_detections = len(class_ids)
        embeddings: List[Optional[np.ndarray]] = [None] * num_detections
        tensors = []
        idx_map: List[int] = []
        for idx, (bbox, class_id) in enumerate(zip(boxes, class_ids)):
            if allowed_classes and class_id not in allowed_classes:
                continue
            crop = _crop_patch(frame, bbox)
            if crop is None:
                continue
            tensor = self.transform(crop)
            tensors.append(tensor)
            idx_map.append(idx)
        if not tensors:
            return embeddings
        batch = torch.stack(tensors).to(self.device)
        outputs = self.model(batch)
        batch_embeddings = outputs["embeddings"].detach().cpu().numpy()
        for local_idx, emb in enumerate(batch_embeddings):
            embeddings[idx_map[local_idx]] = emb.astype(np.float32)
        return embeddings


def _crop_patch(frame: np.ndarray, bbox: Iterable[float]) -> Optional[Image.Image]:
    if frame is None:
        return None
    h, w = frame.shape[:2]
    x1, y1, x2, y2 = map(int, bbox)
    x1 = max(0, min(w - 1, x1))
    x2 = max(x1 + 1, min(w, x2))
    y1 = max(0, min(h - 1, y1))
    y2 = max(y1 + 1, min(h, y2))
    patch = frame[y1:y2, x1:x2]
    if patch.size == 0:
        return None
    patch_rgb = patch[:, :, ::-1]  # BGR -> RGB
    return Image.fromarray(patch_rgb)
