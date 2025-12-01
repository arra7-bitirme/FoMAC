from __future__ import annotations

import csv
import dataclasses
import inspect
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Sequence, Tuple

import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import transforms
from ultralytics import YOLO
from rich.console import Console

from .datasets import SoccerNetReIDDataset, build_pid_lookup

try:
    from torch.amp import autocast as torch_autocast
    _AMP_SUPPORTS_DEVICE_TYPE = True
except ImportError:  # torch < 2.0
    from torch.cuda.amp import autocast as torch_autocast
    _AMP_SUPPORTS_DEVICE_TYPE = False

try:
    from torch.amp import GradScaler as torch_grad_scaler
    _GRADSCALER_SUPPORTS_DEVICE_TYPE = "device_type" in inspect.signature(
        torch_grad_scaler.__init__
    ).parameters
except ImportError:
    from torch.cuda.amp import GradScaler as torch_grad_scaler
    _GRADSCALER_SUPPORTS_DEVICE_TYPE = False

autocast = torch_autocast
GradScaler = torch_grad_scaler


@dataclass
class ReIDTrainerConfig:
    dataset_root: str
    train_split: str = "train"
    val_split: str = "valid"
    image_size: Tuple[int, int] = (256, 128)
    batch_size: int = 32
    num_workers: int = 8
    epochs: int = 10
    learning_rate: float = 1e-4
    weight_decay: float = 5e-4
    classifier_dropout: float = 0.2
    label_smoothing: float = 0.0
    max_train_samples: Optional[int] = None
    max_val_samples: Optional[int] = None
    output_dir: str = "outputs/reid"
    device: str = "auto"
    detection_weights: str = ""
    head_feature_index: int = -1
    feature_module_index: Optional[int] = None
    freeze_detector: bool = True
    embedding_dim: int = 512
    embedding_hidden_dim: Optional[int] = None
    normalize_embeddings: bool = True

    def to_dict(self):
        return dataclasses.asdict(self)


def _resolve_device(device_cfg: str) -> torch.device:
    cfg = (device_cfg or "auto").lower()
    if cfg == "cpu":
        return torch.device("cpu")

    if cfg == "mps" and torch.backends.mps.is_available():
        return torch.device("mps")

    if torch.cuda.is_available():
        if cfg in {"cuda", "gpu", "auto"}:
            return torch.device("cuda:0")
        if cfg.startswith("cuda") or cfg.startswith("gpu"):
            suffix = cfg[4:] if cfg.startswith("cuda") else cfg[3:]
            if suffix.startswith(":"):
                suffix = suffix[1:]
            suffix = suffix.strip()
            if suffix.isdigit():
                return torch.device(f"cuda:{int(suffix)}")
            return torch.device("cuda:0")

    return torch.device("cpu")


class FrozenYOLOFeatureExtractor(nn.Module):
    """Wrap a pretrained YOLO detector and expose a frozen feature map."""

    def __init__(
        self,
        weights: str,
        device: torch.device,
        head_feature_index: int = -1,
        feature_module_index: Optional[int] = None,
        freeze: bool = True,
    ) -> None:
        super().__init__()
        if not weights:
            raise ValueError("Feature extractor weights path must be provided.")
        self.detector = YOLO(weights).model
        self.detector.to(device)
        if freeze:
            self.detector.requires_grad_(False)
        self.detector.eval()
        head = self.detector.model[-1]
        if feature_module_index is not None:
            self.embed_layer_idx = feature_module_index
        else:
            if not hasattr(head, "f"):
                raise ValueError("Detection head does not expose feature indices (f).")
            feature_sources = list(head.f)
            if not feature_sources:
                raise ValueError("Detection head feature list is empty.")
            idx = head_feature_index
            if idx < 0:
                idx = len(feature_sources) + idx
            if idx < 0 or idx >= len(feature_sources):
                raise ValueError(
                    f"Invalid head_feature_index={head_feature_index}. Available range: 0-{len(feature_sources)-1}."
                )
            self.embed_layer_idx = feature_sources[idx]
        self.device = device
        self._feature_dim: Optional[int] = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        embeddings = self.detector(x, embed=[self.embed_layer_idx])
        if isinstance(embeddings, tuple):
            embeddings = torch.stack(embeddings, dim=0)
        return embeddings

    def infer_output_dim(self, image_size: Tuple[int, int]) -> int:
        if self._feature_dim is not None:
            return self._feature_dim
        h, w = image_size
        dtype = next(self.detector.parameters()).dtype
        with torch.no_grad():
            dummy = torch.zeros(1, 3, h, w, device=self.device, dtype=dtype)
            feats = self.forward(dummy)
        self._feature_dim = int(feats.shape[1])
        return self._feature_dim


class YOLOReIDModel(nn.Module):
    """Frozen YOLO backbone with a lightweight ReID projection head."""

    def __init__(self, cfg: ReIDTrainerConfig, num_classes: int, device: torch.device) -> None:
        super().__init__()
        self.feature_extractor = FrozenYOLOFeatureExtractor(
            weights=cfg.detection_weights,
            device=device,
            head_feature_index=cfg.head_feature_index,
            feature_module_index=cfg.feature_module_index,
            freeze=cfg.freeze_detector,
        )
        feature_dim = self.feature_extractor.infer_output_dim(cfg.image_size)
        hidden_dim = cfg.embedding_hidden_dim or feature_dim
        layers: list[nn.Module] = []
        current_dim = feature_dim
        if hidden_dim and hidden_dim != feature_dim:
            layers.extend(
                [
                    nn.Linear(feature_dim, hidden_dim, bias=False),
                    nn.BatchNorm1d(hidden_dim),
                    nn.SiLU(),
                ]
            )
            current_dim = hidden_dim
        layers.append(nn.Dropout(cfg.classifier_dropout))
        layers.append(nn.Linear(current_dim, cfg.embedding_dim))
        self.embedding_head = nn.Sequential(*layers)
        self.classifier = nn.Linear(cfg.embedding_dim, num_classes)
        self.normalize_embeddings = cfg.normalize_embeddings

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        feats = self.feature_extractor(x)
        embeddings_raw = self.embedding_head(feats)
        embeddings = (
            F.normalize(embeddings_raw, p=2, dim=1)
            if self.normalize_embeddings
            else embeddings_raw
        )
        logits = self.classifier(embeddings_raw)
        return {"embeddings": embeddings, "logits": logits}


def _build_dataloaders(cfg: ReIDTrainerConfig):
    root = Path(cfg.dataset_root)
    pid_lookup = build_pid_lookup(root, [cfg.train_split, cfg.val_split])
    num_classes = len(pid_lookup)

    train_transform = transforms.Compose(
        [
            transforms.Resize(cfg.image_size),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
    eval_transform = transforms.Compose(
        [
            transforms.Resize(cfg.image_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    train_dataset = SoccerNetReIDDataset(
        root=root,
        split=cfg.train_split,
        pid_lookup=pid_lookup,
        transform=train_transform,
        max_samples=cfg.max_train_samples,
    )
    val_dataset = SoccerNetReIDDataset(
        root=root,
        split=cfg.val_split,
        pid_lookup=pid_lookup,
        transform=eval_transform,
        max_samples=cfg.max_val_samples,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=cfg.num_workers,
        pin_memory=True,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=cfg.num_workers,
        pin_memory=True,
    )
    return train_loader, val_loader, num_classes


def _build_model(cfg: ReIDTrainerConfig, num_classes: int, device: torch.device) -> nn.Module:
    return YOLOReIDModel(cfg, num_classes, device)


def _train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    scaler: GradScaler,
) -> Tuple[float, float]:
    model.train()
    running_loss = 0.0
    running_correct = 0
    total = 0
    for batch in loader:
        images = batch["image"].to(device, non_blocking=True)
        labels = batch["label"].to(device, non_blocking=True)

        optimizer.zero_grad()
        amp_enabled = device.type == "cuda"
        autocast_kwargs = {"enabled": amp_enabled}
        if _AMP_SUPPORTS_DEVICE_TYPE:
            autocast_kwargs["device_type"] = device.type if amp_enabled else "cpu"
        with autocast(**autocast_kwargs):
            outputs = model(images)
            logits = outputs["logits"]
            loss = criterion(logits, labels)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        running_loss += loss.item() * images.size(0)
        preds = logits.argmax(dim=1)
        running_correct += (preds == labels).sum().item()
        total += images.size(0)
    return running_loss / total, running_correct / total


@torch.no_grad()
def _evaluate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> Tuple[float, float]:
    model.eval()
    running_loss = 0.0
    running_correct = 0
    total = 0
    for batch in loader:
        images = batch["image"].to(device, non_blocking=True)
        labels = batch["label"].to(device, non_blocking=True)
        outputs = model(images)
        logits = outputs["logits"]
        loss = criterion(logits, labels)
        running_loss += loss.item() * images.size(0)
        preds = logits.argmax(dim=1)
        running_correct += (preds == labels).sum().item()
        total += images.size(0)
    return running_loss / total, running_correct / total


def train_reid_model(cfg: ReIDTrainerConfig) -> Path:
    device = _resolve_device(cfg.device)
    train_loader, val_loader, num_classes = _build_dataloaders(cfg)
    if not cfg.detection_weights:
        raise ValueError("detection_weights must be provided in the config.")
    model = _build_model(cfg=cfg, num_classes=num_classes, device=device)
    model.to(device)

    criterion = nn.CrossEntropyLoss(label_smoothing=cfg.label_smoothing)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=cfg.learning_rate, weight_decay=cfg.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg.epochs)
    scaler_kwargs = {"enabled": device.type == "cuda"}
    if _GRADSCALER_SUPPORTS_DEVICE_TYPE:
        scaler_kwargs["device_type"] = "cuda"
    scaler = GradScaler(**scaler_kwargs)

    output_dir = Path(cfg.output_dir)
    checkpoints_dir = output_dir / "checkpoints"
    checkpoints_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = output_dir / "metrics.csv"

    if not metrics_path.exists():
        with metrics_path.open("w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["epoch", "train_loss", "train_acc", "val_loss", "val_acc"])

    best_acc = 0.0
    best_path = checkpoints_dir / "best_reid.pt"
    console = Console()

    for epoch in range(1, cfg.epochs + 1):
        train_loss, train_acc = _train_one_epoch(
            model, train_loader, criterion, optimizer, device, scaler
        )
        val_loss, val_acc = _evaluate(model, val_loader, criterion, device)
        current_lr = optimizer.param_groups[0]["lr"]
        console.log(
            f"Epoch {epoch:02d}/{cfg.epochs} | lr={current_lr:.2e} | "
            f"train_loss={train_loss:.4f} train_acc={train_acc:.3f} | "
            f"val_loss={val_loss:.4f} val_acc={val_acc:.3f}"
        )
        scheduler.step()

        with metrics_path.open("a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([epoch, train_loss, train_acc, val_loss, val_acc])

        checkpoint_path = checkpoints_dir / f"epoch_{epoch:03d}.pt"
        torch.save({"epoch": epoch, "state_dict": model.state_dict()}, checkpoint_path)

        if val_acc >= best_acc:
            best_acc = val_acc
            torch.save(
                {
                    "epoch": epoch,
                    "state_dict": model.state_dict(),
                    "config": cfg.to_dict(),
                    "num_classes": num_classes,
                },
                best_path,
            )

    return best_path
