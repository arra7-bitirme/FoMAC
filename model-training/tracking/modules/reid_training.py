from __future__ import annotations

import csv
import dataclasses
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Sequence, Tuple

import torch
from torch import nn
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader
from torchvision import models, transforms

from .datasets import SoccerNetReIDDataset, build_pid_lookup


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

    def to_dict(self):
        return dataclasses.asdict(self)


def _resolve_device(device_cfg: str) -> torch.device:
    if device_cfg == "cpu":
        return torch.device("cpu")
    if torch.cuda.is_available() and device_cfg in {"cuda", "gpu", "auto"}:
        return torch.device("cuda")
    if torch.backends.mps.is_available() and device_cfg == "mps":
        return torch.device("mps")
    return torch.device("cpu")


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


def _build_model(num_classes: int, dropout: float) -> nn.Module:
    backbone = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
    in_features = backbone.fc.in_features
    backbone.fc = nn.Sequential(
        nn.Dropout(dropout),
        nn.Linear(in_features, num_classes),
    )
    return backbone


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
        with autocast(enabled=device.type == "cuda"):
            logits = model(images)
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
        logits = model(images)
        loss = criterion(logits, labels)
        running_loss += loss.item() * images.size(0)
        preds = logits.argmax(dim=1)
        running_correct += (preds == labels).sum().item()
        total += images.size(0)
    return running_loss / total, running_correct / total


def train_reid_model(cfg: ReIDTrainerConfig) -> Path:
    device = _resolve_device(cfg.device)
    train_loader, val_loader, num_classes = _build_dataloaders(cfg)
    model = _build_model(num_classes=num_classes, dropout=cfg.classifier_dropout)
    model.to(device)

    criterion = nn.CrossEntropyLoss(label_smoothing=cfg.label_smoothing)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=cfg.learning_rate, weight_decay=cfg.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg.epochs)
    scaler = GradScaler(enabled=device.type == "cuda")

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

    for epoch in range(1, cfg.epochs + 1):
        train_loss, train_acc = _train_one_epoch(
            model, train_loader, criterion, optimizer, device, scaler
        )
        val_loss, val_acc = _evaluate(model, val_loader, criterion, device)
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
