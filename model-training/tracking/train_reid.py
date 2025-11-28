from __future__ import annotations

import argparse
from pathlib import Path

import yaml
from rich.console import Console

from modules import ReIDTrainerConfig, train_reid_model


def load_reid_config(path: Path) -> ReIDTrainerConfig:
    data = yaml.safe_load(path.read_text())
    block = data.get("reid_dataset", data)
    return ReIDTrainerConfig(
        dataset_root=block["root"],
        train_split=block.get("train_split", "train"),
        val_split=block.get("val_split", "valid"),
        image_size=tuple(block.get("image_size", [256, 128])),
        batch_size=block.get("batch_size", 32),
        num_workers=block.get("num_workers", 8),
        epochs=block.get("epochs", 10),
        learning_rate=block.get("learning_rate", 1e-4),
        weight_decay=block.get("weight_decay", 5e-4),
        classifier_dropout=block.get("classifier_dropout", 0.2),
        label_smoothing=block.get("label_smoothing", 0.0),
        max_train_samples=block.get("max_train_samples"),
        max_val_samples=block.get("max_val_samples"),
        output_dir=block.get("output_dir", "outputs/reid"),
        device=block.get("device", "auto"),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Train SoccerNet ReID embedder")
    parser.add_argument("--config", default="configs/reid.yaml", type=Path)
    parser.add_argument("--device", default=None, help="Override device (cpu/cuda/mps)")
    parser.add_argument("--output-dir", default=None, help="Optional override for output dir")
    args = parser.parse_args()

    console = Console()
    console.log(f"Loading config {args.config}")
    cfg = load_reid_config(args.config)
    if args.device:
        cfg.device = args.device
    if args.output_dir:
        cfg.output_dir = args.output_dir

    console.log(
        f"Dataset root: {cfg.dataset_root} | epochs: {cfg.epochs} | batch: {cfg.batch_size}"
    )
    best_path = train_reid_model(cfg)
    console.log(f"Training finished. Best checkpoint: {best_path}")


if __name__ == "__main__":
    main()
