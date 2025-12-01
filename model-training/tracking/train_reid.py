from __future__ import annotations

import argparse
from pathlib import Path

import yaml
from rich.console import Console

from modules import ReIDTrainerConfig, train_reid_model


def load_reid_config(path: Path) -> ReIDTrainerConfig:
    data = yaml.safe_load(path.read_text())
    dataset_block = data.get("reid_dataset", data)
    extractor_block = data.get("feature_extractor", {})
    head_block = data.get("embedding_head", {})
    detection_weights = (
        extractor_block.get("weights")
        or data.get("detection_weights")
        or dataset_block.get("weights")
    )
    if not detection_weights:
        raise ValueError("feature_extractor.weights must be set in the config.")

    return ReIDTrainerConfig(
        dataset_root=dataset_block["root"],
        train_split=dataset_block.get("train_split", "train"),
        val_split=dataset_block.get("val_split", "valid"),
        image_size=tuple(dataset_block.get("image_size", [256, 128])),
        batch_size=dataset_block.get("batch_size", 32),
        num_workers=dataset_block.get("num_workers", 8),
        epochs=dataset_block.get("epochs", 10),
        learning_rate=dataset_block.get("learning_rate", 1e-4),
        weight_decay=dataset_block.get("weight_decay", 5e-4),
        classifier_dropout=dataset_block.get("classifier_dropout", 0.2),
        label_smoothing=dataset_block.get("label_smoothing", 0.0),
        max_train_samples=dataset_block.get("max_train_samples"),
        max_val_samples=dataset_block.get("max_val_samples"),
        output_dir=dataset_block.get("output_dir", "outputs/reid"),
        device=dataset_block.get("device", "auto"),
        detection_weights=detection_weights,
        head_feature_index=extractor_block.get("head_feature_index", -1),
        feature_module_index=extractor_block.get("module_index"),
        freeze_detector=extractor_block.get("freeze", True),
        embedding_dim=head_block.get("embedding_dim", 512),
        embedding_hidden_dim=head_block.get("hidden_dim"),
        normalize_embeddings=head_block.get("normalize", True),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Train SoccerNet ReID embedder")
    parser.add_argument("--config", default="configs/reid.yaml", type=Path)
    parser.add_argument("--device", default=None, help="Override device (cpu/cuda/mps)")
    parser.add_argument("--output-dir", default=None, help="Optional override for output dir")
    args = parser.parse_args()

    console = Console()
    script_dir = Path(__file__).parent
    config_path = args.config
    if not config_path.is_absolute():
        config_path = (script_dir / config_path).resolve()
    console.log(f"Loading config {config_path}")
    cfg = load_reid_config(config_path)
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
