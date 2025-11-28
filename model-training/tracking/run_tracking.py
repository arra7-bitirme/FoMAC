from __future__ import annotations

import argparse
from pathlib import Path

from rich.console import Console

from modules import TrackerRunner, TrackingConfig


def main() -> None:
    parser = argparse.ArgumentParser(description="Run YOLO + StrongSORT tracking pipeline")
    parser.add_argument("--config", default="configs/tracking.yaml", type=Path)
    parser.add_argument(
        "--sequence",
        action="append",
        default=None,
        help="Only run the specified sequence (can be repeated)",
    )
    parser.add_argument("--device", default=None, help="Override detector device")
    parser.add_argument("--limit-frames", type=int, default=None)
    parser.add_argument("--weights", type=str, default=None, help="Override detector weights")
    args = parser.parse_args()

    console = Console()
    console.log(f"Loading tracking config {args.config}")
    cfg = TrackingConfig.from_yaml(args.config)

    if args.sequence:
        cfg.sequence_names = args.sequence
    if args.device:
        cfg.device = args.device
    if args.limit_frames is not None:
        cfg.limit_frames = args.limit_frames
    if args.weights:
        cfg.detection_weights = args.weights

    runner = TrackerRunner(cfg)
    runner.run()


if __name__ == "__main__":
    main()
