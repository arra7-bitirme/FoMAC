#!/usr/bin/env python3
"""Phase-aware trainer that runs the pipeline twice with dedicated configs."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List

from main import YOLOTrainingPipeline

PHASE_CONFIGS: Dict[str, Path] = {
    "phase1": Path("configs/phases/phase1"),
    "phase2": Path("configs/phases/phase2"),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run sequential two-phase training with dedicated configs",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--phases",
        nargs="+",
        choices=list(PHASE_CONFIGS.keys()),
        default=list(PHASE_CONFIGS.keys()),
        help="Subset of phases to run in order",
    )
    parser.add_argument(
        "--train-only",
        action="store_true",
        help="Skip extraction and only train for selected phases",
    )
    parser.add_argument(
        "--extract-only",
        action="store_true",
        help="Only extract datasets for selected phases",
    )
    parser.add_argument(
        "--log-level",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        default="INFO",
        help="Logger level forwarded to each pipeline run",
    )
    return parser.parse_args()


def build_phase_args(
    config_dir: Path,
    base_args: argparse.Namespace,
) -> List[str]:
    args: List[str] = [
        "--config-dir",
        str(config_dir),
        "--log-level",
        base_args.log_level,
    ]
    if base_args.train_only:
        args.append("--train-only")
    elif base_args.extract_only:
        args.append("--extract-only")
    return args


def run_phase(phase_name: str, cli_args: List[str]):
    print(f"\n===== Running {phase_name.upper()} =====")
    pipeline = YOLOTrainingPipeline()
    pipeline.run(cli_args)


def main():
    args = parse_args()

    for phase_name in args.phases:
        config_dir = PHASE_CONFIGS[phase_name]
        if not config_dir.exists():
            raise FileNotFoundError(
                f"Missing config directory for {phase_name}: {config_dir}"
            )
        phase_cli_args = build_phase_args(config_dir, args)
        run_phase(phase_name, phase_cli_args)


if __name__ == "__main__":
    main()
