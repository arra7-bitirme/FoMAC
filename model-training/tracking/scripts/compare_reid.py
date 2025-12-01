from __future__ import annotations

import argparse
import dataclasses
from pathlib import Path

import motmetrics as mm

from modules import TrackerRunner, TrackingConfig


def run_tracking(cfg: TrackingConfig, use_reid: bool) -> Path:
    cfg_copy = dataclasses.replace(cfg)
    reid_cfg = dict(cfg_copy.reid_config or {})
    reid_cfg["enabled"] = use_reid
    cfg_copy.reid_config = reid_cfg
    runner = TrackerRunner(cfg_copy)
    runner.run()
    output_dir = Path(cfg_copy.output_dir)
    sequence = cfg_copy.sequence_names[0]
    return output_dir / f"{sequence}.txt"


def mot_summary(gt_path: Path, pred_path: Path, name: str) -> dict:
    gt = mm.io.loadtxt(gt_path, fmt="mot15-02", min_confidence=-1)
    pred = mm.io.loadtxt(pred_path, fmt="mot15-02")
    acc = mm.utils.compare_to_groundtruth(gt, pred, "iou", distth=0.5)
    mh = mm.metrics.create()
    metrics = ["num_frames", "idf1", "idp", "idr", "mts", "mtn", "ids"]
    summary = mh.compute(acc, metrics=metrics, name=name, overall=True)
    return summary.to_dict('index')[name]


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare tracking with/without ReID")
    parser.add_argument("--config", default="configs/tracking.yaml", type=Path)
    parser.add_argument("--sequence", required=True, help="Sequence name to evaluate")
    parser.add_argument("--gt", required=True, type=Path, help="Ground-truth MOT file (gt/gt.txt)")
    parser.add_argument("--output-dir", type=Path, default=None, help="Overrides output directory")
    args = parser.parse_args()

    cfg = TrackingConfig.from_yaml(args.config)
    cfg.sequence_names = [args.sequence]
    if args.output_dir:
        cfg.output_dir = str(args.output_dir)

    baseline_path = run_tracking(cfg, use_reid=False)
    reid_path = run_tracking(cfg, use_reid=True)

    baseline_metrics = mot_summary(args.gt, baseline_path, name="baseline")
    reid_metrics = mot_summary(args.gt, reid_path, name="reid")

    print("Baseline:")
    for key, value in baseline_metrics.items():
        print(f"  {key}: {value}")
    print("\nReID enabled:")
    for key, value in reid_metrics.items():
        print(f"  {key}: {value}")


if __name__ == "__main__":
    main()
