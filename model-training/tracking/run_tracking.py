from __future__ import annotations

import argparse
import os
from pathlib import Path

os.environ.setdefault("PYTORCH_NVML_BASED_CUDA_CHECK", "1")

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
    parser.add_argument("--imgsz", type=int, default=None, help="Override inference resolution")
    parser.add_argument("--conf", type=float, default=None, help="Override detection confidence threshold")
    parser.add_argument("--iou", type=float, default=None, help="Override detection IoU threshold")
    parser.add_argument("--limit-frames", type=int, default=None)
    parser.add_argument("--vid-stride", type=int, default=None, help="Process every Nth frame")
    parser.add_argument("--weights", type=str, default=None, help="Override detector weights")
    parser.add_argument("--player-classes", type=int, nargs="+", help="Player/goalkeeper class ids")
    parser.add_argument("--referee-classes", type=int, nargs="+", help="Referee class ids")
    parser.add_argument("--ball-classes", type=int, nargs="+", help="Ball class ids")
    parser.add_argument("--save-visuals", action="store_true", help="Force visualization saving")
    parser.add_argument("--no-visuals", action="store_true", help="Disable visualization writing")
    parser.add_argument("--no-csv", action="store_true", help="Skip CSV output")
    parser.add_argument("--csv-template", type=str, default=None, help="Override CSV filename template")
    parser.add_argument("--team-color", action="store_true", help="Enable color-based team clustering")
    parser.add_argument("--no-team", action="store_true", help="Disable team classification")
    parser.add_argument("--team-method", choices=["color", "model"], help="Override team classification method")
    parser.add_argument("--team-samples", type=int, help="Samples per track for team clustering")
    parser.add_argument("--team-min-hits", type=int, help="Minimum hits per track before clustering")
    parser.add_argument("--enable-reid", action="store_true", help="Enable ReID embeddings during association")
    parser.add_argument("--disable-reid", action="store_true", help="Disable ReID even if config enables it")
    parser.add_argument("--reid-weights", type=str, help="Override ReID checkpoint path")
    parser.add_argument("--reid-alpha", type=float, help="Weight for IoU term when combining costs")
    parser.add_argument("--reid-beta", type=float, help="Weight for embedding distance term")
    parser.add_argument("--reid-max-distance", type=float, help="Maximum embedding distance for valid matches")
    parser.add_argument("--video", action="append", help="Path to standalone video file (can repeat)")
    parser.add_argument("--video-name", action="append", help="Optional custom name for each --video entry")
    args = parser.parse_args()

    config_path = args.config
    if not config_path.is_absolute():
        config_path = (Path(__file__).resolve().parent / config_path).resolve()

    console = Console()
    console.log(f"Loading tracking config {config_path}")
    cfg = TrackingConfig.from_yaml(config_path)

    if args.sequence:
        cfg.sequence_names = args.sequence
    if args.device:
        cfg.device = args.device
    if args.imgsz:
        cfg.imgsz = args.imgsz
    if args.conf is not None:
        cfg.detection_conf = args.conf
    if args.iou is not None:
        cfg.detection_iou = args.iou
    if args.limit_frames is not None:
        cfg.limit_frames = args.limit_frames
    if args.vid_stride is not None:
        cfg.vid_stride = args.vid_stride
    if args.weights:
        cfg.detection_weights = args.weights
    if args.player_classes:
        cfg.player_classes = tuple(args.player_classes)
        cfg.goalkeeper_classes = tuple(args.player_classes)
    if args.referee_classes:
        cfg.referee_classes = tuple(args.referee_classes)
    if args.ball_classes:
        cfg.ball_classes = tuple(args.ball_classes)
    if args.save_visuals:
        cfg.save_visualizations = True
    if args.no_visuals:
        cfg.save_visualizations = False
    if args.no_csv:
        cfg.output_csv = False
    if args.csv_template:
        cfg.output_csv_template = args.csv_template

    team_cfg = dict(cfg.team_classification or {})
    if args.no_team:
        team_cfg["enabled"] = False
    if args.team_color:
        team_cfg["enabled"] = True
        team_cfg["method"] = "color"
    if args.team_method:
        team_cfg["enabled"] = True
        team_cfg["method"] = args.team_method
    if args.team_samples is not None:
        team_cfg["samples_per_track"] = args.team_samples
    if args.team_min_hits is not None:
        team_cfg["min_track_hits"] = args.team_min_hits
    cfg.team_classification = team_cfg

    reid_cfg = dict(cfg.reid_config or {})
    if args.disable_reid:
        reid_cfg["enabled"] = False
    if args.enable_reid:
        reid_cfg["enabled"] = True
    if args.reid_weights:
        reid_cfg["weights"] = args.reid_weights
    if args.reid_alpha is not None:
        reid_cfg["alpha"] = args.reid_alpha
    if args.reid_beta is not None:
        reid_cfg["beta"] = args.reid_beta
    if args.reid_max_distance is not None:
        reid_cfg["max_distance"] = args.reid_max_distance
    cfg.reid_config = reid_cfg

    if args.video:
        cfg.video_sources = tuple(str(Path(video).resolve()) for video in args.video)
    if args.video_name:
        cfg.video_names = tuple(args.video_name)

    runner = TrackerRunner(cfg)
    try:
        runner.run()
    except SystemExit as exc:
        console.print("[red]SystemExit triggered during tracking run[/red]")
        console.print_exception()
        raise
    except Exception:
        console.print_exception()
        raise


if __name__ == "__main__":
    main()
