from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Set, Tuple

import cv2
import numpy as np
import torch
import yaml
from rich.console import Console
from rich.progress import BarColumn, Progress, SpinnerColumn, TextColumn, TimeElapsedColumn
from ultralytics import YOLO

from .class_aware_tracker import BallTracker, IoUTracker, Tracklet
from .detector_stream import iter_detections
from .datasets import SNMOTDataset, SNMOTSequence
from .visualization import draw_tracks, save_visual_frame


class StageProgressAdapter:
    """Helper to create short-lived Rich tasks for long sub-steps."""

    def __init__(self, progress: Optional[Progress]) -> None:
        self._progress = progress

    def start_task(
        self,
        label: str,
        total: Optional[int] = None,
        *,
        color: str = "magenta",
    ) -> Optional[int]:
        if self._progress is None:
            return None
        description = f"[{color}]{label}[/{color}]"
        return self._progress.add_task(description, total=total)

    def advance_task(self, task_id: Optional[int], advance: float = 1.0) -> None:
        if self._progress is None or task_id is None:
            return
        try:
            self._progress.advance(task_id, advance)
        except Exception:
            pass

    def close_task(self, task_id: Optional[int]) -> None:
        if self._progress is None or task_id is None:
            return
        try:
            self._progress.remove_task(task_id)
        except Exception:
            pass


@dataclass
class TrackingConfig:
    detection_weights: str
    dataset_root: Optional[str] = None
    tracker_yaml: str = ""
    sequence_names: Optional[Sequence[str]] = None
    split_dirs: Optional[Sequence[str]] = None
    detection_conf: float = 0.3
    detection_iou: float = 0.5
    imgsz: int = 1280
    player_classes: Tuple[int, ...] = (0,)
    goalkeeper_classes: Tuple[int, ...] = (0,)
    referee_classes: Tuple[int, ...] = (3,)
    ball_classes: Tuple[int, ...] = (1,)
    device: str = "auto"
    save_visualizations: bool = False
    output_dir: str = "outputs/tracks"
    limit_frames: Optional[int] = None
    reid_weights: Optional[str] = None
    reid_config: Optional[Dict[str, object]] = None
    half: bool = False
    vid_stride: int = 1
    tracker_players: Optional[Dict[str, float]] = None
    tracker_referees: Optional[Dict[str, float]] = None
    tracker_ball: Optional[Dict[str, float]] = None
    association: Optional[Dict[str, float]] = None
    output_csv: bool = True
    output_csv_template: str = "{sequence}.csv"
    output_csv_precision: int = 4
    write_video: bool = False
    show_progress: bool = True
    team_classification: Optional[Dict[str, object]] = None
    video_sources: Optional[Sequence[str]] = None
    video_names: Optional[Sequence[str]] = None
    appearance_embedding: Optional[Dict[str, object]] = None
    use_dataset_sequences: bool = True

    @classmethod
    def from_yaml(cls, path: Path | str) -> "TrackingConfig":
        cfg_path = Path(path)
        data = yaml.safe_load(cfg_path.read_text())
        base_dir = cfg_path.parent

        def _resolve(value: Optional[str]) -> Optional[str]:
            if not value:
                return value
            value_path = Path(value)
            if not value_path.is_absolute():
                value_path = (base_dir / value_path).resolve()
            return str(value_path)

        detection = data.get("detection", {})
        tracking = data.get("tracking", {})
        datasets = data.get("datasets", {})
        videos = data.get("videos", {})
        output = data.get("output", {})
        team_cls = data.get("team_classification")
        reid_cfg = tracking.get("reid")
        appearance_cfg = tracking.get("appearance_embedding")
        classes = detection.get("classes", {})
        use_dataset_sequences = bool(datasets.get("enabled", True))

        def _resolve_seq(values: Optional[Sequence[str]]) -> Optional[Tuple[str, ...]]:
            if not values:
                return None
            resolved: List[str] = []
            for value in values:
                resolved.append(_resolve(value) or value)
            return tuple(resolved)

        video_sources = _resolve_seq(videos.get("sources"))
        video_names = tuple(videos.get("names", [])) if videos.get("names") else None

        dataset_root_value = datasets.get("root")
        dataset_root = _resolve(dataset_root_value) if dataset_root_value else None
        if use_dataset_sequences and not dataset_root:
            raise KeyError("datasets.root must be provided when dataset tracking is enabled")

        return cls(
            detection_weights=_resolve(detection.get("weights")) or detection["weights"],
            detection_conf=detection.get("conf", 0.3),
            detection_iou=detection.get("iou", 0.5),
            imgsz=detection.get("imgsz", 1280),
            player_classes=tuple(classes.get("players", [0])),
            goalkeeper_classes=tuple(classes.get("goalkeepers", [0])),
            referee_classes=tuple(classes.get("referees", [3])),
            ball_classes=tuple(classes.get("ball", [1])),
            tracker_yaml=_resolve(tracking.get("tracker_yaml")) or tracking.get("tracker_yaml", ""),
            reid_weights=_resolve(tracking.get("reid_weights")),
            device=tracking.get("device", "auto"),
            save_visualizations=tracking.get("save_visualizations", False),
            half=tracking.get("half", False),
            vid_stride=tracking.get("vid_stride", 1),
            tracker_players=tracking.get("players"),
            tracker_referees=tracking.get("referees"),
            tracker_ball=tracking.get("ball"),
            association=tracking.get("association"),
            dataset_root=dataset_root,
            sequence_names=datasets.get("sequences"),
            split_dirs=datasets.get("splits"),
            output_dir=_resolve(datasets.get("output_dir")) or datasets.get("output_dir", "outputs/tracks"),
            limit_frames=datasets.get("limit_frames"),
            output_csv=output.get("write_csv", True),
            output_csv_template=output.get("csv_name_template", "{sequence}.csv"),
            output_csv_precision=output.get("csv_precision", 4),
            write_video=output.get("write_video", False),
            show_progress=tracking.get("progress", True),
            team_classification=team_cls,
            reid_config=reid_cfg,
            video_sources=video_sources,
            video_names=video_names,
            appearance_embedding=appearance_cfg,
            use_dataset_sequences=use_dataset_sequences,
        )


class TrackerRunner:
    def __init__(self, cfg: TrackingConfig) -> None:
        self.cfg = cfg
        self.console = Console()
        self.cfg.device = self._normalize_device(self.cfg.device, component="Detector")
        self.output_dir = Path(cfg.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.model = YOLO(cfg.detection_weights)
        self.dataset = None
        self.sequences: List[SNMOTSequence] = []
        if self.cfg.use_dataset_sequences:
            if not self.cfg.dataset_root:
                raise RuntimeError("datasets.root is required when dataset tracking is enabled")
            dataset_root = Path(self.cfg.dataset_root)
            try:
                self.dataset = SNMOTDataset(
                    dataset_root,
                    cfg.sequence_names,
                    cfg.split_dirs,
                )
                self.sequences.extend(self.dataset.sequences)
            except (FileNotFoundError, RuntimeError) as exc:
                if not cfg.video_sources:
                    raise
                self.console.print(f"[yellow]Skipping dataset sequences: {exc}[/yellow]")

        self.video_sequences = self._build_video_sequences()
        self.sequences.extend(self.video_sequences)

        if not self.sequences:
            raise RuntimeError("No dataset sequences or video sources available for tracking.")

        self.player_classes = set(cfg.player_classes) | set(cfg.goalkeeper_classes)
        self.referee_classes = set(cfg.referee_classes)
        self.ball_classes = set(cfg.ball_classes)
        self.reid_settings = cfg.reid_config or {"enabled": False}
        self.reid_enabled = bool(self.reid_settings.get("enabled", False))
        if self.reid_enabled:
            normalized_reid_device = self._normalize_device(
                self.reid_settings.get("device", self.cfg.device),
                component="ReID",
            )
            self.reid_settings["device"] = normalized_reid_device
        self.reid_target_classes = self._resolve_reid_targets()

        self.appearance_cfg = self.cfg.appearance_embedding or {}
        self.simple_embedding_enabled = bool(self.appearance_cfg.get("enabled", False))
        self.simple_embedding_targets = self._resolve_embedding_targets(
            self.appearance_cfg.get("target_classes", "players")
        )
        self.simple_embedding_space = str(self.appearance_cfg.get("color_space", "lab")).lower()
        self.simple_embedding_bins = int(self.appearance_cfg.get("bins", 16))
        default_assoc = self.cfg.association or {}
        default_alpha = float(default_assoc.get("iou_weight", 0.6))
        self.simple_embedding_alpha = float(self.appearance_cfg.get("alpha", default_alpha))
        self.simple_beta = float(self.appearance_cfg.get("beta", 0.35))
        self.simple_max_distance = float(self.appearance_cfg.get("max_distance", 0.45))

        player_assoc = self._build_association_params(target="players")
        referee_assoc = self._build_association_params(target="referees")

        self.player_tracker = IoUTracker(
            class_ids=tuple(self.player_classes),
            max_age=int((cfg.tracker_players or {}).get("max_age", 32)),
            min_hits=int((cfg.tracker_players or {}).get("min_hits", 3)),
            iou_threshold=float((cfg.tracker_players or {}).get("iou_threshold", 0.3)),
            start_id=1,
            association=player_assoc,
        )
        self.ref_tracker = IoUTracker(
            class_ids=tuple(self.referee_classes),
            max_age=int((cfg.tracker_referees or {}).get("max_age", 32)),
            min_hits=int((cfg.tracker_referees or {}).get("min_hits", 3)),
            iou_threshold=float((cfg.tracker_referees or {}).get("iou_threshold", 0.3)),
            start_id=4000,
            association=referee_assoc,
        )
        self.ball_tracker = BallTracker(
            class_ids=tuple(self.ball_classes),
            max_age=int((cfg.tracker_ball or {}).get("max_age", 12)),
            start_id=9000,
        )

        self.reid_extractor = self._build_reid_extractor()

        self.class_colors = self._build_class_colors()
        self.team_settings = self.cfg.team_classification or {"enabled": False}
        self.team_palette = {
            0: (60, 20, 220),  # red-ish
            1: (0, 215, 255),  # amber
            2: (0, 140, 0),    # green
            3: (128, 0, 255),  # violet
        }

    def run(self) -> None:
        total_sequences = len(self.sequences)
        self.console.print(
            f"[bold green]Tracking {total_sequences} sequences using {self.cfg.detection_weights}[/bold green]"
        )
        for sequence in self.sequences:
            self._run_sequence(sequence)

    def _build_video_sequences(self) -> List[SNMOTSequence]:
        sequences: List[SNMOTSequence] = []
        sources = list(self.cfg.video_sources or [])
        names = list(self.cfg.video_names or [])
        for idx, video in enumerate(sources):
            video_path = Path(video)
            if not video_path.exists():
                raise FileNotFoundError(f"Video source not found: {video_path}")
            sequence_name = names[idx] if idx < len(names) and names[idx] else video_path.stem
            fps_value = 25.0
            cap = cv2.VideoCapture(str(video_path))
            if cap.isOpened():
                fps_guess = cap.get(cv2.CAP_PROP_FPS)
                if fps_guess and fps_guess > 1e-2:
                    fps_value = float(fps_guess)
                cap.release()
            sequences.append(
                SNMOTSequence(
                    name=sequence_name,
                    root=video_path.parent,
                    img_dir=video_path,
                    seqinfo={"imdir": str(video_path), "framerate": str(fps_value)},
                    gameinfo={},
                )
            )
        return sequences

    def _create_video_writer(self, path: Path, width: int, height: int, fps: float) -> cv2.VideoWriter:
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        return cv2.VideoWriter(str(path), fourcc, float(max(fps, 1.0)), (int(width), int(height)))

    def _run_sequence(self, sequence: SNMOTSequence) -> None:
        self.console.print(f"\n[cyan]→ Sequence:[/cyan] {sequence.name}")
        mot_rows: List[List[float]] = []
        track_records: List[Dict[str, float]] = []
        sequence_dir = self.output_dir / sequence.name
        sequence_dir.mkdir(parents=True, exist_ok=True)
        csv_path = sequence_dir / self.cfg.output_csv_template.format(sequence=sequence.name)
        mot_path = sequence_dir / f"{sequence.name}.txt"
        frame_tracks: Dict[int, List[Tracklet]] = {}
        collect_frames = self.cfg.save_visualizations or self.cfg.write_video
        last_frame_idx = 0
        progress = None
        progress_task: Optional[int] = None
        stage_task: Optional[int] = None
        stage_labels = ["Tracking frames"]
        if self.team_settings.get("enabled"):
            stage_labels.append("Team classification")
        if self.cfg.output_csv:
            stage_labels.append("Saving CSV")
        if collect_frames:
            stage_labels.append("Rendering visuals")
        stage_labels.append("Completed")
        stage_total = max(len(stage_labels) - 1, 1)
        stage_index = 0
        if self.cfg.show_progress:
            total_frames = self._estimate_total_frames(sequence)
            progress = self._create_progress()
            progress.start()
            progress_task = progress.add_task(
                f"[cyan]{sequence.name}[/cyan]",
                total=total_frames,
            )
            stage_task = progress.add_task(
                f"[yellow]{stage_labels[0]}[/yellow]",
                total=stage_total,
            )
        detail_progress = StageProgressAdapter(progress) if progress is not None else None

        if not sequence.img_dir.exists():
            raise FileNotFoundError(
                f"Image directory not found for {sequence.name}: {sequence.img_dir}"
            )

        frame_iter = iter_detections(
            model=self.model,
            source=str(sequence.img_dir),
            imgsz=self.cfg.imgsz,
            device=self.cfg.device,
            conf=self.cfg.detection_conf,
            iou=self.cfg.detection_iou,
            half=self.cfg.half,
            vid_stride=self.cfg.vid_stride,
            limit_frames=self.cfg.limit_frames,
        )

        tracking_completed = False
        try:
            for frame_data in frame_iter:
                last_frame_idx = frame_data.frame_idx
                if progress_task is not None and progress is not None:
                    progress.advance(progress_task)
                detections = self._extract_detections(frame_data)
                if not detections:
                    continue

                player_dets = [det for det in detections if det["class_id"] in self.player_classes]
                referee_dets = [det for det in detections if det["class_id"] in self.referee_classes]
                ball_dets = [det for det in detections if det["class_id"] in self.ball_classes]

                tracklets: List[Tracklet] = []
                tracklets.extend(self.player_tracker.update(player_dets, frame_data.frame_idx))
                tracklets.extend(self.ref_tracker.update(referee_dets, frame_data.frame_idx))
                tracklets.extend(self.ball_tracker.update(ball_dets, frame_data.frame_idx))

                if not tracklets:
                    continue

                if collect_frames:
                    frame_tracks.setdefault(frame_data.frame_idx, [])
                    for track in tracklets:
                        frame_tracks[frame_data.frame_idx].append(
                            Tracklet(
                                track_id=track.track_id,
                                class_id=track.class_id,
                                bbox=track.bbox.copy(),
                                score=track.score,
                                frame_idx=track.frame_idx,
                                hits=track.hits,
                            )
                        )

                for track in tracklets:
                    mot_rows.append(self._format_mot_row(track))
                    if self.cfg.output_csv:
                        track_records.append(self._track_to_record(track))
            tracking_completed = True
        except Exception:
            self.console.print_exception()
            raise
        finally:
            if tracking_completed:
                stage_index = self._advance_stage(progress, stage_task, stage_labels, stage_index)
        self._write_mot(mot_path, mot_rows)
        team_map: Dict[int, int] = {}
        if self.team_settings.get("enabled"):
            team_map = self._assign_team_ids(sequence, track_records, detail_progress)
            stage_index = self._advance_stage(progress, stage_task, stage_labels, stage_index)
        if self.cfg.output_csv:
            csv_rows = [self._record_to_row(record, team_map) for record in track_records]
            self._write_csv(csv_path, csv_rows, detail_progress)
            stage_index = self._advance_stage(progress, stage_task, stage_labels, stage_index)

        self.console.print(
            f"[green]✔ Saved[/green] {mot_path} ({len(mot_rows)} detections)"
        )
        if self.cfg.output_csv:
            self.console.print(f"[green]✔ Saved[/green] {csv_path}")
        if collect_frames and frame_tracks:
            video_path = self._render_visuals(
                sequence,
                frame_tracks,
                team_map,
                sequence_dir,
                last_frame_idx,
                progress if self.cfg.show_progress else None,
            )
            if video_path:
                self.console.print(f"[green]✔ Saved[/green] {video_path}")
            stage_index = self._advance_stage(progress, stage_task, stage_labels, stage_index)

        if progress is not None:
            progress.stop()

    def _extract_detections(self, frame_data) -> List[Dict[str, float]]:
        boxes = getattr(frame_data.result, "boxes", None)
        if boxes is None or boxes.xyxy is None:
            return []
        xyxy = boxes.xyxy.cpu().numpy()
        confs = boxes.conf.cpu().numpy()
        classes = boxes.cls.int().cpu().tolist()
        embeddings = self._maybe_compute_embeddings(frame_data.frame, xyxy, classes)
        detections: List[Dict[str, float]] = []
        for idx in range(len(classes)):
            detections.append(
                {
                    "bbox": xyxy[idx],
                    "score": float(confs[idx]),
                    "class_id": int(classes[idx]),
                    "embedding": embeddings[idx] if embeddings else None,
                }
            )
        return detections

    def _format_mot_row(self, track: Tracklet) -> List[float]:
        x, y, w, h = track.xywh
        return [
            track.frame_idx,
            track.track_id,
            round(float(x), 2),
            round(float(y), 2),
            round(float(w), 2),
            round(float(h), 2),
            round(float(track.score), 4),
            -1,
            -1,
            -1,
        ]

    def _track_to_record(self, track: Tracklet) -> Dict[str, float]:
        x, y, w, h = track.xywh
        p = self.cfg.output_csv_precision
        return {
            "frame": track.frame_idx,
            "track_id": track.track_id,
            "class_id": track.class_id,
            "x": round(float(x), p),
            "y": round(float(y), p),
            "w": round(float(w), p),
            "h": round(float(h), p),
            "score": round(float(track.score), p),
        }

    def _record_to_row(self, record: Dict[str, float], team_map: Dict[int, int]) -> List[float]:
        team_id = int(team_map.get(record["track_id"], -1))
        return [
            record["frame"],
            record["track_id"],
            record["class_id"],
            record["x"],
            record["y"],
            record["w"],
            record["h"],
            record["score"],
            team_id,
        ]

    def _write_mot(self, path: Path, rows: List[List[float]]) -> None:
        with path.open("w", newline="") as f:
            writer = csv.writer(f)
            writer.writerows(rows)

    def _write_csv(
        self,
        path: Path,
        rows: List[List[float]],
        progress_helper: Optional[StageProgressAdapter] = None,
    ) -> None:
        task_id: Optional[int] = None
        if progress_helper is not None and rows:
            task_id = progress_helper.start_task(
                "Saving CSV rows",
                total=len(rows),
                color="green",
            )
        with path.open("w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["frame", "track_id", "class_id", "x", "y", "w", "h", "score", "team_id"])
            for row in rows:
                writer.writerow(row)
                if progress_helper is not None:
                    progress_helper.advance_task(task_id)
        if progress_helper is not None:
            progress_helper.close_task(task_id)

    def _render_visuals(
        self,
        sequence: SNMOTSequence,
        frame_tracks: Dict[int, List[Tracklet]],
        team_map: Dict[int, int],
        sequence_dir: Path,
        last_frame_idx: int,
        progress: Optional[Progress] = None,
    ) -> Optional[Path]:
        if not (self.cfg.save_visualizations or self.cfg.write_video):
            return None
        if not frame_tracks or last_frame_idx == 0:
            return None

        visual_dir = sequence_dir / "visualizations"
        video_dir = sequence_dir / "video"
        video_path = video_dir / f"{sequence.name}_annotated.mp4"
        video_writer = None

        if self.cfg.write_video:
            video_dir.mkdir(parents=True, exist_ok=True)

        source_is_video = sequence.img_dir.is_file()
        cap = None
        if source_is_video:
            cap = cv2.VideoCapture(str(sequence.img_dir))
            if not cap.isOpened():
                self.console.print(
                    f"[yellow]Unable to reopen video source for {sequence.name}, skipping video render[/yellow]"
                )
                return None

        img_ext = sequence.seqinfo.get("imext") or sequence.seqinfo.get("imExt") or ".jpg"
        if not str(img_ext).startswith("."):
            img_ext = f".{img_ext}"

        render_task: Optional[int] = None
        if progress is not None and last_frame_idx > 0:
            render_task = progress.add_task(
                "[blue]Rendering visuals[/blue]",
                total=last_frame_idx,
            )

        for frame_idx in range(1, last_frame_idx + 1):
            frame = self._fetch_frame(sequence, frame_idx, img_ext, cap)
            if frame is None:
                continue
            tracklets = frame_tracks.get(frame_idx, [])
            if tracklets:
                annotated = draw_tracks(
                    frame.copy(),
                    tracklets,
                    self.class_colors,
                    team_assignments=team_map,
                    team_colors=self.team_palette,
                )
            else:
                annotated = frame
            if self.cfg.save_visualizations:
                save_visual_frame(visual_dir, frame_idx, annotated)
            if self.cfg.write_video:
                if video_writer is None:
                    height, width = annotated.shape[:2]
                    video_writer = self._create_video_writer(video_path, width, height, sequence.frame_rate)
                video_writer.write(annotated)
            if render_task is not None and progress is not None:
                progress.advance(render_task)

        if cap is not None:
            cap.release()
        if video_writer is not None:
            video_writer.release()
            if render_task is not None and progress is not None:
                progress.remove_task(render_task)
            return video_path
        if render_task is not None and progress is not None:
            progress.remove_task(render_task)
        return None

    def _fetch_frame(
        self,
        sequence: SNMOTSequence,
        frame_idx: int,
        img_ext: str,
        cap: Optional[cv2.VideoCapture],
    ) -> Optional[np.ndarray]:
        if cap is not None:
            # Video source: frames are sequential, assume _render_visuals consumes in order
            ret, frame = cap.read()
            if not ret:
                return None
            return frame
        frame_path = sequence.img_dir / f"{frame_idx:06d}{img_ext}"
        if not frame_path.exists():
            return None
        return cv2.imread(str(frame_path))

    def _estimate_total_frames(self, sequence: SNMOTSequence) -> Optional[int]:
        limit = self.cfg.limit_frames
        total: Optional[int] = None
        if sequence.img_dir.is_dir():
            total = sequence.length if sequence.length > 0 else None
        else:
            cap = cv2.VideoCapture(str(sequence.img_dir))
            if cap.isOpened():
                frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                if frame_count > 0:
                    total = frame_count
            cap.release()
        if total is not None and limit:
            total = min(total, limit)
        elif total is None and limit:
            total = limit
        return total

    def _create_progress(self) -> Progress:
        return Progress(
            SpinnerColumn(),
            TextColumn("{task.description}"),
            BarColumn(),
            TextColumn("{task.completed}", justify="right"),
            TimeElapsedColumn(),
            console=self.console,
            transient=True,
        )

    def _advance_stage(
        self,
        progress: Optional[Progress],
        task_id: Optional[int],
        labels: List[str],
        current_index: int,
    ) -> int:
        if progress is None or task_id is None:
            return current_index
        if current_index >= len(labels) - 1:
            return current_index
        next_index = current_index + 1
        try:
            progress.advance(task_id)
        except Exception:
            pass
        label = labels[next_index]
        color = "green" if next_index == len(labels) - 1 else "yellow"
        progress.update(task_id, description=f"[{color}]{label}[/{color}]")
        return next_index

    def _assign_team_ids(
        self,
        sequence: SNMOTSequence,
        records: List[Dict[str, float]],
        progress_helper: Optional[StageProgressAdapter] = None,
    ) -> Dict[int, int]:
        if not records:
            return {}
        try:
            from .team_classifier import TeamClassificationSettings, assign_team_ids
        except ImportError as exc:  # pragma: no cover
            self.console.print(f"[yellow]Team classification skipped: {exc}[/yellow]")
            return {}

        settings = TeamClassificationSettings.from_dict(self.team_settings)
        if not settings.enabled:
            return {}
        return assign_team_ids(
            sequence=sequence,
            records=records,
            settings=settings,
            allowed_classes=self.player_classes,
            progress=progress_helper,
        )

    def _build_class_colors(self) -> Dict[int, Tuple[int, int, int]]:
        palette = {
            0: (0, 255, 0),
            1: (0, 0, 255),
            2: (255, 165, 0),
            3: (255, 215, 0),
        }
        class_ids = self.player_classes | self.referee_classes | self.ball_classes
        colors: Dict[int, Tuple[int, int, int]] = {}
        for cls_id in class_ids:
            if cls_id in palette:
                colors[cls_id] = palette[cls_id]
                continue
            r = (37 * (cls_id + 1)) % 255
            g = (17 * (cls_id + 7)) % 255
            b = (29 * (cls_id + 13)) % 255
            colors[cls_id] = (int(r), int(g), int(b))
        return colors

    def _build_association_params(self, target: str) -> Dict[str, float]:
        params = dict(self.cfg.association or {})
        mode = self._embedding_mode_for(target)
        if mode == "reid":
            params["iou_weight"] = float(self.reid_settings.get("alpha", params.get("iou_weight", 1.0)))
            params["appearance_weight"] = float(self.reid_settings.get("beta", params.get("appearance_weight", 0.0)))
            params["max_distance"] = float(
                self.reid_settings.get("max_distance", params.get("max_distance", 0.7))
            )
        elif mode == "color":
            params["iou_weight"] = float(self.simple_embedding_alpha)
            params["appearance_weight"] = float(self.simple_beta)
            params["max_distance"] = float(self.simple_max_distance)
        else:
            params["appearance_weight"] = 0.0
        return params

    def _embedding_mode_for(self, target: str) -> Optional[str]:
        if target == "players":
            target_classes = self.player_classes
        elif target == "referees":
            target_classes = self.referee_classes
        else:
            target_classes = set()

        if self.reid_enabled and (target_classes & self.reid_target_classes):
            return "reid"
        if self.simple_embedding_enabled and (target_classes & self.simple_embedding_targets):
            return "color"
        return None

    def _resolve_reid_targets(self) -> Set[int]:
        target = str(self.reid_settings.get("target_classes", "players"))
        if target == "all":
            return self.player_classes | self.referee_classes
        return set(self.player_classes)

    def _resolve_embedding_targets(self, target: str) -> Set[int]:
        if target == "all":
            return self.player_classes | self.referee_classes
        if target == "referees":
            return set(self.referee_classes)
        return set(self.player_classes)

    def _build_reid_extractor(self):
        if not self.reid_enabled:
            return None
        try:
            from .reid_inference import ReIDEmbeddingExtractor, ReIDRuntimeConfig
        except ImportError as exc:  # pragma: no cover
            self.console.print(f"[yellow]ReID disabled: {exc}[/yellow]")
            self.reid_enabled = False
            return None

        weights = self.reid_settings.get("weights") or self.cfg.reid_weights
        if not weights:
            self.console.print("[yellow]ReID disabled: weights not provided[/yellow]")
            self.reid_enabled = False
            return None

        runtime = ReIDRuntimeConfig(
            weights=weights,
            device=self.reid_settings.get("device", self.cfg.device),
            image_size=self.reid_settings.get("image_size", (256, 128)),
            target_classes=str(self.reid_settings.get("target_classes", "players")),
        )

        try:
            return ReIDEmbeddingExtractor(runtime)
        except FileNotFoundError as exc:
            self.console.print(f"[yellow]{exc}[/yellow]")
            self.reid_enabled = False
            return None

    def _maybe_compute_embeddings(self, frame, xyxy: np.ndarray, classes: List[int]):
        if not self.reid_enabled or self.reid_extractor is None or frame is None:
            if self.simple_embedding_enabled and frame is not None:
                return self._color_embeddings(frame, xyxy, classes)
            return [None] * len(classes)
        return self.reid_extractor.extract(frame, xyxy, classes, self.reid_target_classes)

    def _color_embeddings(self, frame, xyxy: np.ndarray, classes: List[int]) -> List[Optional[np.ndarray]]:
        embeddings: List[Optional[np.ndarray]] = []
        converted = None
        if self.simple_embedding_space == "lab":
            converted = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
        elif self.simple_embedding_space == "hsv":
            converted = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        else:
            converted = frame
        bins = self.simple_embedding_bins
        for bbox, cls in zip(xyxy, classes):
            if cls not in self.simple_embedding_targets:
                embeddings.append(None)
                continue
            x1, y1, x2, y2 = bbox.astype(int)
            x1 = max(0, x1)
            y1 = max(0, y1)
            x2 = min(converted.shape[1], x2)
            y2 = min(converted.shape[0], y2)
            if x2 <= x1 + 1 or y2 <= y1 + 1:
                embeddings.append(None)
                continue
            patch = converted[y1:y2, x1:x2]
            hist = []
            for ch in range(patch.shape[2]):
                channel_hist = cv2.calcHist([patch], [ch], None, [bins], [0, 256])
                hist.append(channel_hist.flatten())
            descriptor = np.concatenate(hist)
            norm = np.linalg.norm(descriptor)
            if norm > 1e-6:
                descriptor = descriptor / norm
            embeddings.append(descriptor.astype(np.float32))
        return embeddings

    def _normalize_device(self, requested: Optional[str], component: str) -> str:
        device = (requested or "cpu").strip()
        lowered = device.lower()
        if lowered == "auto":
            if torch.cuda.is_available() and torch.cuda.device_count() > 0:
                return "0"
            self.console.print(
                f"[yellow]{component}: CUDA unavailable, falling back to CPU[/yellow]"
            )
            return "cpu"

        if lowered != "cpu" and not torch.cuda.is_available():
            self.console.print(
                f"[yellow]{component}: requested '{device}' but CUDA is unavailable, using CPU[/yellow]"
            )
            return "cpu"

        return device
