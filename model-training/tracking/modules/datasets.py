from __future__ import annotations

import configparser
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, Iterator, List, Optional, Sequence, Tuple

from PIL import Image
from torch.utils.data import Dataset


# ------------------------------- SoccerNet ReID -------------------------------

def parse_soccernet_filename(path: Path) -> Dict[str, str]:
    """Parse SoccerNet/SN-ReID filename metadata.

    Handles both underscore-separated (legacy) and dash-separated (public
    SoccerNet ReID) naming conventions:

    - ``0001_002_003_004_Player_team_left_a_005r000_003c178b000c_146x68.png``
    - ``3032-108-1816-282-Player_team_left-a-005r000_003c178b000c-146x68.png``
    """

    stem = path.stem
    match = re.match(r"(\d+)([-_])", stem)
    delim = match.group(2) if match else "_"

    # Separate the trailing HxW block if present
    main, sep, shape_part = stem.rpartition(delim)
    if sep and "x" in shape_part and all(part.isdigit() for part in shape_part.split("x") if part):
        base = main
    else:
        base = stem
        shape_part = ""

    parts = base.split(delim, 4)
    while len(parts) < 5:
        parts.append("")

    bbox_idx, action_idx, person_uid, frame_idx, tail = parts[:5]
    tail_segments = tail.rsplit(delim, 3)
    while len(tail_segments) < 3:
        tail_segments.append("")

    class_label = tail_segments[0]
    id_part = tail_segments[1]
    uai_part = delim.join(tail_segments[2:]) or ""

    if not shape_part:
        # If HxW was still in the string, pop it now
        maybe_rest, maybe_shape = uai_part.rsplit(delim, 1) if delim in uai_part else (uai_part, "")
        if "x" in maybe_shape and all(part.isdigit() for part in maybe_shape.split("x") if part):
            shape_part = maybe_shape
            uai_part = maybe_rest

    meta = {
        "filename": path.name,
        "bbox_idx": bbox_idx or "0",
        "action_idx": action_idx or "0",
        "person_uid": person_uid or stem,
        "frame_idx": frame_idx or "0",
        "class_label": class_label or "unknown",
        "id": id_part or "unknown",
        "uai": uai_part or "unknown",
        "height": -1,
        "width": -1,
    }

    if "x" in shape_part:
        h, w = shape_part.split("x", 1)
        if h.isdigit():
            meta["height"] = int(h)
        if w.isdigit():
            meta["width"] = int(w)

    return meta


def build_pid_lookup(root: Path, splits: Sequence[str]) -> Dict[str, int]:
    """Create a consistent person_uid -> integer label mapping across splits."""

    lookup: Dict[str, int] = {}
    for split in splits:
        split_dir = root / split
        if not split_dir.exists():
            continue
        for img_path in split_dir.rglob("*.png"):
            meta = parse_soccernet_filename(img_path)
            pid = meta["person_uid"]
            if pid not in lookup:
                lookup[pid] = len(lookup)
    return lookup


class SoccerNetReIDDataset(Dataset):
    """Minimal SoccerNet ReID dataset for classification-based training."""

    def __init__(
        self,
        root: Path,
        split: str,
        pid_lookup: Dict[str, int],
        transform=None,
        max_samples: Optional[int] = None,
        extensions: Sequence[str] = (".png", ".jpg"),
    ) -> None:
        self.root = Path(root)
        self.split = split
        self.transform = transform
        self.pid_lookup = pid_lookup
        split_dir = self.root / split
        if not split_dir.exists():
            raise FileNotFoundError(f"Split '{split}' not found under {self.root}")

        self.samples: List[Tuple[Path, int, Dict[str, str]]] = []
        for ext in extensions:
            if list(split_dir.rglob(f"*{ext}")):
                image_paths = sorted(split_dir.rglob(f"*{ext}"))
                break
        else:
            image_paths = []

        for img_path in image_paths:
            meta = parse_soccernet_filename(img_path)
            pid_key = meta["person_uid"]
            if pid_key not in self.pid_lookup:
                continue
            pid = self.pid_lookup[pid_key]
            self.samples.append((img_path, pid, meta))
            if max_samples and len(self.samples) >= max_samples:
                break

        if not self.samples:
            raise RuntimeError(
                f"No images discovered for split '{split}'. Check dataset extraction."
            )

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int):
        img_path, pid, meta = self.samples[index]
        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return {
            "image": image,
            "label": pid,
            "path": str(img_path),
            "meta": meta,
        }


# ------------------------------- SNMOT dataset --------------------------------


def _read_ini_file(path: Path) -> Dict[str, str]:
    if not path.exists():
        return {}
    parser = configparser.ConfigParser()
    parser.read(path)
    section = parser.sections()[0] if parser.sections() else None
    return dict(parser[section]) if section else {}


@dataclass
class SNMOTSequence:
    name: str
    root: Path
    img_dir: Path
    seqinfo: Dict[str, str]
    gameinfo: Dict[str, str]
    detections: Optional[Path] = None
    ground_truth: Optional[Path] = None

    @property
    def frame_rate(self) -> float:
        return float(self.seqinfo.get("framerate", 25))

    @property
    def length(self) -> int:
        return int(self.seqinfo.get("seqlength", self.seqinfo.get("length", 0)))

    def iter_frames(self) -> Iterator[Tuple[int, Path]]:
        for frame_idx, img_path in enumerate(sorted(self.img_dir.glob("*.jpg")), start=1):
            yield frame_idx, img_path


class SNMOTDataset:
    """Utility to list MOT-style tracking sequences."""

    def __init__(
        self,
        root: Path,
        sequence_names: Optional[Iterable[str]] = None,
        split_dirs: Optional[Iterable[str]] = None,
    ) -> None:
        self.root = Path(root)
        if not self.root.exists():
            raise FileNotFoundError(f"Tracking root does not exist: {self.root}")

        if sequence_names:
            candidates = [self.root / name for name in sequence_names]
        else:
            candidates = []
            if split_dirs:
                for split in split_dirs:
                    split_path = self.root / split
                    if not split_path.exists():
                        continue
                    candidates.extend(p for p in split_path.iterdir() if p.is_dir())
            if not candidates:
                candidates = [p for p in self.root.iterdir() if p.is_dir()]

        self.sequences: List[SNMOTSequence] = []
        for seq_dir in candidates:
            seqinfo_path = seq_dir / "seqinfo.ini"
            if not seqinfo_path.exists():
                continue
            seqinfo = _read_ini_file(seqinfo_path)
            gameinfo = _read_ini_file(seq_dir / "gameinfo.ini")
            img_dir = seq_dir / seqinfo.get("imdir", "img1")
            detections = seq_dir / "det" / "det.txt"
            if not detections.exists():
                detections = None
            ground_truth = seq_dir / "gt" / "gt.txt"
            if not ground_truth.exists():
                ground_truth = None
            self.sequences.append(
                SNMOTSequence(
                    name=seq_dir.name,
                    root=seq_dir,
                    img_dir=img_dir,
                    seqinfo=seqinfo,
                    gameinfo=gameinfo,
                    detections=detections,
                    ground_truth=ground_truth,
                )
            )

        if not self.sequences:
            raise RuntimeError(
                f"No sequences discovered under {self.root}. "
                "Ensure each sequence has a seqinfo.ini file."
            )

    def names(self) -> List[str]:
        return [seq.name for seq in self.sequences]

    def __iter__(self) -> Iterator[SNMOTSequence]:
        return iter(self.sequences)
