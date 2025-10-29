#!/usr/bin/env python3
"""
download_dataset.py

Download a Hugging Face dataset and save it under the repository's `ballDataset` directory.

By default this script saves the dataset using `datasets.Dataset.save_to_disk` (Arrow format).
It also has an `--inspect` mode that prints columns and a sample entry so you can see how images/annotations are stored
before converting them to another format (for example, images + YOLO annotations).

Usage examples:
  # install dependencies first
  pip install datasets huggingface_hub pillow tqdm

  # save dataset in arrow format to model-training/yolo/ball-detection/ballDataset
  python download_dataset.py --mode arrow

  # inspect first sample to learn field names / structure
  python download_dataset.py --mode inspect --max-samples 2

"""
from pathlib import Path
import argparse
import json

from datasets import load_dataset

def save_arrow(dataset, outdir: Path):
    outdir.mkdir(parents=True, exist_ok=True)
    print(f"Saving dataset in Hugging Face Arrow format to: {outdir}")
    dataset.save_to_disk(str(outdir))
    print("Done.")

def inspect_dataset(dataset, n=1):
    # Print column names and the declared features/schema without materializing image bytes
    print("Columns:", dataset.column_names)
    try:
        print("Features:")
        for k, feat in dataset.features.items():
            print(f"  - {k}: {repr(feat)}")
    except Exception:
        print("  (could not read dataset.features)")

    # If underlying Arrow table/schema is available, print it (safe, no image decoding)
    try:
        arrow_table = getattr(dataset, "data", None)
        if arrow_table is not None and hasattr(arrow_table, "schema"):
            print("Arrow schema:")
            print(arrow_table.schema)
    except Exception:
        pass

    print("\nNote: to avoid triggering image decoding errors this inspection avoids reading sample values.\n"
          "If you want to peek at individual fields that are not images, run the script locally and use --mode inspect")

def main():
    parser = argparse.ArgumentParser(description="Download Hugging Face dataset into ballDataset")
    parser.add_argument("--dataset", default="Adit-jain/Soccana_player_ball_detection_v1", help="Hugging Face dataset id")
    parser.add_argument("--split", default="train")
    parser.add_argument("--outdir", default=None, help="Output path; default: model-training/yolo/ball-detection/ballDataset")
    parser.add_argument("--mode", choices=["arrow", "inspect"], default="arrow", help="arrow: save_to_disk; inspect: print sample structure")
    parser.add_argument("--max-samples", type=int, default=1, help="Number of samples to show in inspect mode")
    args = parser.parse_args()

    # Default path relative to this script: ../ballDataset
    here = Path(__file__).resolve().parent
    default_out = (here.parent / "ballDataset").resolve()
    outdir = Path(args.outdir).resolve() if args.outdir else default_out

    print(f"Loading dataset {args.dataset} split={args.split} ...")
    ds = load_dataset(args.dataset, split=args.split)
    print(f"Loaded dataset with {len(ds)} samples")

    if args.mode == "arrow":
        save_arrow(ds, outdir)
    elif args.mode == "inspect":
        inspect_dataset(ds, n=args.max_samples)

if __name__ == "__main__":
    main()
