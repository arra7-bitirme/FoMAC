"""Live class-wise metrics reporter.

Watches a training project's output directory for metric files and prints
per-class metrics as they appear. This is best-effort because Ultralytics
may output different file names depending on version/configuration.

Search order:
 - <project_dir>/metrics_per_class.csv  (expected: class,precision,recall,mAP50,mAP50-95)
 - <project_dir>/results.csv            (falls back to overall metrics when per-class absent)

The reporter runs in a background thread and polls for file changes.
"""
from __future__ import annotations

import csv
import logging
import os
import threading
import time
from pathlib import Path
from typing import Dict, Optional

logger = logging.getLogger(__name__)


class ClasswiseReporter:
    def __init__(self, project_dir: Path, name: str, poll_interval: float = 5.0):
        """Initialize reporter.

        Args:
            project_dir: Root models/project directory where training writes results
            name: training run name (used as subdir under project_dir)
            poll_interval: seconds between file checks
        """
        self.project_dir = Path(project_dir)
        self.name = name
        self.poll_interval = poll_interval
        self._stop_event = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._last_mtime: Dict[Path, float] = {}

    def start(self):
        if self._thread and self._thread.is_alive():
            return
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()
        logger.info(f"ClasswiseReporter started for {self.project_dir}/{self.name}")

    def stop(self):
        self._stop_event.set()
        if self._thread:
            self._thread.join(timeout=5.0)
        logger.info("ClasswiseReporter stopped")

    def _run(self):
        run_dir = self.project_dir / self.name
        # common locations: project/name/ or project/name/metrics/
        possible_files = [run_dir / "metrics_per_class.csv", run_dir / "results.csv"]

        while not self._stop_event.is_set():
            try:
                # prefer per-class file
                per_class_file = run_dir / "metrics_per_class.csv"
                if per_class_file.exists():
                    self._maybe_process_file(per_class_file, per_class=True)
                else:
                    results_csv = run_dir / "results.csv"
                    if results_csv.exists():
                        self._maybe_process_file(results_csv, per_class=False)
                # also check inside 'metrics' subdir
                metrics_dir = run_dir / "metrics"
                if metrics_dir.exists() and metrics_dir.is_dir():
                    pc = metrics_dir / "metrics_per_class.csv"
                    if pc.exists():
                        self._maybe_process_file(pc, per_class=True)
                time.sleep(self.poll_interval)
            except Exception as e:
                logger.debug(f"ClasswiseReporter polling error: {e}")
                time.sleep(self.poll_interval)

    def _maybe_process_file(self, path: Path, per_class: bool):
        try:
            mtime = path.stat().st_mtime
        except Exception:
            return

        last = self._last_mtime.get(path)
        if last is None or mtime > last:
            # file updated
            logger.info(f"Detected metric update: {path}")
            try:
                if per_class:
                    metrics = self._parse_per_class_csv(path)
                    self._log_per_class(metrics)
                else:
                    # fallback: parse last row of results.csv and show overall metrics
                    overall = self._parse_results_csv_latest(path)
                    if overall:
                        logger.info("Overall metrics (latest epoch):")
                        logger.info(overall)
            except Exception as e:
                logger.debug(f"Error processing metric file {path}: {e}")

            self._last_mtime[path] = mtime

    def _parse_per_class_csv(self, path: Path) -> Dict[str, Dict[str, float]]:
        """Parse a per-class CSV into a dictionary.

        Expected CSV header must contain class column and one or more metric columns
        like precision, recall, mAP50, mAP50-95. We tolerate multiple formats.
        """
        metrics = {}
        with open(path, newline='', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                # find a class name key
                cls_name = None
                for k in row.keys():
                    if k.lower() in ('class', 'cls', 'name'):
                        cls_name = row[k]
                        break
                if cls_name is None:
                    # fallback to first column value
                    cls_name = next(iter(row.values()))

                # collect numeric metrics
                data = {}
                for k, v in row.items():
                    if k == '' or v is None:
                        continue
                    key = k.strip()
                    try:
                        data[key] = float(v)
                    except Exception:
                        # skip non-numeric fields
                        pass
                metrics[str(cls_name)] = data
        return metrics

    def _parse_results_csv_latest(self, path: Path) -> Optional[Dict[str, float]]:
        # read last non-empty row
        last = None
        with open(path, newline='', encoding='utf-8') as f:
            reader = csv.reader(f)
            rows = [r for r in reader if r]
            if len(rows) < 2:
                return None
            header = rows[0]
            lastrow = rows[-1]
            if len(header) != len(lastrow):
                # try to align up to min length
                n = min(len(header), len(lastrow))
                header = header[:n]
                lastrow = lastrow[:n]
            try:
                return {h: float(v) for h, v in zip(header, lastrow) if _is_float(v)}
            except Exception:
                return None

    def _log_per_class(self, metrics: Dict[str, Dict[str, float]]):
        if not metrics:
            logger.info("No per-class metrics found in file")
            return

        # sort classes for deterministic output
        for cls in sorted(metrics.keys()):
            row = metrics[cls]
            # attempt to show common metric names
            mAP50 = _first_float_by_keys(row, ['mAP50', 'map50', 'mAP@0.5', 'mAP(0.5)'])
            mAP5095 = _first_float_by_keys(row, ['mAP50-95', 'map50-95', 'mAP@0.5:0.95'])
            prec = _first_float_by_keys(row, ['precision', 'prec'])
            rec = _first_float_by_keys(row, ['recall', 'rec'])

            logger.info(f"Class: {cls} | mAP50: {fmt(mAP50)} | mAP50-95: {fmt(mAP5095)} | P: {fmt(prec)} | R: {fmt(rec)}")


def _is_float(s: str) -> bool:
    try:
        float(s)
        return True
    except Exception:
        return False


def _first_float_by_keys(d: Dict[str, float], keys):
    for k in keys:
        if k in d:
            try:
                return float(d[k])
            except Exception:
                pass
    # try to find any float
    for v in d.values():
        try:
            return float(v)
        except Exception:
            continue
    return None


def fmt(v: Optional[float]) -> str:
    return f"{v:.4f}" if v is not None else "-"
