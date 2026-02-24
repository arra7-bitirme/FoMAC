from __future__ import annotations

import os
import pickle
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterator, Optional, Tuple

import cv2
import numpy as np
from tqdm import tqdm


SN_SPOTTING_BASE = "https://raw.githubusercontent.com/SoccerNet/sn-spotting/main/Features"
DEFAULT_PCA_FILENAME = "pca_512_TF2.pkl"
DEFAULT_AVG_FILENAME = "average_512_TF2.pkl"


@dataclass(frozen=True)
class PCAExtractorAssets:
    pca_path: str
    average_path: str


def _download_if_missing(
    url: str,
    dst_path: str,
    *,
    progress_cb: Optional[Callable[[str, int, int, str], None]] = None,
    stage: str = "assets_download",
    label: str = "",
) -> None:
    dst = Path(dst_path)
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists() and dst.stat().st_size > 0:
        return
    tmp = dst.with_suffix(dst.suffix + ".tmp")
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "FoMAC/1.0"})
        with urllib.request.urlopen(req) as resp, open(tmp, "wb") as out:
            total = 0
            try:
                total = int(resp.headers.get("Content-Length") or 0)
            except Exception:
                total = 0

            downloaded = 0
            pbar = tqdm(
                total=total if total > 0 else None,
                desc=(label or "download"),
                unit="B",
                unit_scale=True,
                unit_divisor=1024,
            )
            if progress_cb is not None:
                try:
                    progress_cb(stage, 0, total if total > 0 else 0, f"Downloading {label}".strip())
                except Exception:
                    pass

            while True:
                chunk = resp.read(1024 * 1024)
                if not chunk:
                    break
                out.write(chunk)
                downloaded += len(chunk)
                try:
                    pbar.update(len(chunk))
                except Exception:
                    pass
                if progress_cb is not None and total > 0:
                    try:
                        progress_cb(stage, downloaded, total, f"Downloading {label}".strip())
                    except Exception:
                        pass
            try:
                pbar.close()
            except Exception:
                pass
        tmp.replace(dst)
        if progress_cb is not None:
            try:
                progress_cb(stage, 1, 1, f"Downloaded {label}".strip())
            except Exception:
                pass
    finally:
        try:
            if tmp.exists():
                tmp.unlink()
        except Exception:
            pass


def ensure_sn_spotting_pca_assets(
    asset_dir: str,
    *,
    progress_cb: Optional[Callable[[str, int, int, str], None]] = None,
) -> PCAExtractorAssets:
    """Ensure PCA512 assets from SoccerNet/sn-spotting exist locally.

    We download the official pickles if they are missing.
    """
    asset_dir_p = Path(asset_dir)
    asset_dir_p.mkdir(parents=True, exist_ok=True)
    pca_path = str(asset_dir_p / DEFAULT_PCA_FILENAME)
    avg_path = str(asset_dir_p / DEFAULT_AVG_FILENAME)

    if progress_cb is not None:
        try:
            progress_cb("assets", 0, 2, "PCA512 asset kontrol")
        except Exception:
            pass

    # PCA
    if Path(pca_path).exists() and Path(pca_path).stat().st_size > 0:
        if progress_cb is not None:
            try:
                progress_cb("assets", 1, 2, f"Asset mevcut: {DEFAULT_PCA_FILENAME}")
            except Exception:
                pass
    else:
        _download_if_missing(
            f"{SN_SPOTTING_BASE}/{DEFAULT_PCA_FILENAME}",
            pca_path,
            progress_cb=progress_cb,
            stage="assets_download",
            label=DEFAULT_PCA_FILENAME,
        )
        if progress_cb is not None:
            try:
                progress_cb("assets", 1, 2, f"Asset indirildi: {DEFAULT_PCA_FILENAME}")
            except Exception:
                pass

    # Average
    if Path(avg_path).exists() and Path(avg_path).stat().st_size > 0:
        if progress_cb is not None:
            try:
                progress_cb("assets", 2, 2, f"Asset mevcut: {DEFAULT_AVG_FILENAME}")
            except Exception:
                pass
    else:
        _download_if_missing(
            f"{SN_SPOTTING_BASE}/{DEFAULT_AVG_FILENAME}",
            avg_path,
            progress_cb=progress_cb,
            stage="assets_download",
            label=DEFAULT_AVG_FILENAME,
        )
        if progress_cb is not None:
            try:
                progress_cb("assets", 2, 2, f"Asset indirildi: {DEFAULT_AVG_FILENAME}")
            except Exception:
                pass

    if progress_cb is not None:
        try:
            progress_cb("assets", 2, 2, "PCA512 asset hazır")
        except Exception:
            pass
    return PCAExtractorAssets(pca_path=pca_path, average_path=avg_path)


def _center_crop_resize_224(frame_bgr: np.ndarray, *, transform: str) -> np.ndarray:
    if frame_bgr is None or frame_bgr.size == 0:
        raise ValueError("empty frame")

    if transform == "crop":
        h, w = frame_bgr.shape[:2]
        # Resize shortest side to 256 (common ImageNet eval protocol) then center-crop 224x224
        scale = 256.0 / float(min(h, w) + 1e-9)
        new_w = max(1, int(round(w * scale)))
        new_h = max(1, int(round(h * scale)))
        resized = cv2.resize(frame_bgr, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
        x0 = max(0, (new_w - 224) // 2)
        y0 = max(0, (new_h - 224) // 2)
        cropped = resized[y0 : y0 + 224, x0 : x0 + 224]
        if cropped.shape[0] != 224 or cropped.shape[1] != 224:
            cropped = cv2.resize(cropped, (224, 224), interpolation=cv2.INTER_LINEAR)
        return cropped

    # default: simple resize
    return cv2.resize(frame_bgr, (224, 224), interpolation=cv2.INTER_LINEAR)


def _iter_sample_times(
    *,
    start_sec: float,
    duration_sec: float,
    fps: float,
) -> Iterator[float]:
    if fps <= 0:
        raise ValueError("fps must be > 0")
    step = 1.0 / float(fps)
    t = float(start_sec)
    end_t = float(start_sec) + float(duration_sec)
    # include the last sample only if strictly before end_t
    while t < end_t:
        yield t
        t += step


def _get_video_duration_sec(cap: cv2.VideoCapture) -> float:
    fps_vid = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
    frames = float(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0.0)
    if fps_vid > 0 and frames > 0:
        return frames / fps_vid
    # fallback: try duration via POS_MSEC seek
    return 0.0


def extract_resnet_tf2_pca512(
    *,
    video_path: str,
    out_features_npy: str,
    start_sec: Optional[float] = None,
    duration_sec: Optional[float] = None,
    fps: float = 2.0,
    transform: str = "crop",
    assets_dir: Optional[str] = None,
    overwrite: bool = False,
    batch_size: int = 64,
    progress_cb: Optional[Callable[[str, int, int, str], None]] = None,
) -> str:
    """Extract SoccerNet-style ResNet152 (TF2) features and reduce to PCA512.

    Output is a `.npy` array with shape (T, 512) at ~`fps`.
    """
    video_path = str(Path(video_path).resolve())
    out_features_npy = str(Path(out_features_npy).resolve())

    if (not overwrite) and os.path.isfile(out_features_npy) and os.path.getsize(out_features_npy) > 0:
        return out_features_npy

    if assets_dir is None:
        assets_dir = str(Path(__file__).resolve().parent / "assets" / "sn_spotting_features")
    assets = ensure_sn_spotting_pca_assets(assets_dir, progress_cb=progress_cb)

    # Lazy imports: tensorflow & sklearn are heavy and should not load unless needed
    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
    try:
        from tensorflow.keras.models import Model  # type: ignore
        from tensorflow.keras.applications.resnet import preprocess_input  # type: ignore
        from tensorflow import keras  # type: ignore
    except Exception as e:  # pragma: no cover
        raise RuntimeError(
            "TensorFlow is required for PCA feature extraction. "
            "Install backend requirements including tensorflow. "
            f"Import error: {e}"
        )

    # load PCA assets
    with open(assets.pca_path, "rb") as f:
        pca_obj = pickle.load(f)
    with open(assets.average_path, "rb") as f:
        avg_obj = pickle.load(f)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video: {video_path}")

    try:
        total_dur = _get_video_duration_sec(cap)
        s = float(start_sec) if start_sec is not None else 0.0
        if duration_sec is None:
            if total_dur > 0:
                d = max(0.0, total_dur - s)
            else:
                # if duration cannot be inferred, we'll sample until read fails
                d = 1e12
        else:
            d = float(duration_sec)

        # create pretrained encoder (ResNet152 pre-trained on ImageNet)
        base_model = keras.applications.resnet.ResNet152(
            include_top=True,
            weights="imagenet",
            input_tensor=None,
            input_shape=None,
            pooling=None,
            classes=1000,
        )
        model = Model(base_model.input, outputs=[base_model.get_layer("avg_pool").output])
        model.trainable = False

        def reduce_features(feat_2048: np.ndarray) -> np.ndarray:
            feat = feat_2048
            if avg_obj is not None:
                feat = feat - avg_obj
            if pca_obj is not None:
                # sklearn PCA object
                if hasattr(pca_obj, "transform"):
                    feat = pca_obj.transform(feat)
                else:
                    # fallback: assume matrix
                    feat = feat @ np.asarray(pca_obj).T
            return feat

        out_chunks: list[np.ndarray] = []
        frames_batch: list[np.ndarray] = []

        def flush_batch() -> None:
            nonlocal frames_batch
            if not frames_batch:
                return
            rgb = np.stack(frames_batch, axis=0).astype(np.float32)
            rgb = preprocess_input(rgb)
            feat = model.predict(rgb, batch_size=min(batch_size, len(frames_batch)), verbose=0)
            feat = np.asarray(feat, dtype=np.float32)
            reduced = reduce_features(feat).astype(np.float32)
            out_chunks.append(reduced)
            frames_batch = []

        def emit(stage: str, cur: int, total: int, msg: str) -> None:
            if progress_cb is None:
                return
            try:
                progress_cb(stage, int(cur), int(total), str(msg))
            except Exception:
                pass

        emit("features", 0, 1, "ResNet152 + PCA512 feature extraction başladı")

        # If we have a real duration, sample with explicit timestamps.
        # If duration is unknown (d very large), we keep sampling forward until read fails.
        if d < 1e11:
            total_steps = max(1, int(np.ceil(float(d) * float(fps))))
            it = _iter_sample_times(start_sec=s, duration_sec=d, fps=fps)
            for i, t in enumerate(tqdm(it, total=total_steps, desc="PCA512 features", unit="frame")):
                cap.set(cv2.CAP_PROP_POS_MSEC, float(t) * 1000.0)
                ok, frame = cap.read()
                if not ok or frame is None:
                    break
                frame = _center_crop_resize_224(frame, transform=transform)
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frames_batch.append(frame_rgb)
                if len(frames_batch) >= batch_size:
                    flush_batch()
                emit("features", i + 1, total_steps, "Feature çıkarılıyor")
            flush_batch()
        else:
            # unknown duration: iterate sequentially with stepping by frames
            fps_vid = float(cap.get(cv2.CAP_PROP_FPS) or 25.0)
            step_frames = max(1, int(round(fps_vid / float(fps))))
            cap.set(cv2.CAP_PROP_POS_MSEC, float(s) * 1000.0)
            frame_idx = int(cap.get(cv2.CAP_PROP_POS_FRAMES) or 0)
            seen = 0
            pbar = tqdm(desc="PCA512 features", unit="frame")
            while True:
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
                ok, frame = cap.read()
                if not ok or frame is None:
                    break
                frame = _center_crop_resize_224(frame, transform=transform)
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frames_batch.append(frame_rgb)
                if len(frames_batch) >= batch_size:
                    flush_batch()
                frame_idx += step_frames
                seen += 1
                pbar.update(1)
                emit("features", seen, max(seen, 1), "Feature çıkarılıyor")
            flush_batch()
            try:
                pbar.close()
            except Exception:
                pass

        if not out_chunks:
            raise RuntimeError("No frames sampled; cannot extract features")

        features_512 = np.concatenate(out_chunks, axis=0)
        os.makedirs(str(Path(out_features_npy).parent), exist_ok=True)
        np.save(out_features_npy, features_512)
        emit("features", 1, 1, f"Features hazır: {Path(out_features_npy).name} (T={features_512.shape[0]}, D={features_512.shape[1]})")
        return out_features_npy
    finally:
        try:
            cap.release()
        except Exception:
            pass
