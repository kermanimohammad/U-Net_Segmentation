"""Shared utilities: seeding, logging, and filesystem helpers."""

from __future__ import annotations

import logging
import os
import random
import sys
from pathlib import Path
from typing import Iterable, List, Optional, Tuple

import numpy as np
import tensorflow as tf

from WWR_Segmentation.config import Config


def setup_logging(level: int = logging.INFO, log_file: Optional[Path] = None) -> logging.Logger:
    """Configure root logger with console and optional file handler."""
    logger = logging.getLogger("wwr_segmentation")
    logger.setLevel(level)
    logger.handlers.clear()

    formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    if log_file is not None:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_file, encoding="utf-8")
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger


def set_random_seeds(seed: int = 42) -> None:
    """Set seeds for Python, NumPy, and TensorFlow for reproducibility."""
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)

    try:
        tf.config.experimental.enable_op_determinism()
    except (AttributeError, tf.errors.NotFoundError):
        pass


def enable_gpu_memory_growth() -> None:
    """Allow GPU memory growth to avoid allocating all VRAM upfront."""
    gpus = tf.config.list_physical_devices("GPU")
    for gpu in gpus:
        try:
            tf.config.experimental.set_memory_growth(gpu, True)
        except RuntimeError:
            pass


def configure_mixed_precision(enabled: bool = True) -> None:
    """Enable or disable mixed-precision training policy."""
    if enabled:
        policy = tf.keras.mixed_precision.Policy("mixed_float16")
        tf.keras.mixed_precision.set_global_policy(policy)


def list_image_files(directory: Path, extensions: Tuple[str, ...]) -> List[Path]:
    """Return sorted list of image file paths in *directory*."""
    if not directory.exists():
        raise FileNotFoundError(f"Directory not found: {directory}")

    files: List[Path] = []
    for ext in extensions:
        files.extend(directory.glob(f"*{ext}"))
        files.extend(directory.glob(f"*{ext.upper()}"))

    files = sorted(set(files))
    if not files:
        raise FileNotFoundError(f"No images found in {directory}")
    return files


def pair_images_with_masks(
    image_paths: Iterable[Path],
    mask_dir: Path,
    extensions: Tuple[str, ...],
) -> List[Tuple[str, str]]:
    """Pair image paths with corresponding mask paths by stem name."""
    pairs: List[Tuple[str, str]] = []
    for img_path in image_paths:
        stem = img_path.stem
        mask_path: Optional[Path] = None
        for ext in extensions:
            candidate = mask_dir / f"{stem}{ext}"
            if candidate.exists():
                mask_path = candidate
                break
            candidate = mask_dir / f"{stem}{ext.upper()}"
            if candidate.exists():
                mask_path = candidate
                break

        if mask_path is None:
            raise FileNotFoundError(
                f"No mask found for image '{img_path.name}' in {mask_dir}"
            )
        pairs.append((str(img_path), str(mask_path)))

    return pairs


def mount_google_drive(mount_point: Path = Path("/content/drive")) -> None:
    """Mount Google Drive in a Colab environment."""
    try:
        from google.colab import drive  # type: ignore[import-untyped]
    except ImportError as exc:
        raise RuntimeError("google.colab is only available in Google Colab.") from exc

    drive.mount(str(mount_point))


def prepare_environment(config: Config) -> logging.Logger:
    """Apply all runtime configuration (seeds, GPU, mixed precision)."""
    set_random_seeds(config.seed)
    enable_gpu_memory_growth()
    configure_mixed_precision(config.mixed_precision)

    if config.use_google_drive:
        mount_google_drive(config.drive_mount_point)

    log_file = config.log_dir / "training.log"
    return setup_logging(log_file=log_file)
