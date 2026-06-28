"""Google Colab setup: Drive mount, local dataset cache, GPU check."""

from __future__ import annotations

import logging
import shutil
import zipfile
from pathlib import Path
from typing import Optional

from WWR_Segmentation.config import Config
from WWR_Segmentation.dataset_layout import attach_test_from_drive, prepare_dataset_root
from WWR_Segmentation.utils import mount_google_drive, prepare_environment, setup_logging

logger = logging.getLogger("wwr_segmentation")

CODE_VERSION = "2026.06.28.1"


def verify_gpu(prefer_a100: bool = True) -> str:
    """Detect GPU and warn if A100 is expected but not available."""
    import tensorflow as tf

    gpus = tf.config.list_physical_devices("GPU")
    if not gpus:
        logger.warning(
            "No GPU detected. In Colab: Runtime → Change runtime type → A100 GPU."
        )
        return "CPU"

    try:
        from tensorflow.python.client import device_lib

        gpu_details = [d for d in device_lib.list_local_devices() if d.device_type == "GPU"]
        if gpu_details:
            name = gpu_details[0].physical_device_desc or gpu_details[0].name
            logger.info("GPU: %s", name)
            if prefer_a100 and "A100" not in name.upper():
                logger.warning(
                    "A100 not detected. Runtime → Change runtime type → A100 GPU."
                )
            return name
    except Exception:
        pass

    return gpus[0].name


def ensure_project_on_path(
    allow_upload: bool = False,
    use_local_copy: bool = True,
) -> Path:
    """Delegate to ``bootstrap_colab`` (Drive / code.zip only)."""
    from WWR_Segmentation.colab_bootstrap import bootstrap_colab

    return bootstrap_colab(allow_upload=allow_upload, use_local_copy=use_local_copy)


def prepare_local_dataset(config: Config, force: bool = False) -> Path:
    """Copy ``data.zip`` from Drive and extract to ``/content/data`` for fast I/O."""
    local_root = Path(config.local_data_root)
    marker = local_root / ".dataset_ready"
    zip_on_drive = Path(config.drive_data_zip)

    if force and local_root.exists():
        shutil.rmtree(local_root, ignore_errors=True)
        marker.unlink(missing_ok=True)

    if marker.exists() and not force and (local_root / "train" / "images").exists():
        logger.info("Dataset already prepared at %s", local_root)
        return local_root

    if not zip_on_drive.exists():
        raise FileNotFoundError(
            f"data.zip not found: {zip_on_drive}\n"
            "Upload to: My Drive/WWR_Seg_Model/data.zip"
        )

    local_zip = Path("/content/data.zip")
    logger.info("Copying data.zip to local storage...")
    shutil.copy2(zip_on_drive, local_zip)

    extract_dir = Path("/content/_data_extract")
    if extract_dir.exists():
        shutil.rmtree(extract_dir)
    extract_dir.mkdir()

    logger.info("Extracting archive to %s ...", extract_dir)
    with zipfile.ZipFile(local_zip, "r") as zf:
        zf.extractall(extract_dir)

    prepare_dataset_root(extract_dir, local_root, materialize=True)
    shutil.rmtree(extract_dir, ignore_errors=True)

    attach_test_from_drive(local_root, Path(config.drive_project_dir))

    marker.write_text("ready\n", encoding="utf-8")
    from WWR_Segmentation.dataset_layout import _count_images

    train_n = _count_images(local_root / "train" / "images")
    test_n = _count_images(local_root / "test" / "images") if (local_root / "test" / "images").exists() else 0
    logger.info(
        "Dataset ready at %s — train: %d images, test: %d (test optional)",
        local_root,
        train_n,
        test_n,
    )
    return local_root


def setup_colab(
    config: Optional[Config] = None,
    mount_drive: bool = True,
    prepare_data: bool = True,
    force_unzip: bool = False,
    prefer_a100: bool = True,
    load_project: bool = False,
) -> Config:
    """Full Colab setup: optional project load → unzip data → TensorFlow → GPU."""
    setup_logging()
    logger.info("WWR Segmentation code version: %s", CODE_VERSION)

    try:
        from WWR_Segmentation import dataset_layout  # noqa: F401
    except ImportError as exc:
        raise ImportError(
            "Old code.zip detected (missing dataset_layout.py). "
            "Re-upload code.zip from your PC: python scripts/pack_for_colab.py"
        ) from exc

    if load_project:
        ensure_project_on_path(allow_upload=False)

    if config is None:
        config = Config.for_colab()

    if mount_drive:
        mount_google_drive(config.drive_mount_point)

    if config.use_onedrive:
        from WWR_Segmentation.colab_onedrive import mount_onedrive

        mount_onedrive(
            mount_point=config.onedrive_mount_point,
            remote=config.rclone_remote_name,
            config_path=config.rclone_config_path,
        )

    config._create_directories()

    if prepare_data and config.use_local_data_cache:
        prepare_local_dataset(config, force=force_unzip)

    prepare_environment(config)
    verify_gpu(prefer_a100=prefer_a100)

    logger.info("Data (local):  %s", config.data_root)
    logger.info("Checkpoints:   %s", config.checkpoint_dir)
    logger.info("Best model:    %s", config.best_model_path)
    return config
