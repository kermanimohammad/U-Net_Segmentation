"""Google Colab setup: project path, Drive mount, local dataset cache, GPU check."""

from __future__ import annotations

import logging
import os
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path
from typing import Optional

from WWR_Segmentation.config import Config
from WWR_Segmentation.utils import mount_google_drive, prepare_environment, setup_logging

logger = logging.getLogger("wwr_segmentation")

REPO_URL = "https://github.com/kermanimohammad/U-Net_Segmentation.git"
LOCAL_REPO = Path("/content/U-Net_Segmentation")

DRIVE_PROJECT_CANDIDATES = (
    LOCAL_REPO,
    Path("/content/drive/MyDrive/WWR_Seg_Model/code"),
    Path("/content/drive/MyDrive/WWR_Seg_Model/U-Net_Segmentation"),
    Path("/content/drive/MyDrive/WWR_Seg_Model"),
    Path.cwd(),
)


def find_project_root() -> Optional[Path]:
    """Return the first directory that contains the ``WWR_Segmentation`` package."""
    for candidate in DRIVE_PROJECT_CANDIDATES:
        if (candidate / "WWR_Segmentation" / "__init__.py").exists():
            return candidate.resolve()
    return None


def download_from_github_zip(
    zip_url: str = "https://github.com/kermanimohammad/U-Net_Segmentation/archive/refs/heads/main.zip",
    target: Path = LOCAL_REPO,
) -> Path:
    """Download repository as ZIP when git clone fails in Colab."""
    import urllib.request

    zip_path = Path("/content/repo_main.zip")
    logger.info("Downloading project ZIP from GitHub...")
    urllib.request.urlretrieve(zip_url, zip_path)

    if target.exists():
        shutil.rmtree(target)

    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall("/content")

    extracted = Path("/content/U-Net_Segmentation-main")
    shutil.move(str(extracted), str(target))
    zip_path.unlink(missing_ok=True)
    return target


def ensure_project_on_path(
    repo_url: str = REPO_URL,
    clone_dir: Path = LOCAL_REPO,
    mount_drive: bool = True,
) -> Path:
    """
    Make ``WWR_Segmentation`` importable in Colab.

    Search order:
    1. Local clone / Drive folders
    2. ``code.zip`` on Google Drive
    3. ``git clone``
    4. GitHub ZIP download (fallback)
    """
    if mount_drive:
        mount_google_drive(Path("/content/drive"))

    project_root = find_project_root()

    if project_root is None:
        code_zip = Path("/content/drive/MyDrive/WWR_Seg_Model/code.zip")
        if code_zip.exists():
            logger.info("Extracting code.zip from Drive...")
            if clone_dir.exists():
                shutil.rmtree(clone_dir)
            with zipfile.ZipFile(code_zip, "r") as zf:
                zf.extractall(clone_dir.parent)
            if (clone_dir / "WWR_Segmentation").exists():
                project_root = clone_dir
            else:
                nested = clone_dir.parent / "U-Net_Segmentation"
                if nested.exists():
                    shutil.move(str(nested), str(clone_dir))
                    project_root = clone_dir

    if project_root is None:
        if clone_dir.exists():
            shutil.rmtree(clone_dir)
        logger.info("Cloning project from %s ...", repo_url)
        result = subprocess.run(
            ["git", "clone", "--depth", "1", repo_url, str(clone_dir)],
            capture_output=True,
            text=True,
        )
        if result.returncode == 0:
            project_root = clone_dir
        else:
            logger.warning("git clone failed: %s", result.stderr.strip())
            project_root = download_from_github_zip(target=clone_dir)

    if not (project_root / "WWR_Segmentation" / "__init__.py").exists():
        raise FileNotFoundError(
            f"WWR_Segmentation package not found under {project_root}."
        )

    os.chdir(project_root)
    root_str = str(project_root)
    if root_str not in sys.path:
        sys.path.insert(0, root_str)

    logger.info("Project root: %s", project_root)
    return project_root


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


def _find_dataset_root(extract_parent: Path, expected: Path) -> Path:
    """Locate dataset root after unzip (supports ``data/train/...`` or ``train/...``)."""
    if (expected / "train" / "images").exists():
        return expected
    nested = extract_parent / "data"
    if (nested / "train" / "images").exists():
        return nested
    if (extract_parent / "train" / "images").exists():
        return extract_parent
    raise FileNotFoundError(
        f"train/images not found after unzip. Expected under {expected} or {nested}."
    )


def prepare_local_dataset(config: Config, force: bool = False) -> Path:
    """Copy ``data.zip`` from Drive and extract to ``/content/data`` for fast I/O."""
    local_root = Path(config.local_data_root)
    marker = local_root / ".dataset_ready"
    zip_on_drive = Path(config.drive_data_zip)

    if marker.exists() and not force:
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

    if force and local_root.exists():
        shutil.rmtree(local_root)

    logger.info("Extracting to %s ...", local_root.parent)
    with zipfile.ZipFile(local_zip, "r") as zf:
        zf.extractall(local_root.parent)

    dataset_root = _find_dataset_root(local_root.parent, local_root)
    if dataset_root != local_root:
        if local_root.exists():
            shutil.rmtree(local_root)
        shutil.move(str(dataset_root), str(local_root))

    marker.write_text("ready\n", encoding="utf-8")
    logger.info("Dataset ready at %s", local_root)
    return local_root


def setup_colab(
    config: Optional[Config] = None,
    mount_drive: bool = True,
    prepare_data: bool = True,
    force_unzip: bool = False,
    prefer_a100: bool = True,
    clone_if_missing: bool = True,
) -> Config:
    """
    Full Colab setup: project path → Drive → unzip data → TensorFlow → GPU check.
    """
    setup_logging()

    if clone_if_missing:
        ensure_project_on_path(mount_drive=mount_drive)
        mount_drive = False  # already mounted

    if config is None:
        config = Config.for_colab()

    if mount_drive:
        mount_google_drive(config.drive_mount_point)
        config._create_directories()

    if prepare_data and config.use_local_data_cache:
        prepare_local_dataset(config, force=force_unzip)

    prepare_environment(config)
    verify_gpu(prefer_a100=prefer_a100)

    logger.info("Data (local):  %s", config.data_root)
    logger.info("Checkpoints:   %s", config.checkpoint_dir)
    logger.info("Best model:    %s", config.best_model_path)
    return config
