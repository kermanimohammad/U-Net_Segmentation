"""
Standalone Colab bootstrap — no project imports required.

Copy this file into a Colab cell, or run before ``from WWR_Segmentation...``.
"""

from __future__ import annotations

import os
import shutil
import sys
import zipfile
from pathlib import Path

LOCAL_REPO = Path("/content/U-Net_Segmentation")
DRIVE_BASE = Path("/content/drive/MyDrive/WWR_Seg_Model")
CODE_ZIP_DRIVE = DRIVE_BASE / "code.zip"


def _has_package(root: Path) -> bool:
    return (root / "WWR_Segmentation" / "__init__.py").exists()


def find_on_drive() -> Path | None:
    """Search WWR_Seg_Model on Drive for the project root."""
    if not DRIVE_BASE.exists():
        return None

    candidates = [
        DRIVE_BASE / "code",
        DRIVE_BASE / "U-Net_Segmentation",
        DRIVE_BASE,
        LOCAL_REPO,
    ]
    for candidate in candidates:
        if _has_package(candidate):
            return candidate.resolve()

    for init_file in DRIVE_BASE.rglob("WWR_Segmentation/__init__.py"):
        return init_file.parent.parent.resolve()

    return None


def _normalize_extracted_root(extract_parent: Path, target: Path) -> Path | None:
    """Find project root after zip extraction."""
    if _has_package(target):
        return target

    for name in ("U-Net_Segmentation", "U-Net_Segmentation-main", "code"):
        nested = extract_parent / name
        if _has_package(nested):
            if target.exists():
                shutil.rmtree(target)
            shutil.move(str(nested), str(target))
            return target

    for init_file in extract_parent.rglob("WWR_Segmentation/__init__.py"):
        root = init_file.parent.parent
        if root != target:
            if target.exists():
                shutil.rmtree(target)
            shutil.move(str(root), str(target))
        return target

    return None


def extract_zip(zip_path: Path, target: Path = LOCAL_REPO) -> Path:
    """Extract a code zip archive to *target*."""
    extract_parent = Path("/content/_extract")
    if extract_parent.exists():
        shutil.rmtree(extract_parent)
    extract_parent.mkdir()

    print(f"Extracting {zip_path.name} ...")
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(extract_parent)

    root = _normalize_extracted_root(extract_parent, target)
    shutil.rmtree(extract_parent, ignore_errors=True)

    if root is None or not _has_package(root):
        raise FileNotFoundError(f"No WWR_Segmentation package found in {zip_path}")

    return root


def copy_to_local(source: Path, target: Path = LOCAL_REPO) -> Path:
    """Copy project from Drive to local SSD for faster I/O."""
    if source.resolve() == target.resolve():
        return target

    print(f"Copying project to local SSD: {target}")
    if target.exists():
        shutil.rmtree(target)
    shutil.copytree(
        source,
        target,
        ignore=shutil.ignore_patterns("__pycache__", "*.pyc", ".git"),
    )
    return target


def upload_code_zip() -> Path:
    """Prompt user to upload code.zip in Colab."""
    from google.colab import files

    print("Upload code.zip (project archive)...")
    uploaded = files.upload()
    if not uploaded:
        raise FileNotFoundError("No file uploaded.")

    zip_name = next(iter(uploaded))
    zip_path = Path(f"/content/{zip_name}")
    zip_path.write_bytes(uploaded[zip_name])
    return extract_zip(zip_path)


def bootstrap_colab(
    allow_upload: bool = True,
    use_local_copy: bool = True,
) -> Path:
    """
    Load WWR_Segmentation in Colab (Google Drive only — no GitHub).

    Order:
    1. Find project folder on Drive
    2. Extract ``My Drive/WWR_Seg_Model/code.zip``
    3. Upload ``code.zip`` interactively (if *allow_upload*)
    """
    from google.colab import drive

    if not Path("/content/drive/MyDrive").exists():
        drive.mount("/content/drive")

    project_root: Path | None = find_on_drive()

    if project_root is None and CODE_ZIP_DRIVE.exists():
        project_root = extract_zip(CODE_ZIP_DRIVE)

    if project_root is None and allow_upload:
        project_root = upload_code_zip()

    if project_root is None:
        raise FileNotFoundError(
            "Project code not found.\n\n"
            "On your PC run:\n"
            "  python scripts/pack_for_colab.py\n\n"
            "Then upload code.zip to:\n"
            "  My Drive/WWR_Seg_Model/code.zip\n\n"
            "Re-run this cell (or set allow_upload=True to upload now)."
        )

    if use_local_copy and str(project_root).startswith("/content/drive"):
        project_root = copy_to_local(project_root)

    os.chdir(project_root)
    root_str = str(project_root)
    if root_str not in sys.path:
        sys.path.insert(0, root_str)

    print(f"Project root : {project_root}")
    print(f"Package OK   : {_has_package(project_root)}")
    return project_root
