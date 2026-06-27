"""Discover and normalize dataset folder layouts after unzip."""

from __future__ import annotations

import logging
import os
import shutil
from pathlib import Path
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger("wwr_segmentation")

IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".tif", ".tiff"}


def _count_images(folder: Path) -> int:
    if not folder.exists():
        return 0
    return sum(1 for f in folder.iterdir() if f.suffix.lower() in IMAGE_EXTENSIONS)


def _first_existing(base: Path, relative_paths: Tuple[str, ...]) -> Optional[Path]:
    for rel in relative_paths:
        candidate = base / rel
        if candidate.exists() and _count_images(candidate) > 0:
            return candidate
        if candidate.exists() and any(candidate.iterdir()):
            return candidate
    return None


def _link_or_copy(src: Path, dst: Path) -> None:
    """Symlink *src* to *dst* (fast); fall back to directory junction/copy."""
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists() or dst.is_symlink():
        if dst.is_symlink() or dst.is_file():
            dst.unlink()
        else:
            shutil.rmtree(dst)
    try:
        os.symlink(src.resolve(), dst, target_is_directory=src.is_dir())
    except OSError:
        if src.is_dir():
            shutil.copytree(src, dst)
        else:
            shutil.copy2(src, dst)


def find_standard_layout(search_root: Path) -> Optional[Path]:
    """Return path whose ``train/images`` contains files (searched recursively)."""
    if (search_root / "train" / "images").exists() and _count_images(search_root / "train" / "images") > 0:
        return search_root

    for images_dir in search_root.rglob("images"):
        if images_dir.parent.name == "train" and images_dir.parent.parent.name != "test":
            if _count_images(images_dir) > 0:
                return images_dir.parent.parent

    return None


def _find_test_folders(search_root: Path) -> Tuple[Optional[Path], Optional[Path]]:
    """Locate independent test split folders anywhere under *search_root*."""
    test_images = _first_existing(
        search_root,
        (
            "test_images/final_images",
            "test/final_images",
            "test/images",
            "final_images",
        ),
    )
    test_masks = _first_existing(
        search_root,
        (
            "test_images/final_masks",
            "test/final_masks",
            "test/masks",
            "final_masks",
        ),
    )
    return test_images, test_masks


def find_flat_layout(search_root: Path) -> Optional[Dict[str, Path]]:
    """
    Detect flat ``images/`` + ``masks/`` layout (user's data.zip structure).

    Example::
        data.zip
        ├── images/   (*_texture.png)
        └── masks/    (*_mask.png)
    """
    bases = [search_root]
    nested = search_root / "data"
    if nested.exists():
        bases.append(nested)

    for base in bases:
        images_dir = base / "images"
        masks_dir = base / "masks"
        if not images_dir.is_dir() or not masks_dir.is_dir():
            continue
        if _count_images(images_dir) == 0:
            continue
        # Skip if already part of train/test standard tree
        if base.name in ("train", "test"):
            continue

        test_images, test_masks = _find_test_folders(search_root)
        logger.info(
            "Flat layout at %s — train: %d images, test: %s",
            base,
            _count_images(images_dir),
            _count_images(test_images) if test_images else "not in zip",
        )
        return {
            "train_images": images_dir,
            "train_masks": masks_dir,
            "test_images": test_images,
            "test_masks": test_masks,
        }

    return None


def attach_test_from_drive(target: Path, drive_base: Path) -> None:
    """
    Link test set from Google Drive if not already present locally.

    Looks for ``test_images/final_images`` under *drive_base* or ``test.zip``.
    """
    if (target / "test" / "images").exists() and _count_images(target / "test" / "images") > 0:
        return

    test_images = _first_existing(
        drive_base,
        ("test_images/final_images", "test/final_images", "test/images"),
    )
    test_masks = _first_existing(
        drive_base,
        ("test_images/final_masks", "test/final_masks", "test/masks"),
    )

    if test_images is None:
        test_zip = drive_base / "test.zip"
        if test_zip.exists():
            import zipfile

            tmp = Path("/content/_test_extract")
            if tmp.exists():
                shutil.rmtree(tmp)
            tmp.mkdir()
            with zipfile.ZipFile(test_zip, "r") as zf:
                zf.extractall(tmp)
            test_images, test_masks = _find_test_folders(tmp)
            if test_images is None:
                flat = find_flat_layout(tmp)
                if flat:
                    test_images = flat.get("train_images")
                    test_masks = flat.get("train_masks")
            shutil.rmtree(tmp, ignore_errors=True)

    if test_images is None:
        logger.warning(
            "Test set not found on Drive (%s/test_images). "
            "Upload test_images/ folder or test.zip for evaluation.",
            drive_base,
        )
        return

    _link_or_copy(test_images, target / "test" / "images")
    if test_masks:
        _link_or_copy(test_masks, target / "test" / "masks")
    logger.info(
        "Test set linked from Drive — %d images",
        _count_images(target / "test" / "images"),
    )


def find_legacy_layout(search_root: Path) -> Optional[Dict[str, Path]]:
    """
    Detect legacy folder names from the original U-Net project.

    Handles layouts like::
        data/png_images + data/png_masks + test_images/final_*
    """
    test_images_keys = (
        "test_images/final_images",
        "test/final_images",
        "test/images",
        "final_images",
    )
    test_masks_keys = (
        "test_images/final_masks",
        "test/final_masks",
        "test/masks",
        "final_masks",
    )

    for png_dir in search_root.rglob("png_images"):
        if not png_dir.is_dir():
            continue
        base = png_dir.parent
        train_masks = base / "png_masks"
        if not train_masks.exists():
            continue

        test_images = _first_existing(base, test_images_keys) or _first_existing(
            search_root, test_images_keys
        )
        test_masks = _first_existing(base, test_masks_keys) or _first_existing(
            search_root, test_masks_keys
        )

        layout = {
            "train_images": png_dir,
            "train_masks": train_masks,
            "test_images": test_images,
            "test_masks": test_masks,
        }
        logger.info(
            "Legacy layout — train: %d images at %s, test: %s",
            _count_images(png_dir),
            png_dir,
            _count_images(test_images) if test_images else "not found",
        )
        return layout

    return None


def normalize_to_standard(source: Dict[str, Path], target: Path) -> Path:
    """
    Map legacy folders to the standard layout at *target*:

    ``train/images``, ``train/masks``, ``test/images``, ``test/masks``
    """
    if target.exists():
        shutil.rmtree(target)
    target.mkdir(parents=True)

    mapping = {
        "train/images": source["train_images"],
        "train/masks": source["train_masks"],
    }
    if source.get("test_images"):
        mapping["test/images"] = source["test_images"]
    if source.get("test_masks"):
        mapping["test/masks"] = source["test_masks"]
    elif source.get("test_images"):
        logger.warning(
            "Test images found but test masks missing — evaluation may fail."
        )

    for rel, src in mapping.items():
        dst = target / rel
        logger.info("Linking %s → %s", src, dst)
        _link_or_copy(src, dst)

    return target


def prepare_dataset_root(extract_parent: Path, target: Path) -> Path:
    """
    Locate or normalize extracted data into *target* (``/content/data``).

    Supports:
    - Standard: ``data/train/images``, ``data/test/images``
    - Flat: ``images/``, ``masks/`` (train data in data.zip)
    - Legacy: ``png_images``, ``png_masks``, ``test_images/final_*``
    """
    standard = find_standard_layout(extract_parent)
    if standard is not None and standard != target:
        if target.exists():
            shutil.rmtree(target)
        shutil.move(str(standard), str(target))
        standard = target

    if standard is not None:
        logger.info(
            "Standard layout at %s — train: %d, test: %d",
            target,
            _count_images(target / "train" / "images"),
            _count_images(target / "test" / "images"),
        )
        return target

    flat = find_flat_layout(extract_parent)
    if flat is not None:
        normalize_to_standard(flat, target)
        logger.info(
            "Normalized flat layout → %s — train: %d, test: %d",
            target,
            _count_images(target / "train" / "images"),
            _count_images(target / "test" / "images"),
        )
        return target

    legacy = find_legacy_layout(extract_parent)
    if legacy is not None:
        normalize_to_standard(legacy, target)
        logger.info(
            "Normalized legacy layout → %s — train: %d, test: %d",
            target,
            _count_images(target / "train" / "images"),
            _count_images(target / "test" / "images"),
        )
        return target

    # List top-level contents to help debugging
    contents: List[str] = []
    for p in sorted(extract_parent.rglob("*")):
        if p.is_dir() and p.parent == extract_parent or (p.parent.name == "data" and p.parent.parent == extract_parent):
            contents.append(str(p.relative_to(extract_parent)))
    hint = "\n  ".join(contents[:20]) if contents else "(empty)"

    raise FileNotFoundError(
        f"Could not find dataset under {extract_parent}.\n"
        f"Expected either:\n"
        f"  data/train/images/  +  data/test/images/\n"
        f"  OR flat: images/  masks/  (train data)\n"
        f"  OR legacy: png_images/  png_masks/\n\n"
        f"Top-level folders found:\n  {hint}\n\n"
        f"Re-pack data.zip with the correct folder structure."
    )
