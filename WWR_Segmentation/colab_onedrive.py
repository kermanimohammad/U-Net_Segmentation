"""Mount Microsoft OneDrive in Google Colab via rclone."""

from __future__ import annotations

import logging
import subprocess
import time
from pathlib import Path

logger = logging.getLogger("wwr_segmentation")

DEFAULT_MOUNT = Path("/content/onedrive")
DEFAULT_CONFIG = Path("/content/rclone.conf")
DEFAULT_REMOTE = "onedrive"


def install_rclone() -> None:
    """Install rclone and FUSE (required once per Colab session)."""
    subprocess.run(["apt-get", "update", "-qq"], check=False)
    subprocess.run(
        ["apt-get", "install", "-qq", "-y", "rclone", "fuse"],
        check=True,
    )
    logger.info("rclone installed.")


def is_mounted(mount_point: Path = DEFAULT_MOUNT) -> bool:
    """Return True if *mount_point* exists and is non-empty."""
    return mount_point.is_dir() and any(mount_point.iterdir())


def mount_onedrive(
    mount_point: Path = DEFAULT_MOUNT,
    remote: str = DEFAULT_REMOTE,
    config_path: Path = DEFAULT_CONFIG,
    install: bool = True,
) -> Path:
    """
    Mount OneDrive at *mount_point* using a pre-configured ``rclone.conf``.

    Setup (one time on your PC):
    1. Install rclone: https://rclone.org/downloads/
    2. Run: ``rclone config`` → New remote → Microsoft OneDrive → name it ``onedrive``
    3. Upload ``rclone.conf`` to Colab (or keep a copy on minimal Google Drive)

    Args:
        mount_point: Local mount path (e.g. ``/content/onedrive``).
        remote: rclone remote name from ``rclone.conf``.
        config_path: Path to ``rclone.conf`` in Colab.
        install: Run apt install for rclone if needed.

    Returns:
        Path to mount point.
    """
    if is_mounted(mount_point):
        logger.info("OneDrive already mounted at %s", mount_point)
        return mount_point

    if install:
        try:
            install_rclone()
        except subprocess.CalledProcessError:
            pass  # rclone may already be present

    if not config_path.exists():
        raise FileNotFoundError(
            f"rclone.conf not found at {config_path}.\n\n"
            "One-time setup:\n"
            "  1. On your PC: install rclone → run 'rclone config' → add OneDrive remote\n"
            "  2. Upload rclone.conf to Colab:\n"
            "       from google.colab import files\n"
            "       files.upload()  # select rclone.conf\n"
            "  3. Re-run this cell."
        )

    mount_point.mkdir(parents=True, exist_ok=True)

    # Kill stale mount if any
    subprocess.run(["fusermount", "-u", str(mount_point)], check=False)

    cmd = [
        "rclone",
        "mount",
        f"{remote}:",
        str(mount_point),
        "--config",
        str(config_path),
        "--vfs-cache-mode",
        "writes",
        "--daemon",
    ]
    logger.info("Mounting OneDrive (%s) → %s", remote, mount_point)
    subprocess.run(cmd, check=True)
    time.sleep(3)

    if not mount_point.exists():
        raise RuntimeError(f"OneDrive mount failed at {mount_point}")

    logger.info("OneDrive ready at %s", mount_point)
    return mount_point
