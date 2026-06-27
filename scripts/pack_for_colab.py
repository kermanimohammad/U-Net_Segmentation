"""Create code.zip for Google Colab upload (My Drive/WWR_Seg_Model/code.zip)."""

from __future__ import annotations

import shutil
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUT = PROJECT_ROOT / "code.zip"

INCLUDE = (
    "WWR_Segmentation",
    "WWR_Seg_Model.ipynb",
    "requirements.txt",
    "AI_Guide.md",
)


def main() -> None:
    if OUTPUT.exists():
        OUTPUT.unlink()

    staging = PROJECT_ROOT / "_colab_staging"
    if staging.exists():
        shutil.rmtree(staging)
    staging.mkdir()

    for name in INCLUDE:
        src = PROJECT_ROOT / name
        if not src.exists():
            print(f"WARNING: missing {name}", file=sys.stderr)
            continue
        dest = staging / name
        if src.is_dir():
            shutil.copytree(src, dest)
        else:
            shutil.copy2(src, dest)

    shutil.make_archive(str(OUTPUT.with_suffix("")), "zip", staging)
    shutil.rmtree(staging)

    size_mb = OUTPUT.stat().st_size / 1e6
    print(f"Created: {OUTPUT}")
    print(f"Size:    {size_mb:.1f} MB")
    print("Upload to Google Drive: My Drive/WWR_Seg_Model/code.zip")


if __name__ == "__main__":
    main()
