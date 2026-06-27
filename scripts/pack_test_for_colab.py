"""Pack test_images folder as test.zip for Google Colab upload."""

from __future__ import annotations

import shutil
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
TEST_DIR = PROJECT_ROOT / "test_images"
OUTPUT = PROJECT_ROOT / "test.zip"


def main() -> None:
    if not TEST_DIR.exists():
        print(f"ERROR: {TEST_DIR} not found", file=sys.stderr)
        sys.exit(1)

    if OUTPUT.exists():
        OUTPUT.unlink()

    shutil.make_archive(str(OUTPUT.with_suffix("")), "zip", TEST_DIR)
    size_mb = OUTPUT.stat().st_size / 1e6
    print(f"Created: {OUTPUT}")
    print(f"Size:    {size_mb:.1f} MB")
    print("Upload to Google Drive: My Drive/WWR_Seg_Model/test.zip")
    print("  OR upload the test_images/ folder directly to that location.")


if __name__ == "__main__":
    main()
