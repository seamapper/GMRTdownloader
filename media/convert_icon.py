#!/usr/bin/env python3
"""
Build Windows (.ico) and macOS (.icns) application icons from GMRT-logo2020.png.

Requires Pillow: pip install Pillow

Run from the project root:
    python media/convert_icon.py
"""

from __future__ import annotations

import os
import sys

from PIL import Image

_MEDIA_DIR = os.path.dirname(os.path.abspath(__file__))
PNG_PATH = os.path.join(_MEDIA_DIR, "GMRT-logo2020.png")
ICO_PATH = os.path.join(_MEDIA_DIR, "GMRT-logo2020.ico")
ICNS_PATH = os.path.join(_MEDIA_DIR, "GMRT-logo2020.icns")

ICO_SIZES = [(16, 16), (32, 32), (48, 48), (64, 64), (128, 128), (256, 256)]


def _load_rgba_png() -> Image.Image:
    if not os.path.exists(PNG_PATH):
        raise FileNotFoundError(f"{PNG_PATH} not found")
    img = Image.open(PNG_PATH)
    if img.mode != "RGBA":
        img = img.convert("RGBA")
    return img


def convert_png_to_ico() -> bool:
    """Write GMRT-logo2020.ico from the PNG source."""
    try:
        img = _load_rgba_png()
        img.save(ICO_PATH, format="ICO", sizes=ICO_SIZES)
        print(f"Wrote {ICO_PATH}")
        return True
    except Exception as exc:
        print(f"Error writing ICO: {exc}", file=sys.stderr)
        return False


def convert_png_to_icns() -> bool:
    """Write GMRT-logo2020.icns from the PNG source (Pillow, any OS)."""
    try:
        img = _load_rgba_png()
        img.save(ICNS_PATH, format="ICNS")
        print(f"Wrote {ICNS_PATH}")
        return True
    except Exception as exc:
        print(f"Error writing ICNS: {exc}", file=sys.stderr)
        return False


def main() -> int:
    ok_ico = convert_png_to_ico()
    ok_icns = convert_png_to_icns()
    return 0 if ok_ico and ok_icns else 1


if __name__ == "__main__":
    sys.exit(main())
