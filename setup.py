"""
Setup script for creating a Mac .app bundle with py2app.

Install py2app first (do not use setup_requires):
    pip install py2app PyQt6 requests numpy
    pip install rasterio netCDF4   # optional formats

Build (macOS only):
    python setup.py py2app

Use a dedicated build venv without PyInstaller installed — py2app's scanner
will try to import PyInstaller.hooks.* and fail on hook-PyQt5 / hook-PySide6 names.
"""

from __future__ import annotations

import re

from setuptools import setup

version = "unknown"
try:
    with open("config.py", encoding="utf-8") as f:
        for line in f:
            match = re.search(r'__version__\s*=\s*["\']([^"\']+)["\']', line)
            if match and not line.lstrip().startswith("#"):
                version = match.group(1)
except OSError:
    pass

APP = ["GMRT_Downloader.py"]
DATA_FILES = [
    (
        "media",
        [
            "media/GMRT-logo2020.ico",
            "media/GMRT-logo2020.png",
            "media/GMRT-logo2020.icns",
        ],
    ),
    ("", ["gmrtgrab_config.json"]),
]

# Packages that must never be bundled (break py2app modulegraph or bloat the app).
_BUILD_TOOL_EXCLUDES = [
    "PyInstaller",
    "PyInstaller.hooks",
    "_pyinstaller_hooks_contrib",
    "pyinstaller_hooks_contrib",
    "py2exe",
    "cx_Freeze",
    "setuptools",
    "pip",
    "wheel",
]

_QT_EXCLUDES = [
    "PyQt5",
    "PySide2",
    "PySide6",
]

_OTHER_EXCLUDES = [
    "matplotlib",
    "scipy",
    "pandas",
    "IPython",
    "jupyter",
    "test",
    "tests",
    "numpy.tests",
    "pytest",
]

OPTIONS = {
    "argv_emulation": True,
    "iconfile": "media/GMRT-logo2020.icns",
    "plist": {
        "CFBundleName": "GMRT Downloader",
        "CFBundleDisplayName": "GMRT Bathymetry Grid Downloader",
        "CFBundleGetInfoString": f"GMRT Bathymetry Grid Downloader v{version}",
        "CFBundleIdentifier": "org.gmrt.downloader",
        "CFBundleVersion": version,
        "CFBundleShortVersionString": version,
        "NSHighResolutionCapable": True,
        "NSRequiresAquaSystemAppearance": False,
        "LSMinimumSystemVersion": "10.13",
        "NSHumanReadableCopyright": "Copyright © 2026 Paul Johnson",
        "NSAppTransportSecurity": {
            "NSAllowsArbitraryLoads": True,
        },
    },
    "includes": [
        "config",
        "converters",
        "PyQt6.QtCore",
        "PyQt6.QtGui",
        "PyQt6.QtWidgets",
        "requests",
        "numpy",
    ],
    "packages": [
        "ui",
        "workers",
        "rasterio",
        "netCDF4",
    ],
    "excludes": _BUILD_TOOL_EXCLUDES + _QT_EXCLUDES + _OTHER_EXCLUDES,
}

setup(
    name="GMRT Downloader",
    app=APP,
    data_files=DATA_FILES,
    options={"py2app": OPTIONS},
)
