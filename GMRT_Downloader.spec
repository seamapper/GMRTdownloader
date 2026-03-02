# -*- mode: python ; coding: utf-8 -*-
"""
PyInstaller spec file for GMRT Downloader Windows executable
Updated for current project structure
"""

import re
from PyInstaller.utils.hooks import collect_all

# Extract version from GMRT_Downloader.py (get the last uncommented version)
version = "unknown"
try:
    with open('GMRT_Downloader.py', 'r', encoding='utf-8') as f:
        lines = f.readlines()
        # Look for the last uncommented __version__ line
        for line in reversed(lines):
            # Match __version__ that is not commented out (line doesn't start with #)
            stripped = line.lstrip()
            if not stripped.startswith('#'):
                match = re.search(r'__version__\s*=\s*["\']([^"\']+)["\']', line)
                if match:
                    version = match.group(1)
                    break
except Exception:
    pass

# Include media files and config
datas = [
    ('media', 'media'),
    ('gmrtgrab_config.json', '.'),
]
binaries = []

# Hidden imports - modules that PyInstaller might miss
hiddenimports = [
    'PyQt6.QtCore',
    'PyQt6.QtGui',
    'PyQt6.QtWidgets',
    'requests',
    'json',
    'datetime',
    'time',
    'os',
    'sys',
    'numpy',
    'shutil',
    'math',
    'rasterio',
    'rasterio.warp',
    'rasterio.windows',
    'rasterio.transform',
    'rasterio.enums',
    'netCDF4',
]

# Collect rasterio data files and dependencies
try:
    tmp_ret = collect_all('rasterio')
    datas += tmp_ret[0]
    binaries += tmp_ret[1]
    hiddenimports += tmp_ret[2]
except:
    pass

a = Analysis(
    ['GMRT_Downloader.py'],  # Main script
    pathex=[],                # Additional paths to search
    binaries=binaries,
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],            # Custom hooks directory
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],             # Modules to exclude (add unused packages here)
    noarchive=False,
    optimize=0,
)

pyz = PYZ(a.pure, a.zipped_data, cipher=None)

exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.datas,
    [],
    name=f'GMRT_Bathymetry_Downloader_v{version}',  # Output executable name with version
    debug=False,                         # Set to True for debugging
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,                           # Use UPX compression (smaller exe)
    upx_exclude=[],
    runtime_tmpdir=None,
    console=False,                      # No console window (GUI app)
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    icon='media\\GMRT-logo2020.ico',   # Application icon
)

