# -*- mode: python ; coding: utf-8 -*-
"""
PyInstaller spec file for GMRT Downloader Windows executable
Updated for current project structure
"""

from PyInstaller.utils.hooks import collect_all

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
    name='GMRT_Bathymetry_Downloader',  # Output executable name
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
    icon='media\\mgds.ico',            # Application icon
)

