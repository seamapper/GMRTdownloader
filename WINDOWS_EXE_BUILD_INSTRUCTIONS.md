# Creating a Windows .exe from GMRT Downloader

This guide provides step-by-step instructions for converting the GMRT Bathymetry Grid Downloader Python application into a standalone Windows executable (.exe) file.

## Table of Contents

1. [Prerequisites](#prerequisites)
2. [Method 1: Using PyInstaller (Recommended)](#method-1-using-pyinstaller-recommended)
3. [Method 2: Using cx_Freeze (Alternative)](#method-2-using-cx_freeze-alternative)
4. [Method 3: Using py2exe (Legacy)](#method-3-using-py2exe-legacy)
5. [Creating a Distribution Package](#creating-a-distribution-package)
6. [Code Signing (Optional)](#code-signing-optional)
7. [Troubleshooting](#troubleshooting)

---

## Prerequisites

### Required Software

1. **Windows 10/11** (64-bit recommended)
2. **Python 3.8+** (Python 3.9 or 3.10 recommended)
3. **pip** (usually comes with Python)

### Required Python Packages

Install the required packages in a virtual environment:

```bash
# Create a virtual environment (recommended)
python -m venv venv
venv\Scripts\activate

# Install dependencies
pip install PyQt6
pip install requests
pip install numpy
pip install rasterio  # Optional but recommended for GeoTIFF support
pip install netCDF4   # Optional but recommended for NetCDF support
```

### Optional: Install Microsoft Visual C++ Redistributable

Some Python packages (like numpy) may require the Visual C++ Redistributable. Download and install from:
- [Microsoft Visual C++ Redistributable](https://aka.ms/vs/17/release/vc_redist.x64.exe)

---

## Method 1: Using PyInstaller (Recommended)

PyInstaller is the most popular and reliable tool for creating Windows executables from Python applications.

### Step 1: Install PyInstaller

```bash
pip install pyinstaller
```

### Step 2: Create a .spec File

Create a file named `GMRT_Downloader.spec` in your project directory:

```python
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
```

### Step 3: Build the Executable

#### Option A: Using the .spec File (Recommended)

```bash
pyinstaller GMRT_Downloader.spec
```

This creates:
- `dist/GMRT_Bathymetry_Downloader_v{version}.exe` - The executable file (version extracted from GMRT_Downloader.py)
- `build/` - Temporary build files (can be deleted)

#### Option B: Using Command Line (Quick Test)

For a quick test without creating a spec file:

```bash
pyinstaller --onefile --windowed --name="GMRT_Bathymetry_Downloader" --icon=media\GMRT-logo2020.ico --add-data "media;media" --add-data "gmrtgrab_config.json;." GMRT_Downloader.py
```

**Command breakdown:**
- `--onefile` - Create a single executable file
- `--windowed` or `-w` - No console window (GUI app)
- `--name` - Name of the output executable
- `--icon` - Path to icon file
- `--add-data` - Include data files (format: `source;destination`)
- `GMRT_Downloader.py` - Main script

### Step 4: Test the Executable

```bash
# Navigate to dist folder
cd dist

# Run the executable (replace {version} with actual version, e.g., v2025.06)
.\GMRT_Bathymetry_Downloader_v{version}.exe
```

### Step 5: Clean Build (Optional)

To clean previous builds:

```bash
# Remove build artifacts
rmdir /s /q build dist
```

Or manually delete the `build` and `dist` folders.

---

## Method 2: Using cx_Freeze (Alternative)

cx_Freeze is another option for creating Windows executables.

### Step 1: Install cx_Freeze

```bash
pip install cx_Freeze
```

### Step 2: Create setup.py

Create a file named `setup_cx.py`:

```python
from cx_Freeze import setup, Executable
import sys

# Include files
include_files = [
    'media',
    'gmrtgrab_config.json',
]

# Packages to include
packages = [
    'PyQt6',
    'requests',
    'numpy',
    'rasterio',
    'netCDF4',
    'json',
    'datetime',
]

# Build options
build_exe_options = {
    'packages': packages,
    'include_files': include_files,
    'excludes': ['matplotlib', 'scipy', 'pandas'],
    'optimize': 2,
}

# Executable configuration
exe = Executable(
    script='GMRT_Downloader.py',
    base='Win32GUI',  # Use 'Win32GUI' for no console, 'Console' for console
    icon='media\\GMRT-logo2020.ico',
    target_name='GMRT_Bathymetry_Downloader.exe',
)

setup(
    name='GMRT Bathymetry Grid Downloader',
    version='2025.05',
    description='GMRT Bathymetry Grid Downloader',
    options={'build_exe': build_exe_options},
    executables=[exe],
)
```

### Step 3: Build

```bash
python setup_cx.py build
```

The executable will be in `build/exe.win-amd64-3.x/` directory.

---

## Method 3: Using py2exe (Legacy)

**Note:** py2exe is no longer actively maintained but still works for simple applications.

### Step 1: Install py2exe

```bash
pip install py2exe
```

### Step 2: Create setup.py

```python
from distutils.core import setup
import py2exe

setup(
    windows=[{
        'script': 'GMRT_Downloader.py',
        'icon_resources': [(1, 'media\\mgds.ico')],
    }],
    options={
        'py2exe': {
            'packages': ['PyQt6', 'requests', 'numpy'],
            'includes': ['rasterio', 'netCDF4'],
        }
    },
    data_files=[('media', ['media\\mgds.ico']), ('', ['gmrtgrab_config.json'])],
)
```

### Step 3: Build

```bash
python setup.py py2exe
```

---

## Creating a Distribution Package

### Option 1: Single Executable (One-File)

The PyInstaller spec file above creates a single executable. This is the simplest distribution method:

**Pros:**
- Single file to distribute
- Easy for users

**Cons:**
- Slower startup (extracts to temp folder on each run)
- Larger file size

### Option 2: One-Directory (One-Folder)

Modify the spec file to create a folder distribution:

```python
# In the spec file, replace the EXE section with:
exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,  # Changed from False
    name='GMRT_Bathymetry_Downloader',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=False,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    icon='media\\GMRT-logo2020.ico',
)

coll = COLLECT(
    exe,
    a.binaries,
    a.zipfiles,
    a.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name='GMRT_Bathymetry_Downloader',
)
```

**Pros:**
- Faster startup
- Smaller executable

**Cons:**
- Multiple files to distribute
- Need to distribute entire folder

### Creating a ZIP Archive

For distribution, create a ZIP file:

```powershell
# Using PowerShell (replace {version} with actual version, e.g., v2025.06)
Compress-Archive -Path dist\GMRT_Bathymetry_Downloader_v*.exe -DestinationPath GMRT_Downloader_v{version}.zip
```

Or manually:
1. Right-click on `dist/GMRT_Bathymetry_Downloader_v{version}.exe` (replace {version} with actual version)
2. Send to → Compressed (zipped) folder

### Creating an Installer (Advanced)

For a professional installer, use tools like:

1. **Inno Setup** (Free, recommended)
   - Download from: https://jrsoftware.org/isinfo.php
   - Create an installer script (.iss file)

2. **NSIS (Nullsoft Scriptable Install System)** (Free)
   - Download from: https://nsis.sourceforge.io/

3. **Advanced Installer** (Commercial, has free version)

---

## Code Signing (Optional)

Code signing helps prevent Windows Defender and antivirus warnings.

### Prerequisites

1. **Code Signing Certificate** from a trusted Certificate Authority (CA)
   - DigiCert, GlobalSign, Sectigo, etc.
   - Costs typically $100-500/year

2. **signtool.exe** (comes with Windows SDK)

### Signing the Executable

```bash
# Sign the executable (replace {version} with actual version)
signtool sign /f "path\to\certificate.pfx" /p "password" /t http://timestamp.digicert.com /d "GMRT Bathymetry Grid Downloader" /v "dist\GMRT_Bathymetry_Downloader_v{version}.exe"

# Verify signature
signtool verify /pa /v "dist\GMRT_Bathymetry_Downloader_v{version}.exe"
```

**Note:** Code signing is optional for personal/internal use but recommended for public distribution.

---

## Troubleshooting

### Common Issues

#### 1. "Failed to execute script" Error

**Problem:** Executable crashes immediately with "Failed to execute script" error.

**Solutions:**
- Build with `console=True` temporarily to see error messages
- Check that all dependencies are included in `hiddenimports`
- Verify all data files are included in `datas`

```python
# In spec file, change:
console=True,  # Temporarily enable console
```

#### 2. Missing DLL Errors

**Problem:** "The program can't start because MSVCP140.dll is missing" or similar.

**Solution:** Install Microsoft Visual C++ Redistributable:
- Download from: https://aka.ms/vs/17/release/vc_redist.x64.exe
- Or include it in your distribution

#### 3. PyQt6 Not Working

**Problem:** GUI doesn't appear or crashes.

**Solutions:**
- Ensure PyQt6 plugins are included:

```python
# Add to hiddenimports in spec file:
hiddenimports = [
    # ... existing imports ...
    'PyQt6.QtCore',
    'PyQt6.QtGui',
    'PyQt6.QtWidgets',
    'PyQt6.QtOpenGL',
]
```

- Check that icon file exists and is valid

#### 4. Large Executable Size

**Problem:** Executable is very large (>200MB).

**Solutions:**
- Use UPX compression (already enabled in spec file)
- Exclude unused packages:

```python
excludes=['matplotlib', 'scipy', 'pandas', 'IPython', 'jupyter'],
```

- Use one-folder distribution instead of one-file

#### 5. Network Requests Failing

**Problem:** App can't connect to GMRT servers.

**Solutions:**
- Ensure `requests` is in `hiddenimports`
- Check Windows Firewall settings
- Test with `console=True` to see error messages

#### 6. Import Errors for Optional Dependencies

**Problem:** Errors about missing rasterio or netCDF4.

**Solution:** These are optional in your code (wrapped in try/except). You can:
- Exclude them if not needed: `excludes=['rasterio', 'netCDF4']`
- Or ensure they're properly installed before building

#### 7. Config File Not Found

**Problem:** `gmrtgrab_config.json` not found at runtime.

**Solutions:**
- Ensure it's included in `datas` in spec file
- Check the path in your code matches the bundled location

```python
# In your Python code, use:
if getattr(sys, 'frozen', False):
    # Running as compiled executable
    base_path = sys._MEIPASS
else:
    # Running as script
    base_path = os.path.dirname(__file__)
    
config_file = os.path.join(base_path, 'gmrtgrab_config.json')
```

#### 8. Antivirus False Positives

**Problem:** Windows Defender or antivirus flags the .exe as suspicious.

**Solutions:**
- Code sign the executable (see Code Signing section)
- Submit to antivirus vendors for whitelisting
- Add exclusion in antivirus settings

#### 9. "Windows protected your PC" Warning

**Problem:** Windows SmartScreen shows warning on first run.

**Solutions:**
- Code sign the executable
- Distribute through a trusted source
- Users can click "More info" → "Run anyway" for first run

### Debugging Tips

1. **Enable Console Window:**

```python
# In spec file:
console=True,  # Shows console window with error messages
```

2. **Check Build Log:**

```bash
# PyInstaller outputs detailed logs
pyinstaller GMRT_Downloader.spec --log-level=DEBUG
```

3. **Test with Console Enabled:**

```bash
# Build with console to see errors
pyinstaller --onefile --console GMRT_Downloader.py
```

4. **Verify Included Files:**

```bash
# Check what's in the executable
pyinstaller --onefile --log-level=DEBUG GMRT_Downloader.spec
# Look for "Analyzing" messages to see what's included
```

5. **Test in Clean Environment:**

- Test on a clean Windows VM or different machine
- Ensures all dependencies are included

---

## Quick Start Checklist

- [ ] Install Python 3.8+ and required packages
- [ ] Install PyInstaller: `pip install pyinstaller`
- [ ] Create or update `GMRT_Downloader.spec` file
- [ ] Ensure `media/GMRT-logo2020.ico` exists
- [ ] Ensure `gmrtgrab_config.json` exists
- [ ] Build: `pyinstaller GMRT_Downloader.spec`
- [ ] Test: Run `dist/GMRT_Bathymetry_Downloader_v*.exe` (versioned executable)
- [ ] Test on a different machine (if possible)
- [ ] Create ZIP archive for distribution
- [ ] (Optional) Code sign the executable

---

## Example Build Script

Create a batch file `build.bat` to automate the build process:

```batch
@echo off
echo Building GMRT Downloader executable...

REM Activate virtual environment (if using one)
call venv\Scripts\activate

REM Clean previous builds
if exist build rmdir /s /q build
if exist dist rmdir /s /q dist

REM Build the executable
pyinstaller GMRT_Downloader.spec

REM Check if build was successful (check for versioned executable)
set BUILD_SUCCESS=0
for %%f in (dist\GMRT_Bathymetry_Downloader_v*.exe) do (
    echo Build successful!
    echo Executable location: dist\%%~nxf
    set BUILD_SUCCESS=1
    goto :build_check_done
)
:build_check_done
if %BUILD_SUCCESS%==0 (
    echo Build failed!
    exit /b 1
)

pause
```

Run with:
```bash
build.bat
```

---

## Additional Resources

- [PyInstaller Documentation](https://pyinstaller.readthedocs.io/)
- [PyInstaller Manual](https://pyinstaller.org/en/stable/)
- [cx_Freeze Documentation](https://cx-freeze.readthedocs.io/)
- [py2exe Tutorial](https://www.py2exe.org/)
- [Windows Code Signing](https://docs.microsoft.com/en-us/windows/win32/win_cert/code-signing-best-practices)

---

## Version History

- **2025.06** - Updated for v2025.06, corrected icon path to GMRT-logo2020.ico, updated executable naming to include version
- **2025.05** - Initial Windows .exe build instructions

---

**Author:** Paul Johnson  
**Last Updated:** July 2025

