# Creating a Mac App from GMRT Downloader

This guide provides step-by-step instructions for converting the GMRT Bathymetry Grid Downloader Python application into a standalone Mac application (.app bundle).

## Table of Contents

1. [Prerequisites](#prerequisites)
2. [Method 1: Using py2app (Recommended for Mac)](#method-1-using-py2app-recommended-for-mac)
3. [Method 2: Using PyInstaller (Cross-platform)](#method-2-using-pyinstaller-cross-platform)
4. [Creating an Icon File](#creating-an-icon-file)
5. [Code Signing and Notarization](#code-signing-and-notarization)
6. [Distribution](#distribution)
7. [Troubleshooting](#troubleshooting)

---

## Prerequisites

### Required Software

1. **macOS** (10.13 High Sierra or later recommended)
2. **Python 3.8+** (Python 3.9 or 3.10 recommended)
3. **Xcode Command Line Tools**:
   ```bash
   xcode-select --install
   ```

### Required Python Packages

Install the required packages in a virtual environment:

```bash
# Create a virtual environment
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install PyQt6
pip install requests
pip install numpy
pip install rasterio  # Optional but recommended
pip install netCDF4   # Optional but recommended
```

### Optional: Install Homebrew

If you don't have Homebrew installed:
```bash
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
```

---

## Method 1: Using py2app (Recommended for Mac)

`py2app` is specifically designed for creating Mac applications and produces cleaner, more native-looking app bundles.

### Step 1: Install py2app

```bash
pip install py2app
```

### Step 2: Create setup.py

Create a file named `setup.py` in your project directory with the following content:

```python
"""
Setup script for creating a Mac app bundle using py2app
"""

from setuptools import setup

APP = ['GMRT_Downloader.py']
DATA_FILES = [
    ('media', ['media/GMRT-logo2020.ico', 'media/GMRT-logo2020.png']),
    ('', ['gmrtgrab_config.json']),
]
OPTIONS = {
    'argv_emulation': True,
    'iconfile': 'media/GMRT-logo2020.icns',  # You'll need to convert .ico or .png to .icns (see Icon section)
    'plist': {
        'CFBundleName': 'GMRT Downloader',
        'CFBundleDisplayName': 'GMRT Bathymetry Grid Downloader',
        'CFBundleGetInfoString': 'GMRT Bathymetry Grid Downloader v2025.06',
        'CFBundleIdentifier': 'org.gmrt.downloader',
        'CFBundleVersion': '2025.06',
        'CFBundleShortVersionString': '2025.06',
        'NSHighResolutionCapable': True,
        'NSRequiresAquaSystemAppearance': False,
        'LSMinimumSystemVersion': '10.13',
        'NSHumanReadableCopyright': 'Copyright © 2025 Paul Johnson',
        'NSAppTransportSecurity': {
            'NSAllowsArbitraryLoads': True  # Required for GMRT API calls
        }
    },
    'includes': [
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
    ],
    'packages': [
        'rasterio',
        'netCDF4',
    ],
    'excludes': [
        'matplotlib',
        'scipy',
        'pandas',
    ],
}

setup(
    name='GMRT Downloader',
    app=APP,
    data_files=DATA_FILES,
    options={'py2app': OPTIONS},
    setup_requires=['py2app'],
)
```

### Step 3: Build the App

#### First Build (Creates .app bundle)

```bash
python setup.py py2app
```

This will create:
- `dist/GMRT_Downloader.app` - The Mac application bundle

#### Development Build (Aliased, faster rebuilds)

For faster rebuilds during development:

```bash
python setup.py py2app -A
```

**Note:** The aliased version is faster but creates symbolic links to your Python installation. Use the full build for distribution.

### Step 4: Test the App

```bash
# Test from command line
open dist/GMRT_Downloader.app

# Or run directly
./dist/GMRT_Downloader.app/Contents/MacOS/GMRT_Downloader
```

### Step 5: Clean Build (Optional)

To clean previous builds:

```bash
python setup.py clean
rm -rf build dist
```

---

## Method 2: Using PyInstaller (Cross-platform)

PyInstaller works on Mac, Windows, and Linux, but may produce larger app bundles.

### Step 1: Install PyInstaller

```bash
pip install pyinstaller
```

### Step 2: Create .spec File for Mac

Create a file named `GMRT_Downloader_Mac.spec`:

```python
# -*- mode: python ; coding: utf-8 -*-
from PyInstaller.utils.hooks import collect_all

datas = [
    ('media', 'media'),
    ('gmrtgrab_config.json', '.'),
]
binaries = []
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

# Collect rasterio data files
try:
    tmp_ret = collect_all('rasterio')
    datas += tmp_ret[0]
    binaries += tmp_ret[1]
    hiddenimports += tmp_ret[2]
except:
    pass

a = Analysis(
    ['GMRT_Downloader.py'],
    pathex=[],
    binaries=binaries,
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    noarchive=False,
    optimize=0,
)

pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.datas,
    [],
    name='GMRT_Downloader',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,  # UPX can cause issues on macOS
    upx_exclude=[],
    runtime_tmpdir=None,
    console=False,  # Set to True for debugging
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)

app = BUNDLE(
    exe,
    name='GMRT_Downloader.app',
    icon='media/GMRT-logo2020.icns',  # Convert .ico or .png to .icns first (see Icon section)
    bundle_identifier='org.gmrt.downloader',
    info_plist={
        'NSHighResolutionCapable': True,
        'LSMinimumSystemVersion': '10.13',
        'CFBundleName': 'GMRT Downloader',
        'CFBundleDisplayName': 'GMRT Bathymetry Grid Downloader',
        'CFBundleVersion': '2025.06',
        'CFBundleShortVersionString': '2025.06',
        'NSHumanReadableCopyright': 'Copyright © 2025 Paul Johnson',
        'NSAppTransportSecurity': {
            'NSAllowsArbitraryLoads': True
        }
    },
)
```

### Step 3: Build with PyInstaller

```bash
pyinstaller GMRT_Downloader_Mac.spec
```

This creates `dist/GMRT_Downloader.app`.

### Step 4: Test the App

```bash
open dist/GMRT_Downloader.app
```

---

## Creating an Icon File

Mac apps require `.icns` format icons. You need to convert your `.ico` file.

### Option 1: Using iconutil (macOS built-in)

1. **Create an iconset directory structure:**

```bash
mkdir -p GMRT_Downloader.iconset
```

2. **Convert .ico to individual PNG files** (you can use online tools or ImageMagick):

```bash
# Install ImageMagick if needed
brew install imagemagick

# Convert .ico or .png to PNG at various sizes
# Using .ico file:
convert media/GMRT-logo2020.ico -resize 16x16 GMRT_Downloader.iconset/icon_16x16.png
convert media/GMRT-logo2020.ico -resize 32x32 GMRT_Downloader.iconset/icon_16x16@2x.png
convert media/GMRT-logo2020.ico -resize 32x32 GMRT_Downloader.iconset/icon_32x32.png
convert media/GMRT-logo2020.ico -resize 64x64 GMRT_Downloader.iconset/icon_32x32@2x.png
convert media/GMRT-logo2020.ico -resize 128x128 GMRT_Downloader.iconset/icon_128x128.png
convert media/GMRT-logo2020.ico -resize 256x256 GMRT_Downloader.iconset/icon_128x128@2x.png
convert media/GMRT-logo2020.ico -resize 256x256 GMRT_Downloader.iconset/icon_256x256.png
convert media/GMRT-logo2020.ico -resize 512x512 GMRT_Downloader.iconset/icon_256x256@2x.png
convert media/GMRT-logo2020.ico -resize 512x512 GMRT_Downloader.iconset/icon_512x512.png
convert media/GMRT-logo2020.ico -resize 1024x1024 GMRT_Downloader.iconset/icon_512x512@2x.png
```

3. **Create .icns file:**

```bash
iconutil -c icns GMRT_Downloader.iconset -o media/GMRT-logo2020.icns
```

### Option 2: Using Online Tools

1. Use an online converter like:
   - https://cloudconvert.com/ico-to-icns
   - https://convertio.co/ico-icns/
2. Upload your `GMRT-logo2020.ico` or `GMRT-logo2020.png` file
3. Download the converted `GMRT-logo2020.icns` file
4. Place it in the `media/` directory

### Option 3: Using sips (macOS built-in, simpler)

If you have a high-resolution PNG (512x512 or larger):

```bash
# Create iconset
mkdir -p GMRT_Downloader.iconset

# Use sips to create all sizes
sips -z 16 16 media/GMRT-logo2020.png --out GMRT_Downloader.iconset/icon_16x16.png
sips -z 32 32 media/GMRT-logo2020.png --out GMRT_Downloader.iconset/icon_16x16@2x.png
sips -z 32 32 media/GMRT-logo2020.png --out GMRT_Downloader.iconset/icon_32x32.png
sips -z 64 64 media/GMRT-logo2020.png --out GMRT_Downloader.iconset/icon_32x32@2x.png
sips -z 128 128 media/GMRT-logo2020.png --out GMRT_Downloader.iconset/icon_128x128.png
sips -z 256 256 media/GMRT-logo2020.png --out GMRT_Downloader.iconset/icon_128x128@2x.png
sips -z 256 256 media/GMRT-logo2020.png --out GMRT_Downloader.iconset/icon_256x256.png
sips -z 512 512 media/GMRT-logo2020.png --out GMRT_Downloader.iconset/icon_256x256@2x.png
sips -z 512 512 media/GMRT-logo2020.png --out GMRT_Downloader.iconset/icon_512x512.png
sips -z 1024 1024 media/GMRT-logo2020.png --out GMRT_Downloader.iconset/icon_512x512@2x.png

# Convert to .icns
iconutil -c icns GMRT_Downloader.iconset -o media/GMRT-logo2020.icns
```

---

## Code Signing and Notarization

For distribution outside the Mac App Store, you should code sign and optionally notarize your app.

### Prerequisites

1. **Apple Developer Account** (free or paid)
2. **Developer ID Application Certificate** from Apple Developer Portal

### Code Signing

1. **Get your certificate identity:**

```bash
# List available identities
security find-identity -v -p codesigning
```

2. **Sign the app:**

```bash
# Sign with your Developer ID
codesign --deep --force --verify --verbose --sign "Developer ID Application: Your Name (TEAM_ID)" dist/GMRT_Downloader.app

# Verify the signature
codesign --verify --verbose dist/GMRT_Downloader.app
spctl --assess --verbose dist/GMRT_Downloader.app
```

### Notarization (Optional but Recommended)

Notarization is required for macOS 10.15+ distribution.

1. **Create an app-specific password** in Apple ID account settings

2. **Submit for notarization:**

```bash
# Create a zip file
ditto -c -k --keepParent dist/GMRT_Downloader.app GMRT_Downloader.zip

# Submit for notarization
xcrun notarytool submit GMRT_Downloader.zip \
    --apple-id "your@email.com" \
    --team-id "TEAM_ID" \
    --password "app-specific-password" \
    --wait
```

3. **Staple the notarization ticket:**

```bash
xcrun stapler staple dist/GMRT_Downloader.app
xcrun stapler validate dist/GMRT_Downloader.app
```

---

## Distribution

### Creating a DMG (Disk Image)

A DMG file is the standard way to distribute Mac applications.

1. **Create a DMG:**

```bash
# Create a temporary directory structure
mkdir -p dmg_build
cp -R dist/GMRT_Downloader.app dmg_build/

# Create a symbolic link to Applications
ln -s /Applications dmg_build/Applications

# Create the DMG
hdiutil create -volname "GMRT Downloader" -srcfolder dmg_build -ov -format UDZO GMRT_Downloader.dmg

# Clean up
rm -rf dmg_build
```

2. **Create a nicer DMG with background (optional):**

Use a tool like [create-dmg](https://github.com/create-dmg/create-dmg):

```bash
brew install create-dmg

create-dmg \
  --volname "GMRT Downloader" \
  --window-pos 200 120 \
  --window-size 800 400 \
  --icon-size 100 \
  --icon "GMRT_Downloader.app" 200 190 \
  --hide-extension "GMRT_Downloader.app" \
  --app-drop-link 600 185 \
  "GMRT_Downloader.dmg" \
  "dist/"
```

### Creating a ZIP Archive

For simpler distribution:

```bash
cd dist
zip -r ../GMRT_Downloader_Mac.zip GMRT_Downloader.app
cd ..
```

---

## Troubleshooting

### Common Issues

#### 1. "App is damaged" Error

**Solution:** If you see this error when opening the app:

```bash
# Remove quarantine attribute
xattr -cr dist/GMRT_Downloader.app
```

Or if distributing via download, make sure the app is properly code signed.

#### 2. Missing Dependencies

**Problem:** App crashes or can't find modules.

**Solution:** 
- Check that all dependencies are included in `hiddenimports` or `includes`
- Use `--debug=all` with PyInstaller to see what's missing
- Test imports in a clean Python environment

#### 3. PyQt6 Issues

**Problem:** PyQt6 plugins not loading.

**Solution:** Add explicit plugin paths in your code:

```python
import os
from PyQt6.QtCore import QCoreApplication

if getattr(sys, 'frozen', False):
    # Running as compiled app
    QCoreApplication.setLibraryPaths([
        os.path.join(sys._MEIPASS, 'PyQt6', 'Qt6', 'plugins')
    ])
```

#### 4. Large App Size

**Problem:** App bundle is very large (>200MB).

**Solutions:**
- Use `--exclude-module` to remove unused packages
- Strip unnecessary binaries
- Use UPX compression (PyInstaller) - but test thoroughly

#### 5. Network Requests Failing

**Problem:** App can't connect to GMRT servers.

**Solution:** Ensure `NSAppTransportSecurity` is set in Info.plist (see setup.py example above).

#### 6. Console Window Appears

**Problem:** Console window shows when running the app.

**Solution:** 
- Set `console=False` in PyInstaller spec file
- Set `argv_emulation=False` in py2app setup.py

### Debugging Tips

1. **Run with console enabled:**

```python
# In setup.py or spec file, temporarily set:
console=True  # PyInstaller
# or
'argv_emulation': False  # py2app
```

2. **Check app bundle contents:**

```bash
# List Contents
ls -la dist/GMRT_Downloader.app/Contents/

# Check executable
file dist/GMRT_Downloader.app/Contents/MacOS/GMRT_Downloader

# Check Info.plist
plutil -p dist/GMRT_Downloader.app/Contents/Info.plist
```

3. **Run from terminal:**

```bash
./dist/GMRT_Downloader.app/Contents/MacOS/GMRT_Downloader
```

This will show any error messages in the terminal.

---

## Quick Start Checklist

- [ ] Install Python 3.8+ and required packages
- [ ] Install py2app or PyInstaller
- [ ] Convert icon from .ico/.png to .icns (GMRT-logo2020.icns)
- [ ] Create setup.py or .spec file
- [ ] Build the app
- [ ] Test the app locally
- [ ] Code sign the app (for distribution)
- [ ] Create DMG or ZIP for distribution

---

## Additional Resources

- [py2app Documentation](https://py2app.readthedocs.io/)
- [PyInstaller Documentation](https://pyinstaller.readthedocs.io/)
- [Apple Code Signing Guide](https://developer.apple.com/library/archive/documentation/Security/Conceptual/CodeSigningGuide/)
- [macOS Notarization](https://developer.apple.com/documentation/security/notarizing_macos_software_before_distribution)

---

## Version History

- **2025.06** - Updated for v2025.06, corrected icon paths to GMRT-logo2020.ico/png/icns
- **2025.05** - Initial Mac app build instructions

---

**Author:** Paul Johnson  
**Last Updated:** July 2025

