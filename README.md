# GMRT Bathymetry Grid Downloader

A PyQt6-based GUI application for downloading bathymetry data from the Global Multi-Resolution Topography (GMRT) synthesis.

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/)
[![PyQt6](https://img.shields.io/badge/PyQt6-6.0+-green.svg)](https://www.riverbankcomputing.com/software/pyqt/)
[![License](https://img.shields.io/badge/License-CC%20BY%204.0-lightgrey.svg)](https://creativecommons.org/licenses/by/4.0/)

## Overview

The GMRT Bathymetry Grid Downloader provides an intuitive graphical interface for downloading high-resolution bathymetry data from the GMRT synthesis. It supports multiple output formats, resolution options, and includes features like map preview and automatic tiling for large datasets.

## Features

- 🗺️ **Interactive Map Preview** - Visual preview of selected areas using GMRT ImageServer
- 📥 **Multiple Output Formats** - GeoTIFF, NetCDF, COARDS, and ESRI ASCII
- 🎯 **Flexible Resolution Options** - High, low, medium, max resolution or custom meter resolution
- 🧩 **Automatic Tiling** - Handles large datasets by automatically breaking them into manageable tiles
- 📊 **Real-time Activity Logging** - Detailed log of all operations with timestamps
- ✅ **Coordinate Validation** - Automatic validation of geographic coordinates
- 🎨 **User-Friendly GUI** - Clean, intuitive interface built with PyQt6

## Screenshots

*Screenshots coming soon*

## Installation

### Option 1: Pre-built Executables

Download the latest release for your platform:
- **Windows**: `GMRT_Bathymetry_Downloader.exe` (from [Releases](https://github.com/seamapper/GMRTdownloader/releases))
- **macOS**: `GMRT_Downloader.app` (from [Releases](https://github.com/seamapper/GMRTdownloader/releases))

### Option 2: Run from Source

1. **Clone the repository:**
   ```bash
   git clone https://github.com/seamapper/GMRTdownloader.git
   cd GMRTdownloader
   ```

2. **Create a virtual environment (recommended):**
   ```bash
   python -m venv venv
   
   # On Windows:
   venv\Scripts\activate
   
   # On macOS/Linux:
   source venv/bin/activate
   ```

3. **Install dependencies:**
   ```bash
   pip install PyQt6 requests numpy
   
   # Optional but recommended:
   pip install rasterio netCDF4
   ```

4. **Run the application:**
   ```bash
   python GMRT_Downloader.py
   ```

## Usage

### Quick Start

1. **Launch the application** (double-click executable or run from source)
2. **Set geographic boundaries:**
   - West (minlongitude): -180° to 180°
   - East (maxlongitude): -180° to 180°
   - South (minlatitude): -90° to 90°
   - North (maxlatitude): -90° to 90°
3. **Choose output format:** GeoTIFF, NetCDF, COARDS, or ESRI ASCII
4. **Select resolution:** High, low, medium, max, or enter custom meter resolution
5. **Choose layer:** Topo or topo-mask
6. **Click "Refresh Map"** to preview the selected area
7. **Click "Download Grid"** to start the download

### Map Preview

- Use **"Refresh Map"** to load a preview of your selected area
- Toggle **"Show High-Res Mask"** to highlight high-resolution data coverage
- The map shows actual data availability for your coordinates

### Download Options

- **Single File**: Download the entire area as one file
- **Tiled Dataset**: Automatically breaks large areas (>3°) into 3°×3° tiles with overlap
- **Progress Tracking**: Real-time status updates and completion notifications

### Output Formats

- **GeoTIFF (.tif)**: Raster format with embedded georeference information (recommended for GIS)
- **NetCDF (.nc)**: Scientific data format with comprehensive metadata
- **COARDS (.grd)**: ASCII grid format
- **ESRI ASCII (.asc)**: Simple ASCII grid format

## Building from Source

### Windows Executable

See [WINDOWS_EXE_BUILD_INSTRUCTIONS.md](WINDOWS_EXE_BUILD_INSTRUCTIONS.md) for detailed instructions.

**Quick build:**
```bash
pip install pyinstaller
pyinstaller GMRT_Downloader.spec
# Or use the automated build script:
build.bat
```

### macOS Application

See [MAC_APP_BUILD_INSTRUCTIONS.md](MAC_APP_BUILD_INSTRUCTIONS.md) for detailed instructions.

**Quick build with py2app:**
```bash
pip install py2app
python setup.py py2app
```

## System Requirements

### Minimum Requirements

- **Operating System**: Windows 10/11 (64-bit) or macOS 10.13+ or Linux
- **Python**: 3.8 or higher (if running from source)
- **Internet Connection**: Required for downloading data
- **Disk Space**: Varies based on download area (can be several GB for large areas)

### Recommended

- **RAM**: 4GB or more
- **Storage**: SSD with sufficient free space
- **Network**: Stable broadband connection for large downloads

## Dependencies

### Required
- `PyQt6` >= 6.0 - GUI framework
- `requests` - HTTP library for API calls
- `numpy` - Numerical operations

### Optional (Recommended)
- `rasterio` - GeoTIFF support and mosaicking
- `netCDF4` - NetCDF format support

## Troubleshooting

### Common Issues

1. **"No Data Returned"**
   - The selected area may be outside available data coverage
   - Try adjusting coordinates or checking the map preview

2. **"Request Too Large"**
   - Enable tiled downloads for large areas
   - Reduce the download area size

3. **"Connection Error"**
   - Check your internet connection
   - Verify GMRT servers are accessible: https://www.gmrt.org/

4. **"Timeout"**
   - The GMRT server may be busy
   - Try again later or reduce download area

5. **Map Preview Not Loading**
   - Check internet connection
   - Verify coordinates are valid
   - Try refreshing the map

### Getting Help

- Check the [GMRT_DOWNLOADER_README.md](GMRT_DOWNLOADER_README.md) for additional documentation
- Review build instructions for platform-specific issues:
  - [Windows Build Instructions](WINDOWS_EXE_BUILD_INSTRUCTIONS.md)
  - [macOS Build Instructions](MAC_APP_BUILD_INSTRUCTIONS.md)
- For GMRT data issues, visit: https://www.gmrt.org/

## Data Source & Attribution

This application downloads data from the **Global Multi-Resolution Topography (GMRT)** synthesis, hosted by the Lamont-Doherty Earth Observatory of Columbia University.

### Citation

When using GMRT data, please cite:

> Ryan, W.B.F., S.M. Carbotte, J.O. Coplan, S. O'Hara, A. Melkonian, R. Arko, R.A. Weissel, V. Ferrini, A. Goodwillie, F. Nitsche, J. Bonczkowski, and R. Zemsky (2009), Global Multi-Resolution Topography synthesis, Geochem. Geophys. Geosyst., 10, Q03014, doi: 10.1029/2008GC002332

### Data License

- **License**: Creative Commons Attribution 4.0 International
- **GMRT Website**: https://www.gmrt.org/
- **Funding**: US National Science Foundation (NSF)

## Project Structure

```
GMRTdownloader/
│
├── GMRT_Downloader.py          # Main application script
├── gmrtgrab_config.json         # Configuration file
├── GMRT_Downloader.spec         # PyInstaller spec file (Windows)
├── build.bat                    # Windows build script
│
├── media/
│   └── mgds.ico                 # Application icon
│
├── README.md                    # This file
├── GMRT_DOWNLOADER_README.md    # Detailed usage documentation
├── WINDOWS_EXE_BUILD_INSTRUCTIONS.md  # Windows build guide
└── MAC_APP_BUILD_INSTRUCTIONS.md      # macOS build guide
```

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request. For major changes, please open an issue first to discuss what you would like to change.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## Version History

- **v2025.05** - UI cleanup and improvements
- **v2025.04** - Fixed tile size parameter functionality
- **v2025.03** - Fixed tiling section to avoid gaps
- **v2025.02** - Fixed zoom issues

## License

This software is provided as-is. The GMRT data downloaded through this application is licensed under Creative Commons Attribution 4.0 International.

## Author

**Paul Johnson**  
Email: pjohnson@ccom.unh.edu

## Acknowledgments

- **GMRT Team** at Lamont-Doherty Earth Observatory for providing the bathymetry data
- **PyQt6** developers for the excellent GUI framework
- All contributors and users of this project

## Links

- **GMRT Website**: https://www.gmrt.org/
- **GMRT Documentation**: https://www.gmrt.org/services/
- **GMRT Contact**: https://www.gmrt.org/about/contact.php
- **PyQt6 Documentation**: https://www.riverbankcomputing.com/static/Docs/PyQt6/

---

**Note**: This application is a community tool for accessing GMRT data. For official GMRT support, please contact the GMRT team directly.

