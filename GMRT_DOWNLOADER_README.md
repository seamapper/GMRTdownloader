# GMRT Bathymetry Grid Downloader

A standalone executable for downloading bathymetry data from the Global Multi-Resolution Topography (GMRT) synthesis.

## Features

- **Interactive GUI**: User-friendly interface for selecting download parameters
- **Map Preview**: Visual preview of the selected area using GMRT ImageServer
- **Direct Map Selection**: Move the mouse over the map (crosshair cursor) and left-drag to draw a red rectangle that sets the Area of Interest; the North/West/East/South fields and estimated pixel count update automatically
- **Multiple Formats**: Support for GeoTIFF, NetCDF, and COARDS formats
- **Resolution Options**: Dropdown menu with preset resolutions: 100m, 200m, 400m, 800m (default: 400m)
- **Tiled Downloads**: Automatic tiling for large downloads
- **AOI Zoom History**: *Zoom to Previous* and *Zoom to Next* buttons let you navigate up to 10 previous/next Areas of Interest, plus *Zoom to Defaults* to return to the global view
- **Estimated Pixel Count**: Live estimate of grid width × height (and total pixels) based on current bounds and cell resolution
- **Always-on Dark Mode**: Application uses a consistent dark theme regardless of the Windows light/dark setting
- **Activity Logging**: Detailed log of all operations with timestamps
- **Coordinate Validation**: Automatic validation of geographic coordinates

## Usage

### Running the Executable

1. **Double-click** `GMRT_Bathymetry_Downloader.exe` to launch the application
2. **Set coordinates** for your area of interest:
   - Either type values directly into **West**, **East**, **South**, **North**
   - Or left-drag on the map (crosshair cursor) to draw a red rectangle; the fields will update automatically
3. **Check estimated pixel count** under the Area of Interest section to understand approximate grid size before downloading
4. **Choose format**: GeoTIFF, NetCDF, or COARDS
5. **Select cell resolution**: Choose from dropdown (100m, 200m, 400m, or 800m per pixel) - default is 400m
6. **Choose layer**: 
   - **Topo-Bathy**: Standard bathymetry and topography data
   - **Topo-Bathy (Observed Only)**: Only direct measurements, no interpolated data
6. **Enable tiling** if downloading large areas
7. **Click "Refresh Map"** to preview the selected area
8. **(Optional) Use Zoom History**:
   - **Zoom to Previous**: Go back to a previous Area of Interest
   - **Zoom to Next**: Go forward to the next Area of Interest (after using Zoom to Previous)
   - **Zoom to Defaults**: Return to the starting global view
9. **Click "Download Grid"** to start the download

### Map Preview

- **Refresh Map**: Load a preview of the selected area
- **Show High-Res Mask**: Toggle between regular bathymetry and high-resolution data highlighting
- **Draw Bounds by Dragging**: Left-drag on the map (crosshair cursor) to draw a red rectangle that defines the Area of Interest
- **Zoom History Controls**:
  - **Zoom to Previous** / **Zoom to Next**: Step backward/forward through up to 10 Areas of Interest
  - **Zoom to Defaults**: Return to the starting global view
- The map shows the actual data coverage for your selected coordinates

### Download Options

- **Single File**: Download the entire area as one file
- **Tiled Dataset**: Automatically breaks large areas into manageable tiles
- **Intelligent Mosaicing**: Automatically combines downloaded tiles into a final mosaic
- **Progress Tracking**: Real-time status updates and completion notifications

## File Formats

- **GeoTIFF (.tif)**: Raster format with embedded georeference information (recommended for GIS)
- **NetCDF (.nc)**: Scientific data format with comprehensive metadata (full mosaicing support)
- **COARDS (.grd)**: ASCII grid format

All formats support automatic tiling and mosaicing when enabled.

## System Requirements

- **Windows 10/11** (64-bit)
- **Internet connection** for downloading data
- **Sufficient disk space** for downloaded files (can be several GB for large areas)

## Troubleshooting

### Common Issues

1. **"No Data Returned"**: The selected area may be outside available data coverage
2. **"Request Too Large"**: Try enabling tiled downloads for large areas
3. **"Connection Error"**: Check your internet connection
4. **"Timeout"**: The server may be busy, try again later

### Map Preview Issues

- If the map doesn't load, check your internet connection
- Try refreshing the map or adjusting coordinates
- The map preview uses a lower resolution than the actual download

## Data Source

This application downloads data from the **Global Multi-Resolution Topography (GMRT)** synthesis, hosted by the Lamont-Doherty Earth Observatory of Columbia University.

- **Website**: https://www.gmrt.org/
- **Data License**: Creative Commons Attribution 4.0 International
- **Funding**: US National Science Foundation (NSF)

## Support

For issues with the application or data downloads, please refer to:
- GMRT documentation: https://www.gmrt.org/
- GMRT contact: https://www.gmrt.org/about/contact.php

## Version Information

- **Application**: GMRT Bathymetry Grid Downloader
- **Current Version**: v2026.01
- **Python**: PyQt6-based GUI
- **Dependencies**: requests, PyQt6, numpy, rasterio, netCDF4

## Version History

- **v2026.01** - Refactored codebase to use modular structure (main.py, config.py, ui/, workers/, converters.py); launcher remains GMRT_Downloader.py for builds
- **v2025.06** - UI refactor, dark mode, and AOI improvements (standalone executable aligned with source README)
  - Streamlined download process by simplifying tiling behavior
  - Forced dark theme so the executable always appears in dark mode on Windows
  - Reworked Area of Interest controls into a North/West/East/South “+” layout with live estimated pixel count
  - Enabled direct map-based AOI selection via left-drag (red rectangle) with crosshair cursor
  - Added AOI zoom history controls: Zoom to Previous / Zoom to Next (up to 10 levels) and Zoom to Defaults
  - Reduced padding in Credit box for more compact UI and improved overall spacing and layout
- **v2025.05** - Major UI and functionality improvements
  - Changed Cell Resolution to dropdown menu (100m, 200m, 400m, 800m)
  - Removed Data Resolution option (always use Cell Resolution)
  - Relabeled layer types for clarity (Topo-Bathy, Topo-Bathy Observed Only)
  - Mosaicing now prefers shallower values in overlapping areas
  - Fixed directory selection to remember last directory within session
  - Added comprehensive NetCDF file support for mosaicing
  - Implemented automatic tile overlap calculation (2 grid cells based on resolution)
  - Removed manual overlap control (now automatic)
  - Updated application icon to GMRT-logo2020.ico 