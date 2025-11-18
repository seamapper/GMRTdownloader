# GMRT Bathymetry Grid Downloader

A standalone executable for downloading bathymetry data from the Global Multi-Resolution Topography (GMRT) synthesis.

## Features

- **Interactive GUI**: User-friendly interface for selecting download parameters
- **Map Preview**: Visual preview of the selected area using GMRT ImageServer
- **Multiple Formats**: Support for GeoTIFF, NetCDF, and COARDS formats
- **Resolution Options**: Dropdown menu with preset resolutions: 100m, 200m, 400m, 800m (default: 400m)
- **Tiled Downloads**: Automatic tiling for large downloads
- **Activity Logging**: Detailed log of all operations with timestamps
- **Coordinate Validation**: Automatic validation of geographic coordinates

## Usage

### Running the Executable

1. **Double-click** `GMRT_Bathymetry_Downloader.exe` to launch the application
2. **Set coordinates** for your area of interest:
   - West (minlongitude): -180° to 180°
   - East (maxlongitude): -180° to 180°
   - South (minlatitude): -90° to 90°
   - North (maxlatitude): -90° to 90°
3. **Choose format**: GeoTIFF, NetCDF, or COARDS
4. **Select cell resolution**: Choose from dropdown (100m, 200m, 400m, or 800m per pixel) - default is 400m
5. **Choose layer**: 
   - **Topo-Bathy**: Standard bathymetry and topography data
   - **Topo-Bathy (Observed Only)**: Only direct measurements, no interpolated data
6. **Enable tiling** if downloading large areas
7. **Click "Refresh Map"** to preview the selected area
8. **Click "Download Grid"** to start the download

### Map Preview

- **Refresh Map**: Load a preview of the selected area
- **Show High-Res Mask**: Toggle between regular bathymetry and high-resolution data highlighting
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
- **Current Version**: v2025.06
- **Python**: PyQt6-based GUI
- **Dependencies**: requests, PyQt6, numpy, rasterio, netCDF4

## Version History

- **v2025.06** - Streamlined download process and UI improvements
  - Major change to streamline the download process by removing tiling options
  - Reduced padding in Credit box for more compact UI
  - Improved overall UI spacing and layout
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