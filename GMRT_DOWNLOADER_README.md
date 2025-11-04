# GMRT Bathymetry Grid Downloader

A standalone executable for downloading bathymetry data from the Global Multi-Resolution Topography (GMRT) synthesis.

## Features

- **Interactive GUI**: User-friendly interface for selecting download parameters
- **Map Preview**: Visual preview of the selected area using GMRT ImageServer
- **Multiple Formats**: Support for GeoTIFF, NetCDF, COARDS, and ESRI ASCII formats
- **Resolution Options**: High, low, medium, max resolution or custom meter resolution
- **Tiled Downloads**: Automatic tiling for large datasets (>3 degrees)
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
3. **Choose format**: GeoTIFF, NetCDF, COARDS, or ESRI ASCII
4. **Select resolution**: High, low, medium, max, or custom meter resolution
5. **Choose layer**: Topo or topo-mask
6. **Enable tiling** if downloading large areas
7. **Click "Refresh Map"** to preview the selected area
8. **Click "Download Grid"** to start the download

### Map Preview

- **Refresh Map**: Load a preview of the selected area
- **Show High-Res Mask**: Toggle between regular bathymetry and high-resolution data highlighting
- The map shows the actual data coverage for your selected coordinates

### Download Options

- **Single File**: Download the entire area as one file
- **Tiled Dataset**: Automatically break large areas into 3°x3° tiles with overlap
- **Progress Tracking**: Real-time status updates and completion notifications

## File Formats

- **GeoTIFF (.tif)**: Raster format with embedded georeference information
- **NetCDF (.nc)**: Scientific data format with metadata
- **COARDS (.grd)**: ASCII grid format
- **ESRI ASCII (.asc)**: Simple ASCII grid format

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
- **Build Date**: July 2025
- **Python**: PyQt6-based GUI
- **Dependencies**: requests, PyQt6 