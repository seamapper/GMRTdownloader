"""
GMRT Bathymetry Grid Downloader

A PyQt6-based GUI application for downloading bathymetry data from the 
Global Multi-Resolution Topography (GMRT) synthesis.

Credit: Ryan, W.B.F., S.M. Carbotte, J.O. Coplan, S. O'Hara, A. Melkonian, R. Arko, R.A. Weissel, V. Ferrini, A. Goodwillie, F. Nitsche, J. Bonczkowski, and R. Zemsky (2009), Global Multi-Resolution Topography synthesis, Geochem. Geophys. Geosyst., 10, Q03014, doi: 10.1029/2008GC002332

Features:
- Interactive map preview using GMRT ImageServer
- Multiple output formats (GeoTIFF, NetCDF, COARDS, ESRI ASCII)
- Resolution options (high, low, med, max) and custom meter resolution
- Tiled downloads for large datasets
- Real-time activity logging
- Coordinate validation

Author: Paul Johnson
Date: July 2025 
"""

import sys
import os
import time
import json
from datetime import datetime
from PyQt6.QtWidgets import (
    QApplication, QWidget, QLabel, QLineEdit, QComboBox, QPushButton, 
    QVBoxLayout, QHBoxLayout, QFileDialog, QMessageBox, QDoubleSpinBox, 
    QGroupBox, QFormLayout, QCheckBox, QTextEdit, QSplitter, QFrame
)
from PyQt6.QtCore import Qt, QTimer, QThread, pyqtSignal, QUrl, QRect, QPoint
from PyQt6.QtGui import QPixmap, QImage, QPainter, QPen, QColor, QIcon
import requests
import numpy as np
import shutil
try:
    import rasterio
except ImportError:
    rasterio = None
try:
    import netCDF4
except ImportError:
    netCDF4 = None
import math

#__version__ = "2025.02"  # Updated to fix zoom issues   
# __version__ = "2025.03"  # Fixed tiling section to avoid gaps   
# __version__ = "2025.04"  # Made the tile size parameter work    
__version__ = "2025.05"  # Cleaned up the UI   

# GMRT API endpoints
GMRT_URL = "https://www.gmrt.org/services/GridServer"  # For downloading bathymetry data
GMRT_IMAGE_URL = "https://www.gmrt.org/services/ImageServer"  # For map preview images

class MapWorker(QThread):
    """
    Worker thread for loading map preview images from GMRT ImageServer.
    
    This class runs in a separate thread to prevent the GUI from freezing
    while downloading map images. It communicates with the main thread
    using PyQt signals.
    """
    # Signals to communicate with the main thread
    map_loaded = pyqtSignal(QPixmap)  # Emitted when map loads successfully
    map_error = pyqtSignal(str)       # Emitted when map loading fails
    
    def __init__(self, west, east, south, north, width=800, mask=True):
        """
        Initialize the map worker with geographic bounds and display options.
        
        Args:
            west (float): Western longitude boundary
            east (float): Eastern longitude boundary  
            south (float): Southern latitude boundary
            north (float): Northern latitude boundary
            width (int): Image width in pixels (max 8000)
            mask (bool): Whether to show high-resolution data mask
        """
        super().__init__()
        self.west = west
        self.east = east
        self.south = south
        self.north = north
        self.width = width
        self.mask = mask
    
    def run(self):
        """
        Main worker method that downloads the map image from GMRT ImageServer.
        
        This method runs in a separate thread and should not directly
        interact with GUI elements. Instead, it emits signals to
        communicate results back to the main thread.
        """
        print("[DEBUG] MapWorker.run: started")
        try:
            # Prepare parameters for GMRT ImageServer API
            params = {
                "minlongitude": self.west,
                "maxlongitude": self.east,
                "minlatitude": self.south,
                "maxlatitude": self.north,
                "width": self.width,
                "mask": "1" if self.mask else "0"  # Convert boolean to string
            }
            
            # Log the image download request
            param_str = ", ".join([f"{k}={v}" for k, v in params.items()])
            print(f"[MapWorker] Downloading map image: {param_str}")
            print(f"[DEBUG] MapWorker.run: sending request to {GMRT_IMAGE_URL}")
            
            # Download the map image from GMRT ImageServer
            response = requests.get(GMRT_IMAGE_URL, params=params, timeout=30)
            print(f"[DEBUG] MapWorker.run: response status {response.status_code}")
            
            if response.status_code == 200:
                # Successfully downloaded image data
                # Convert the binary response content to a QPixmap
                image = QImage()
                image.loadFromData(response.content)
                pixmap = QPixmap.fromImage(image)
                
                print(f"[MapWorker] Map image downloaded successfully ({len(response.content)} bytes)")
                
                # Emit signal with the loaded pixmap
                self.map_loaded.emit(pixmap)
            else:
                # HTTP error occurred
                print(f"[MapWorker] Failed to download map image: HTTP {response.status_code}")
                self.map_error.emit(f"Failed to load map: HTTP {response.status_code}")
                
        except Exception as e:
            # Network or other error occurred
            print(f"[MapWorker] Map image download error: {str(e)}")
            self.map_error.emit(f"Map loading error: {str(e)}")

class MosaicWorker(QThread):
    """Worker thread for handling all mosaicking operations"""
    progress = pyqtSignal(str)  # progress message
    finished = pyqtSignal(bool, str)  # success, result
    
    def __init__(self, downloaded_tile_files, download_dir, layer_type,
                 west_spin, south_spin,
                 east_spin, north_spin, delete_tiles_checkbox, split_checkbox, format_type=None):
        super().__init__()
        self.downloaded_tile_files = downloaded_tile_files
        self.download_dir = download_dir
        self.layer_type = layer_type
        self.west_spin = west_spin
        self.south_spin = south_spin
        self.east_spin = east_spin
        self.north_spin = north_spin
        self.delete_tiles_checkbox = delete_tiles_checkbox
        self.split_checkbox = split_checkbox
        self.format_type = format_type
        self.mosaic_path = None
    
    def _open_raster_file(self, tile_file):
        """
        Open a raster file (GeoTIFF, NetCDF, etc.) using rasterio.
        Handles different file formats appropriately.
        
        Args:
            tile_file (str): Path to the raster file
            
        Returns:
            rasterio.DatasetReader: Opened dataset or None if failed
        """
        try:
            # Try opening with rasterio - it should auto-detect format
            dataset = rasterio.open(tile_file)
            return dataset
        except Exception as e1:
            print(f"[DEBUG] Failed to open {tile_file} with default driver: {e1}")
            # If it's a NetCDF file, try explicitly with NetCDF driver
            if tile_file.lower().endswith('.nc'):
                try:
                    print(f"[DEBUG] Trying NetCDF driver for {os.path.basename(tile_file)}")
                    dataset = rasterio.open(tile_file, driver='NetCDF')
                    return dataset
                except Exception as e2:
                    print(f"[DEBUG] Failed to open NetCDF file with NetCDF driver: {e2}")
                    # Try alternative NetCDF access patterns using GDAL virtual dataset syntax
                    # GMRT NetCDF files might need subdataset specification
                    gdal_patterns = [
                        f'NETCDF:"{tile_file}":z',           # Try 'z' variable (common for elevation)
                        f'NETCDF:"{tile_file}":elevation',   # Try 'elevation' variable
                        f'NETCDF:"{tile_file}":topo',        # Try 'topo' variable
                        f'NETCDF:"{tile_file}":bathy',       # Try 'bathy' variable
                        f'NETCDF:"{tile_file}"',             # Try without subdataset
                    ]
                    
                    for gdal_path in gdal_patterns:
                        try:
                            print(f"[DEBUG] Trying GDAL path: {gdal_path}")
                            dataset = rasterio.open(gdal_path)
                            print(f"[DEBUG] Successfully opened with pattern: {gdal_path}")
                            return dataset
                        except Exception as e3:
                            print(f"[DEBUG] Failed with pattern {gdal_path}: {e3}")
                            continue
                    
                    # If all GDAL patterns fail, try using netCDF4 to read and create a temporary GeoTIFF
                    if netCDF4 is not None:
                        try:
                            print(f"[DEBUG] Attempting NetCDF4-based conversion for {os.path.basename(tile_file)}")
                            return self._open_netcdf_with_netcdf4(tile_file)
                        except Exception as e4:
                            print(f"[DEBUG] NetCDF4 conversion also failed: {e4}")
                    
                    print(f"[DEBUG] All NetCDF open attempts failed for {tile_file}")
                    return None
            return None
    
    def _open_netcdf_with_netcdf4(self, tile_file):
        """
        Open a NetCDF file using netCDF4 library and convert to rasterio-compatible format.
        This is a fallback when direct rasterio opening fails.
        
        Args:
            tile_file (str): Path to the NetCDF file
            
        Returns:
            rasterio.DatasetReader: Opened dataset or raises exception
        """
        import numpy as np
        import tempfile
        
        # Open NetCDF file
        nc = netCDF4.Dataset(tile_file, 'r')
        
        # Find the data variable (usually 2D or 3D)
        data_var = None
        for var_name in nc.variables:
            var = nc.variables[var_name]
            if len(var.dimensions) >= 2:
                data_var = var_name
                break
        
        if data_var is None:
            nc.close()
            raise Exception("No 2D data variable found in NetCDF file")
        
        # Read the data (handle 3D arrays by taking first slice if needed)
        data = nc.variables[data_var][:]
        if len(data.shape) == 3:
            data = data[0, :, :]  # Take first band if 3D
        
        # Get coordinate variables
        dims = nc.variables[data_var].dimensions
        lat_var = None
        lon_var = None
        
        for dim in dims:
            if dim in nc.variables:
                var = nc.variables[dim]
                if hasattr(var, 'standard_name'):
                    if 'lat' in var.standard_name.lower():
                        lat_var = dim
                    elif 'lon' in var.standard_name.lower():
                        lon_var = dim
        
        # Fallback: try common names
        if lat_var is None:
            for name in ['lat', 'latitude', 'y']:
                if name in nc.variables:
                    lat_var = name
                    break
        if lon_var is None:
            for name in ['lon', 'longitude', 'x']:
                if name in nc.variables:
                    lon_var = name
                    break
        
        if lat_var is None or lon_var is None:
            nc.close()
            raise Exception("Could not find latitude/longitude variables in NetCDF file")
        
        # Get coordinates
        lats = nc.variables[lat_var][:]
        lons = nc.variables[lon_var][:]
        
        # Calculate transform
        lat_min, lat_max = float(lats.min()), float(lats.max())
        lon_min, lon_max = float(lons.min()), float(lons.max())
        
        # Calculate pixel size
        if len(lats) > 1:
            lat_res = abs(float(lats[1] - lats[0]))
        else:
            lat_res = abs(float(lat_max - lat_min))
        
        if len(lons) > 1:
            lon_res = abs(float(lons[1] - lons[0]))
        else:
            lon_res = abs(float(lon_max - lon_min))
        
        # Create transform
        from rasterio.transform import from_bounds
        transform = from_bounds(lon_min, lat_min, lon_max, lat_max, 
                               data.shape[1], data.shape[0])
        
        # Get CRS if available
        crs = None
        if hasattr(nc, 'crs') or hasattr(nc, 'spatial_ref'):
            try:
                import rasterio.crs
                # Try to get CRS from global attributes
                if hasattr(nc, 'crs_wkt'):
                    crs = rasterio.crs.CRS.from_wkt(nc.crs_wkt)
                elif hasattr(nc, 'epsg'):
                    crs = rasterio.crs.CRS.from_epsg(int(nc.epsg))
            except:
                pass
        
        if crs is None:
            crs = rasterio.crs.CRS.from_epsg(4326)  # Default to WGS84
        
        # Create a temporary GeoTIFF
        temp_tiff = tempfile.NamedTemporaryFile(suffix='.tif', delete=False)
        temp_tiff.close()
        
        # Write data to temporary GeoTIFF
        with rasterio.open(
            temp_tiff.name, 'w',
            driver='GTiff',
            height=data.shape[0],
            width=data.shape[1],
            count=1,
            dtype=data.dtype,
            crs=crs,
            transform=transform,
            nodata=nc.variables[data_var]._FillValue if hasattr(nc.variables[data_var], '_FillValue') else None
        ) as dst:
            dst.write(data, 1)
        
        nc.close()
        
        # Open the temporary GeoTIFF with rasterio
        dataset = rasterio.open(temp_tiff.name)
        # Store temp file path so we can clean it up later
        dataset._temp_file = temp_tiff.name
        return dataset
    
    def run(self):
        try:
            print("[DEBUG] MosaicWorker.run() started")
            self.progress.emit("Starting mosaicking process...")
            
            if not self.downloaded_tile_files:
                print("[DEBUG] No tiles to mosaic")
                self.finished.emit(False, "No tiles to mosaic")
                return
            
            print(f"[DEBUG] Mosaicking {len(self.downloaded_tile_files)} tiles...")
            self.progress.emit(f"Mosaicking {len(self.downloaded_tile_files)} tiles...")
            
            # Check if all tile files exist and are readable
            print("[DEBUG] Checking tile files...")
            missing_files = []
            for f in self.downloaded_tile_files:
                if not os.path.exists(f):
                    print(f"[DEBUG] Missing file: {f}")
                    missing_files.append(f)
                else:
                    try:
                        with open(f, 'rb') as test_file:
                            test_file.read(1)
                        print(f"[DEBUG] File readable: {os.path.basename(f)}")
                    except Exception as e:
                        print(f"[DEBUG] File not readable: {f} - {e}")
                        missing_files.append(f)
            
            if missing_files:
                print(f"[DEBUG] Missing files: {missing_files}")
                self.finished.emit(False, f"Missing or unreadable tile files: {missing_files}")
                return
            
            if rasterio is None:
                print("[DEBUG] rasterio not available")
                self.finished.emit(False, "rasterio not available for mosaicking")
                return
            
            # Create output filename for mosaicked file
            current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
            mosaic_filename = f"gmrt_{self.layer_type}_mosaic_{current_time}.tif"
            self.mosaic_path = os.path.join(self.download_dir, mosaic_filename)
            print(f"[DEBUG] Output mosaic path: {self.mosaic_path}")
            self.progress.emit(f"Output mosaic path: {self.mosaic_path}")
            
            # Read all tile files and get their bounds
            print("[DEBUG] Opening tile files with rasterio...")
            datasets = []
            bounds_list = []
            cell_sizes = []
            
            self.progress.emit("Opening tile files with rasterio...")
            for i, tile_file in enumerate(self.downloaded_tile_files):
                if os.path.exists(tile_file):
                    try:
                        print(f"[DEBUG] Opening tile {i+1}/{len(self.downloaded_tile_files)}: {os.path.basename(tile_file)}")
                        dataset = self._open_raster_file(tile_file)
                        if dataset is None:
                            print(f"[DEBUG] Failed to open tile {tile_file}")
                            continue
                        datasets.append(dataset)
                        bounds_list.append(dataset.bounds)
                        
                        # Get cell size information
                        transform = dataset.transform
                        cell_size_x = abs(transform[0])
                        cell_size_y = abs(transform[4])
                        cell_sizes.append((cell_size_x, cell_size_y))
                        print(f"[DEBUG] Successfully loaded tile: {os.path.basename(tile_file)} (cell size: {cell_size_x:.2f}m x {cell_size_y:.2f}m)")
                        
                    except Exception as e:
                        print(f"[DEBUG] Error reading tile {tile_file}: {str(e)}")
                        import traceback
                        print(f"[DEBUG] Traceback: {traceback.format_exc()}")
                        continue
            
            if not datasets:
                print("[DEBUG] No valid tiles found for mosaicking")
                self.finished.emit(False, "No valid tiles found for mosaicking")
                return
            
            # Use rasterio for mosaicking
            print("[DEBUG] Using rasterio for mosaicking")
            self.progress.emit("Using rasterio for mosaicking...")
            
            fallback_datasets = []
            try:
                # Reopen datasets for rasterio fallback
                print("[DEBUG] Reopening datasets for rasterio fallback...")
                for tile_file in self.downloaded_tile_files:
                    if os.path.exists(tile_file):
                        try:
                            dataset = self._open_raster_file(tile_file)
                            if dataset is None:
                                print(f"[DEBUG] Failed to reopen tile {tile_file}")
                                continue
                            fallback_datasets.append(dataset)
                            print(f"[DEBUG] Reopened tile: {os.path.basename(tile_file)}")
                        except Exception as e:
                            print(f"[DEBUG] Error reopening tile {tile_file}: {str(e)}")
                            import traceback
                            print(f"[DEBUG] Traceback: {traceback.format_exc()}")
                            continue
                
                if fallback_datasets:
                    print(f"[DEBUG] Using {len(fallback_datasets)} datasets for rasterio fallback")
                    original_bounds = (self.west_spin.value(), self.south_spin.value(), 
                                     self.east_spin.value(), self.north_spin.value())
                    self._mosaic_with_rasterio_fallback(fallback_datasets, self.mosaic_path, original_bounds)
                    print("[DEBUG] Rasterio fallback completed successfully")
                    self.finished.emit(True, "Rasterio mosaicking completed successfully")
                else:
                    print("[DEBUG] No valid tiles found for fallback")
                    self.finished.emit(False, "No valid tiles found for fallback")
                    
            except Exception as fallback_error:
                print(f"[DEBUG] Rasterio fallback failed: {str(fallback_error)}")
                self.finished.emit(False, f"Rasterio fallback also failed: {str(fallback_error)}")
            finally:
                # Clean up fallback_datasets if they weren't handled by _mosaic_with_rasterio_fallback
                # (This handles the case where exception occurs before calling the fallback function)
                for dataset in fallback_datasets:
                    try:
                        temp_file = None
                        if hasattr(dataset, '_temp_file'):
                            temp_file = dataset._temp_file
                        dataset.close()
                        if temp_file and os.path.exists(temp_file):
                            try:
                                os.remove(temp_file)
                                print(f"[DEBUG] Cleaned up temporary file: {temp_file}")
                            except:
                                pass
                    except:
                        pass
            
        except Exception as e:
            import traceback
            print(f"[DEBUG] CRITICAL ERROR in MosaicWorker: {str(e)}")
            print(f"[DEBUG] Traceback: {traceback.format_exc()}")
            self.finished.emit(False, f"Mosaic worker error: {str(e)}\n{traceback.format_exc()}")
    
    def _mosaic_with_rasterio_fallback(self, datasets, mosaic_path, original_bounds):
        """Improved rasterio mosaicking with proper cell size handling"""
        print("[DEBUG] Starting improved rasterio mosaicking...")
        import rasterio
        from rasterio.warp import calculate_default_transform, reproject, Resampling
        from rasterio.windows import from_bounds
        import numpy as np
        
        try:
            # Check if we have valid datasets
            if not datasets:
                raise Exception("No valid datasets provided for rasterio fallback")
            
            # Analyze all datasets to determine the finest resolution
            print("[DEBUG] Analyzing tile resolutions...")
            cell_sizes = []
            bounds_list = []
            
            for i, dataset in enumerate(datasets):
                transform = dataset.transform
                bounds = dataset.bounds
                print(f"[DEBUG] Tile {i+1} transform: {transform}")
                print(f"[DEBUG] Tile {i+1} bounds: {bounds}")
                print(f"[DEBUG] Tile {i+1} CRS: {dataset.crs}")
                print(f"[DEBUG] Tile {i+1} size: {dataset.width} x {dataset.height}")
                
                # Get cell size from transform (units depend on CRS - degrees for geographic, meters for projected)
                # Note: transform[4] is typically negative for geographic CRS (y-axis points down in image space)
                cell_size_x = abs(transform[0])
                cell_size_y = abs(transform[4])  # Use absolute value since y-axis is typically flipped
                
                # Validate cell size from transform
                if cell_size_x <= 0 or cell_size_y <= 0:
                    # Fallback: calculate from bounds and dimensions
                    if bounds[2] > bounds[0] and bounds[3] > bounds[1] and dataset.width > 0 and dataset.height > 0:
                        cell_size_x = (bounds[2] - bounds[0]) / dataset.width
                        cell_size_y = abs((bounds[3] - bounds[1]) / dataset.height)
                        print(f"[DEBUG] Tile {i+1} using fallback cell size calculation")
                
                # Calculate cell size from actual dimensions and bounds for validation
                if bounds[2] > bounds[0] and bounds[3] > bounds[1] and dataset.width > 0 and dataset.height > 0:
                    calc_cell_x = (bounds[2] - bounds[0]) / dataset.width
                    calc_cell_y = (bounds[3] - bounds[1]) / dataset.height
                    print(f"[DEBUG] Tile {i+1} calculated cell size from bounds: {calc_cell_x:.9f} x {calc_cell_y:.9f}")
                
                cell_sizes.append((cell_size_x, cell_size_y))
                bounds_list.append(bounds)
                print(f"[DEBUG] Tile {i+1}: transform cell size {cell_size_x:.9f} x {cell_size_y:.9f}")
            
            # Find the finest resolution (smallest cell size)
            min_cell_x = min(cell_size[0] for cell_size in cell_sizes)
            min_cell_y = min(cell_size[1] for cell_size in cell_sizes)
            print(f"[DEBUG] Finest resolution: {min_cell_x:.9f} x {min_cell_y:.9f}")
            
            # Calculate overall bounds
            min_x = min(bounds[0] for bounds in bounds_list)
            min_y = min(bounds[1] for bounds in bounds_list)
            max_x = max(bounds[2] for bounds in bounds_list)
            max_y = max(bounds[3] for bounds in bounds_list)
            
            print(f"[DEBUG] Overall bounds: min_x={min_x:.6f}, max_x={max_x:.6f}, min_y={min_y:.6f}, max_y={max_y:.6f}")
            print(f"[DEBUG] Bounds span: x={max_x - min_x:.6f}, y={max_y - min_y:.6f}")
            
            # Validate bounds
            if max_x <= min_x or max_y <= min_y:
                raise Exception(f"Invalid bounds: max_x ({max_x}) <= min_x ({min_x}) or max_y ({max_y}) <= min_y ({min_y})")
            
            # Validate cell sizes
            if min_cell_x <= 0 or min_cell_y <= 0:
                raise Exception(f"Invalid cell size: min_cell_x={min_cell_x}, min_cell_y={min_cell_y}")
            
            # Calculate dimensions for the finest resolution
            # Note: cell_size and bounds are in the same units (degrees for geographic CRS)
            width = int((max_x - min_x) / min_cell_x)
            height = int((max_y - min_y) / min_cell_y)
            
            print(f"[DEBUG] Calculated output dimensions: {width} x {height} pixels")
            
            # Validate dimensions
            if width <= 0 or height <= 0:
                raise Exception(f"Invalid dimensions calculated: width={width}, height={height}. "
                              f"This may indicate a units mismatch between bounds and cell size.")
            
            print(f"[DEBUG] Validated output dimensions: {width} x {height} pixels")
            
            # Create the output transform
            output_transform = rasterio.transform.from_bounds(min_x, min_y, max_x, max_y, width, height)
            
            # Initialize the output array with nodata values
            output_array = np.full((height, width), -99999, dtype=np.float32)
            
            # Process each dataset
            print("[DEBUG] Processing tiles with proper resampling...")
            for i, dataset in enumerate(datasets):
                print(f"[DEBUG] Processing tile {i+1}/{len(datasets)}...")
                
                # Read the data
                data = dataset.read(1)  # Read first band
                
                # Replace NaN values with nodata
                data = np.where(np.isnan(data), -99999, data)
                
                # Replace infinite values
                data = np.where(np.isinf(data), -99999, data)
                
                # Get the source transform and bounds
                src_transform = dataset.transform
                src_bounds = dataset.bounds
                
                print(f"[DEBUG] Tile {i+1} bounds: {src_bounds}")
                print(f"[DEBUG] Tile {i+1} data shape: {data.shape}")
                print(f"[DEBUG] Tile {i+1} transform: {src_transform}")
                
                # Calculate the window in the output array
                col_start = int((src_bounds[0] - min_x) / min_cell_x)
                col_end = int((src_bounds[2] - min_x) / min_cell_x)
                row_start = int((max_y - src_bounds[3]) / min_cell_y)
                row_end = int((max_y - src_bounds[1]) / min_cell_y)
                
                print(f"[DEBUG] Tile {i+1} output window (before bounds check): row {row_start}-{row_end}, col {col_start}-{col_end}")
                
                # Ensure indices are within bounds
                col_start = max(0, col_start)
                col_end = min(width, col_end)
                row_start = max(0, row_start)
                row_end = min(height, row_end)
                
                print(f"[DEBUG] Tile {i+1} output window (after bounds check): row {row_start}-{row_end}, col {col_start}-{col_end}")
                
                if col_end <= col_start or row_end <= row_start:
                    print(f"[DEBUG] Warning: Tile {i+1} has no overlap with output area")
                    continue
                
                # Calculate the corresponding window in the source data
                # Use rasterio's transform to convert coordinates to pixel coordinates
                from rasterio.transform import rowcol
                
                # Calculate the bounds of the output window in geographic coordinates
                output_west = min_x + col_start * min_cell_x
                output_east = min_x + col_end * min_cell_x
                output_north = max_y - row_start * min_cell_y
                output_south = max_y - row_end * min_cell_y
                
                print(f"[DEBUG] Tile {i+1} output bounds: W={output_west:.6f}, E={output_east:.6f}, S={output_south:.6f}, N={output_north:.6f}")
                
                # Convert these bounds to source pixel coordinates
                src_row_start, src_col_start = rowcol(src_transform, output_west, output_north)
                src_row_end, src_col_end = rowcol(src_transform, output_east, output_south)
                
                print(f"[DEBUG] Tile {i+1} source window (before bounds check): row {src_row_start}-{src_row_end}, col {src_col_start}-{src_col_end}")
                
                # Ensure indices are within bounds
                src_row_start = max(0, src_row_start)
                src_row_end = min(data.shape[0], src_row_end)
                src_col_start = max(0, src_col_start)
                src_col_end = min(data.shape[1], src_col_end)
                
                if src_col_end <= src_col_start or src_row_end <= src_row_start:
                    print(f"[DEBUG] Warning: Tile {i+1} source window is invalid (src: {src_row_start}-{src_row_end}, {src_col_start}-{src_col_end})")
                    continue
                
                # Extract the source data window
                src_data = data[src_row_start:src_row_end, src_col_start:src_col_end]
                
                # Resample if necessary
                if src_data.shape != (row_end - row_start, col_end - col_start):
                    # Use scipy for resampling
                    from scipy.ndimage import zoom
                    zoom_factors = (
                        (row_end - row_start) / src_data.shape[0],
                        (col_end - col_start) / src_data.shape[1]
                    )
                    src_data = zoom(src_data, zoom_factors, order=1, mode='nearest')
                
                # Merge data using maximum (shallowest) values
                output_window = output_array[row_start:row_end, col_start:col_end]
                valid_mask = (src_data != -99999) & (src_data != np.nan) & (src_data != np.inf)
                
                if np.any(valid_mask):
                    # Only update where source data is valid and shallower (more positive/less negative)
                    update_mask = valid_mask & (
                        (output_window == -99999) | 
                        (src_data > output_window)
                    )
                    output_window[update_mask] = src_data[update_mask]
                    output_array[row_start:row_end, col_start:col_end] = output_window
                    
                    valid_count = np.sum(valid_mask)
                    print(f"[DEBUG] Tile {i+1}: Updated {valid_count} valid pixels")
                else:
                    print(f"[DEBUG] Tile {i+1}: No valid data to merge")
            
            # Final data validation
            print("[DEBUG] Performing final data validation...")
            output_array = self.validate_bathymetry_data(output_array)
            
            # Crop to exact requested bounds if different from overall bounds
            if original_bounds != (min_x, min_y, max_x, max_y):
                print("[DEBUG] Cropping to exact requested bounds...")
                try:
                    window = from_bounds(*original_bounds, output_transform)
                    row_start = max(0, int(window.row_off))
                    row_end = min(height, int(window.row_off + window.height))
                    col_start = max(0, int(window.col_off))
                    col_end = min(width, int(window.col_off + window.width))
                    
                    output_array = output_array[row_start:row_end, col_start:col_end]
                    output_transform = from_bounds(*original_bounds, col_end - col_start, row_end - row_start)
                except Exception as e:
                    print(f"[DEBUG] Warning: Could not crop to exact bounds: {e}")
            
            # Create output profile
            profile = {
                'driver': 'GTiff',
                'height': output_array.shape[0],
                'width': output_array.shape[1],
                'count': 1,
                'dtype': output_array.dtype,
                'crs': datasets[0].crs,
                'transform': output_transform,
                'compress': 'lzw',
                'tiled': True,
                'nodata': -99999
            }
            
            # Write the mosaic
            print(f"[DEBUG] Writing mosaic to: {mosaic_path}")
            with rasterio.open(mosaic_path, 'w', **profile) as dst:
                dst.write(output_array, 1)
            
            # Log final statistics
            valid_pixels = np.sum(output_array != -99999)
            total_pixels = output_array.size
            print(f"[DEBUG] Mosaic completed: {valid_pixels}/{total_pixels} valid pixels")
            print(f"[DEBUG] Data range: {np.nanmin(output_array[output_array != -99999]):.2f} to {np.nanmax(output_array[output_array != -99999]):.2f}")
            
            # Log final mosaic cell size
            final_cell_x = abs(output_transform[0])
            final_cell_y = abs(output_transform[4])
            print(f"[DEBUG] Final mosaic cell size: {final_cell_x:.6f}m x {final_cell_y:.6f}m")
            
        except Exception as e:
            print(f"[DEBUG] Rasterio fallback failed: {str(e)}")
            import traceback
            print(f"[DEBUG] Traceback: {traceback.format_exc()}")
            raise
        finally:
            # Close all datasets and clean up temporary files
            for dataset in datasets:
                try:
                    # Clean up temporary files if they exist (from NetCDF conversion)
                    temp_file = None
                    if hasattr(dataset, '_temp_file'):
                        temp_file = dataset._temp_file
                    dataset.close()
                    # Delete temporary file after closing
                    if temp_file and os.path.exists(temp_file):
                        try:
                            os.remove(temp_file)
                            print(f"[DEBUG] Cleaned up temporary file: {temp_file}")
                        except Exception as e:
                            print(f"[DEBUG] Could not delete temporary file {temp_file}: {e}")
                except Exception as e:
                    print(f"[DEBUG] Error closing dataset: {e}")
                    pass

    def validate_bathymetry_data(self, data):
        """
        Validate bathymetry/topography data and set unrealistic values to nodata.
        
        Args:
            data (numpy.ndarray): Input bathymetry/topography data
            
        Returns:
            numpy.ndarray: Data with unrealistic values set to nodata
        """
        import numpy as np
        
        # Define realistic bathymetry/topography limits for Earth
        # These are conservative limits based on known Earth topography
        max_elevation = 9000.0    # Mount Everest is ~8848m, use 9000m as upper limit
        min_elevation = -12000.0  # Mariana Trench is ~11000m, use 12000m as lower limit
        
        # Create a copy to avoid modifying the original
        validated_data = data.copy()
        
        # Set unrealistic values to nodata value
        nodata_value = -99999
        
        # Count original nodata values
        original_nodata = (validated_data == nodata_value).sum()
        
        # Values above max_elevation (too high)
        too_high = validated_data > max_elevation
        # Values below min_elevation (too deep)
        too_deep = validated_data < min_elevation
        
        # Set unrealistic values to nodata value (-99999)
        validated_data[too_high] = nodata_value
        validated_data[too_deep] = nodata_value
        
        # Count new nodata values
        new_nodata = (validated_data == nodata_value).sum()
        invalid_count = new_nodata - original_nodata
        
        if invalid_count > 0:
            print(f"[DEBUG] Set {invalid_count} unrealistic values to nodata (-99999) (outside {min_elevation}m to {max_elevation}m range)")
        else:
            print("[DEBUG] All bathymetry/topography values are within realistic range")
        
        return validated_data


class DownloadWorker(QThread):
    """
    Worker thread for downloading bathymetry data files from GMRT GridServer.
    
    This class runs in a separate thread to prevent the GUI from freezing
    during large file downloads. It handles streaming downloads and provides
    detailed error messages for different failure scenarios.
    """
    # Signals to communicate with the main thread
    finished = pyqtSignal(bool, str)  # (success, filename/error_message)
    
    def __init__(self, params, filename):
        """
        Initialize the download worker with request parameters and target filename.
        
        Args:
            params (dict): Parameters for the GMRT GridServer API request
            filename (str): Full path where the downloaded file should be saved
        """
        super().__init__()
        self.params = params
        self.filename = filename
    
    def get_error_message(self, status_code):
        """
        Get human-readable error message for HTTP status codes.
        
        Args:
            status_code (int): HTTP status code from the server response
            
        Returns:
            str: Human-readable error message
        """
        # Common GMRT error messages
        error_messages = {
            404: "No Data Returned - The requested area may be outside available data coverage",
            413: "Request Too Large - The requested area is too large for the specified resolution"
        }
        return error_messages.get(status_code, f"HTTP Error {status_code}")
    
    def run(self):
        """
        Main worker method that downloads the bathymetry data file.
        
        This method runs in a separate thread and downloads the file in chunks
        to handle large files efficiently. It provides detailed error handling
        for various failure scenarios.
        """
        try:
            # Log the grid download request
            param_str = ", ".join([f"{k}={v}" for k, v in self.params.items()])
            print(f"[DownloadWorker] Downloading bathymetry grid: {param_str}")
            print(f"[DEBUG] DownloadWorker.run: sending request to {GMRT_URL}")
            with requests.get(GMRT_URL, params=self.params, stream=True, timeout=120) as r:
                print(f"[DEBUG] DownloadWorker.run: response status {r.status_code}")
                if r.status_code == 200:
                    # Successfully connected, download the file in chunks
                    total_bytes = 0
                    with open(self.filename, 'wb') as f:
                        for chunk in r.iter_content(chunk_size=8192):  # 8KB chunks
                            if chunk:
                                f.write(chunk)
                                total_bytes += len(chunk)
                    
                    print(f"[DownloadWorker] Grid download completed successfully ({total_bytes} bytes)")
                    
                    # Download completed successfully
                    self.finished.emit(True, self.filename)
                else:
                    # Handle HTTP error responses
                    error_msg = self.get_error_message(r.status_code)
                    
                    if r.status_code == 404:
                        # Try to get more specific error from response content
                        try:
                            content = r.text.lower()
                            if "invalid output format" in content:
                                error_msg = "Invalid Output Format Specified"
                            elif "invalid layer" in content:
                                error_msg = "Invalid Layer Specified"
                            elif "invalid bounds" in content or "w/e/s/n" in content:
                                error_msg = "Invalid W/E/S/N bounds specified"
                            elif "invalid resolution" in content:
                                error_msg = "Invalid Resolution"
                            else:
                                error_msg = "No Data Returned - The requested area may be outside available data coverage"
                        except:
                            # If we can't parse the error content, use the default message
                            pass
                    
                    detailed_error = f"Server Error {r.status_code}: {error_msg}"
                    print(f"[DownloadWorker] Grid download failed: {detailed_error}")
                    self.finished.emit(False, detailed_error)
                    
        except requests.exceptions.Timeout:
            # Network timeout occurred
            print(f"[DownloadWorker] Grid download timeout")
            self.finished.emit(False, "Request Timeout - The server took too long to respond")
        except requests.exceptions.ConnectionError:
            # Network connection failed
            print(f"[DownloadWorker] Grid download connection error")
            self.finished.emit(False, "Connection Error - Unable to connect to GMRT server")
        except Exception as e:
            # Any other unexpected error
            print(f"[DownloadWorker] Grid download error: {str(e)}")
            self.finished.emit(False, f"Download Error: {str(e)}")

class GMRTGrabber(QWidget):
    """
    Main GUI application for downloading GMRT bathymetry data.
    
    This class provides a comprehensive interface for:
    - Setting geographic boundaries and download parameters
    - Previewing the selected area using GMRT ImageServer
    - Downloading bathymetry data in various formats
    - Managing tiled downloads for large areas
    - Logging all operations with timestamps
    """
    
    def __init__(self):
        print("[DEBUG] Entering GMRTGrabber.__init__")
        super().__init__()
        self.setWindowTitle(f"GMRT Bathymetry Grid Downloader v{__version__} - pjohnson@ccom.unh.edu")
        print("[DEBUG] Set window title")
        # Set window icon
        icon_path = os.path.join(os.path.dirname(__file__), "media", "GMRT-logo2020.ico")
        if os.path.exists(icon_path):
            self.setWindowIcon(QIcon(icon_path))
            print(f"[DEBUG] Set window icon: {icon_path}")
        else:
            print(f"[DEBUG] Icon file not found: {icon_path}")
        # Data structures for managing tiled downloads
        self.tiles_to_download = []        # List of tile boundaries to download
        self.current_tile_index = 0        # Current tile being downloaded
        self.download_dir = ""             # Directory for saving downloaded files
        self.downloaded_tile_files = []    # List of successfully downloaded tile files
        print("[DEBUG] Initialized download state variables")
        # Timer for managing sequential tile downloads
        self.download_timer = QTimer()
        self.download_timer.timeout.connect(self.download_next_tile)
        print("[DEBUG] Initialized download timer")
        # Configuration and state management
        self.config_file = os.path.join(os.path.dirname(__file__), "gmrtgrab_config.json")
        print(f"[DEBUG] Config file path: {self.config_file}")
        self.last_download_dir = self.load_last_download_dir()
        print(f"[DEBUG] Last download dir: {self.last_download_dir}")
        # Worker threads for background operations
        self.current_worker = None         # Current download worker
        self.current_map_worker = None     # Current map preview worker
        print("[DEBUG] About to call init_ui()")
        # Initialize the user interface
        self.init_ui()
        print("[DEBUG] Finished init_ui()")
        # Set default window size
        self.resize(1200, 800)
        print("[DEBUG] Window resized to 1200x800")

    def load_last_download_dir(self):
        print("[DEBUG] Entering load_last_download_dir")
        """
        Load the last used download directory from the configuration file.
        
        This method attempts to read the configuration file and extract
        the last download directory. If the file doesn't exist or is
        invalid, it returns the user's home directory as a default.
        
        Returns:
            str: Path to the last download directory or user's home directory
        """
        try:
            if os.path.exists(self.config_file):
                print("[DEBUG] Config file exists")
                with open(self.config_file, 'r') as f:
                    config = json.load(f)
                    last_dir = config.get('last_download_dir', '')
                    print(f"[DEBUG] Loaded last_download_dir from config: {last_dir}")
                    # Verify the directory still exists
                    if last_dir and os.path.exists(last_dir):
                        print("[DEBUG] Last download dir exists on disk")
                        return last_dir
        except Exception as e:
            print(f"[DEBUG] Exception in load_last_download_dir: {e}")
            # If any error occurs (file doesn't exist, invalid JSON, etc.),
            # we'll use the default directory
            pass
        
        # Default to user's home directory
        print("[DEBUG] Returning default home directory")
        return os.path.expanduser("~")

    def get_layer_type(self):
        """
        Get the layer type value for API calls, mapping display text to API values.
        
        Returns:
            str: The layer type value to send to the API
        """
        display_text = self.layer_combo.currentText()
        # Map display text to API values
        layer_mapping = {
            "Topo-Bathy": "topo",
            "Topo-Bathy (Observed Only)": "topo-mask"
        }
        return layer_mapping.get(display_text, display_text)

    def save_last_download_dir(self, directory):
        """
        Save the current download directory to the configuration file.
        
        This method saves the directory path so it can be restored
        the next time the application is launched. It also updates
        the instance variable so the directory is remembered within
        the same session. If saving fails, the error is silently
        ignored to prevent application crashes.
        
        Args:
            directory (str): Path to the directory to save
        """
        # Update the instance variable so it's remembered within the same session
        self.last_download_dir = directory
        try:
            config = {'last_download_dir': directory}
            with open(self.config_file, 'w') as f:
                json.dump(config, f, indent=2)
        except Exception:
            # Silently fail if we can't save the configuration
            # This prevents application crashes due to file system issues
            pass

    def init_ui(self):
        print("[DEBUG] Entering init_ui")
        """
        Initialize and create the complete user interface.
        
        This method creates a split-pane layout with controls on the left
        and map preview on the right. It sets up all form elements,
        buttons, and connects signals to their respective slots.
        """
        # Create main splitter for left (controls) and right (map) panels
        main_splitter = QSplitter(Qt.Orientation.Horizontal)
        
        # Left panel for controls
        left_panel = QWidget()
        left_panel.setMaximumWidth(450)  # Set maximum width to prevent over-expansion
        left_layout = QVBoxLayout()
        
        # === GRID PARAMETERS SECTION ===
        form_group = QGroupBox("Download Parameters")
        form_layout = QFormLayout()

        # Area of Interest subgroup inside Grid Parameters
        aoi_group = QGroupBox("Area of Interest")
        aoi_form = QFormLayout()

        # Geographic boundary controls (longitude/latitude)
        # These spin boxes allow users to set the exact geographic bounds
        # for the bathymetry data they want to download
        
        # Western boundary (minimum longitude)
        self.west_spin = QDoubleSpinBox()
        self.west_spin.setRange(-180, 180)  # Valid longitude range
        self.west_spin.setDecimals(6)       # 6 decimal places for precision
        self.west_spin.valueChanged.connect(self.update_map_preview)  # Auto-update map
        aoi_form.addRow("West (min lon)", self.west_spin)

        # Eastern boundary (maximum longitude)
        self.east_spin = QDoubleSpinBox()
        self.east_spin.setRange(-180, 180)  # Valid longitude range
        self.east_spin.setDecimals(6)       # 6 decimal places for precision
        self.east_spin.valueChanged.connect(self.update_map_preview)  # Auto-update map
        aoi_form.addRow("East (max lon)", self.east_spin)

        # Southern boundary (minimum latitude)
        self.south_spin = QDoubleSpinBox()
        self.south_spin.setRange(-85, 85)   # Valid latitude range (GMRT data limit)
        self.south_spin.setDecimals(6)      # 6 decimal places for precision
        self.south_spin.valueChanged.connect(self.update_map_preview)  # Auto-update map
        aoi_form.addRow("South (min lat)", self.south_spin)

        # Northern boundary (maximum latitude)
        self.north_spin = QDoubleSpinBox()
        self.north_spin.setRange(-85, 85)   # Valid latitude range (GMRT data limit)
        self.north_spin.setDecimals(6)      # 6 decimal places for precision
        self.north_spin.valueChanged.connect(self.update_map_preview)  # Auto-update map
        aoi_form.addRow("North (max lat)", self.north_spin)

        aoi_group.setLayout(aoi_form)
        form_layout.addRow(aoi_group)

        # === OUTPUT FORMAT AND RESOLUTION CONTROLS ===

        # Grid Parameters subgroup inside Download Parameters
        grid_group = QGroupBox("Grid Parameters")
        grid_form = QFormLayout()
        
        # Output format selection
        # Different formats are suitable for different applications
        self.format_combo = QComboBox()
        self.format_combo.addItems([
            "geotiff",    # GeoTIFF - Raster format with embedded georeference
            "netcdf",     # NetCDF - Scientific data format with metadata
            "coards",     # COARDS - ASCII grid format
            "esriascii"   # ESRI ASCII - Simple ASCII grid format
        ])
        grid_form.addRow("Format", self.format_combo)

        # Cell resolution selection
        # Allows users to specify meter-per-pixel resolution
        self.mres_combo = QComboBox()
        self.mres_combo.addItems(["100", "200", "400", "800"])
        self.mres_combo.setCurrentText("400")  # Default to 400 meters/pixel
        grid_form.addRow("Cell Resolution (meters/pixel)", self.mres_combo)

        # Data layer selection
        # Different layers provide different types of bathymetry data
        self.layer_combo = QComboBox()
        self.layer_combo.addItems([
            "Topo-Bathy",                    # Topography - Standard bathymetry data
            "Topo-Bathy (Observed Only)"     # Topography with mask - High-resolution data only
        ])
        grid_form.addRow("Layer", self.layer_combo)

        # Split Bathymetry/Topography option
        # When enabled, downloaded grids will be split into topography (>=0) and bathymetry (<0) files
        self.split_checkbox = QCheckBox("Split Grid Into Bathymetry and Topography Grids")
        self.split_checkbox.setChecked(False)  # Default to unchecked
        self.split_checkbox.setToolTip(
            "If checked, each downloaded grid will be split into two files: one with values >= 0 (topography, _topo) and one with values < 0 (bathymetry, _bathy)."
        )
        grid_form.addRow(self.split_checkbox)

        grid_group.setLayout(grid_form)
        form_layout.addRow(grid_group)

        # Tiling Parameters subgroup inside Download Parameters
        tiling_group = QGroupBox("Tiling Parameters")
        tiling_form = QFormLayout()

        # Tiled download option
        # For large areas, breaking into tiles can prevent timeouts and improve reliability
        self.tile_checkbox = QCheckBox("Tile Dataset for Download")
        self.tile_checkbox.setToolTip(
            "Break large datasets into tiles with configurable size and automatic overlap based on cell resolution"
        )
        tiling_form.addRow(self.tile_checkbox)
        
        # Tile size parameter
        self.tile_size_combo = QComboBox()
        self.tile_size_combo.addItems(["1", "2", "3", "4"])
        self.tile_size_combo.setCurrentText("2")  # Default to 2 degrees
        self.tile_size_combo.setToolTip(
            "Size of each tile in degrees (applies when tiling is enabled)"
        )
        tiling_form.addRow("Tile Size (deg)", self.tile_size_combo)

        # Mosaic tiles option
        # When enabled, automatically assemble all tiles into a single GeoTIFF
        self.mosaic_checkbox = QCheckBox("Mosaic Downloaded Tiles")
        self.mosaic_checkbox.setChecked(True)  # Default to checked
        self.mosaic_checkbox.setToolTip(
            "Automatically assemble all tiles into a single GeoTIFF file"
        )
        tiling_form.addRow(self.mosaic_checkbox)
        
        # Delete tiles option
        # When enabled, individual tile files are deleted after mosaicking
        self.delete_tiles_checkbox = QCheckBox("Delete Tiles After Mosaicing")
        self.delete_tiles_checkbox.setChecked(True)  # Start checked by default
        self.delete_tiles_checkbox.setToolTip(
            "Delete individual tile files after mosaicking (only applies when mosaicking is enabled)"
        )
        tiling_form.addRow(self.delete_tiles_checkbox)

        tiling_group.setLayout(tiling_form)
        form_layout.addRow(tiling_group)
        
        # Connect tile checkbox to enable/disable mosaic checkbox
        self.tile_checkbox.toggled.connect(self.on_tile_checkbox_toggled)
        # Connect mosaic checkbox to enable/disable delete tiles checkbox
        self.mosaic_checkbox.toggled.connect(self.on_mosaic_checkbox_toggled)
        
        # Connect coordinate spinboxes to update tile checkbox availability
        self.west_spin.valueChanged.connect(self.update_tile_checkbox_availability)
        self.east_spin.valueChanged.connect(self.update_tile_checkbox_availability)
        self.south_spin.valueChanged.connect(self.update_tile_checkbox_availability)
        self.north_spin.valueChanged.connect(self.update_tile_checkbox_availability)
        
        # React to tile size changes and run initial availability check
        self.tile_size_combo.currentTextChanged.connect(self.update_tile_checkbox_availability)
        self.update_tile_checkbox_availability()

        # Complete the grid parameters section
        form_group.setLayout(form_layout)
        left_layout.addWidget(form_group)

        # === MAP PREVIEW CONTROLS ===
        # This section allows users to preview the selected area before downloading
        map_controls_group = QGroupBox("Map Preview")
        map_controls_layout = QVBoxLayout()
        
        # Map control buttons
        map_buttons_layout = QHBoxLayout()
        
        # Refresh map button - manually update the preview
        self.refresh_map_btn = QPushButton("Refresh Map")
        self.refresh_map_btn.clicked.connect(self.update_map_preview)
        map_buttons_layout.addWidget(self.refresh_map_btn)
        
        # High-resolution mask toggle
        # When checked, shows only high-resolution data areas
        self.mask_checkbox = QCheckBox("Show High-Res Mask")
        self.mask_checkbox.setChecked(True)  # Default to showing mask
        self.mask_checkbox.setToolTip("Highlight high-resolution data areas")
        self.mask_checkbox.toggled.connect(self.update_map_preview)  # Auto-update when toggled
        map_buttons_layout.addWidget(self.mask_checkbox)
        
        map_controls_layout.addLayout(map_buttons_layout)
        
        # Map status indicator - shows loading state and errors
        self.map_status_label = QLabel("Map: Ready")
        map_controls_layout.addWidget(self.map_status_label)
        
        map_controls_group.setLayout(map_controls_layout)
        left_layout.addWidget(map_controls_group)

        # === DOWNLOAD CONTROLS ===
        
        # Main download button - initiates the bathymetry data download
        self.download_btn = QPushButton("Download Grid")
        self.download_btn.clicked.connect(self.download_grid)
        left_layout.addWidget(self.download_btn)

        # Status display - shows download progress and results
        self.status_label = QLabel("")
        left_layout.addWidget(self.status_label)

        # === ACTIVITY LOG SECTION ===
        # Provides a detailed log of all operations with timestamps
        log_group = QGroupBox("Activity Log")
        log_layout = QVBoxLayout()
        
        # Text area for displaying log messages
        self.log_area = QTextEdit()
        self.log_area.setMinimumHeight(100)  # Set minimum height for usability
        self.log_area.setReadOnly(True)      # Users can't edit the log
        self.log_area.setFont(QApplication.font())  # Use system font
        log_layout.addWidget(self.log_area)
        
        # Clear log button - allows users to reset the log
        clear_log_btn = QPushButton("Clear Log")
        clear_log_btn.clicked.connect(self.clear_log)
        log_layout.addWidget(clear_log_btn)
        
        log_group.setLayout(log_layout)
        left_layout.addWidget(log_group, 1)  # Add stretch factor to make it expand

        # Complete the left panel layout
        left_panel.setLayout(left_layout)
        
        # === RIGHT PANEL - MAP PREVIEW ===
        # This panel displays the visual preview of the selected area
        right_panel = QWidget()
        right_layout = QVBoxLayout()
        
        # Map preview container
        map_group = QGroupBox("Area Preview")
        map_layout = QVBoxLayout()
        
        # Map display widget - shows the actual bathymetry preview with drawing capability
        self.map_widget = MapWidget()
        self.map_widget.bounds_selected.connect(self.on_rectangle_selected)
        map_layout.addWidget(self.map_widget)

        # Drawing controls
        draw_controls_layout = QHBoxLayout()
        self.draw_rect_btn = QPushButton("Draw Bounds")
        self.draw_rect_btn.setCheckable(True)
        self.draw_rect_btn.setToolTip("Draw a rectangle to set bounds")
        self.draw_rect_btn.toggled.connect(self.map_widget.enable_drawing)
        draw_controls_layout.addWidget(self.draw_rect_btn)

        self.zoom_default_btn = QPushButton("Zoom to Defaults")
        self.zoom_default_btn.setToolTip("Zoom to starting map defaults")
        self.zoom_default_btn.clicked.connect(self.zoom_to_default)
        draw_controls_layout.addWidget(self.zoom_default_btn)
        
        map_layout.addLayout(draw_controls_layout)
        
        map_group.setLayout(map_layout)
        right_layout.addWidget(map_group)
        
        # === CREDIT LINE ===
        # Required attribution for GMRT data usage
        # This citation must be included when using GMRT data
        credit_label = QLabel(
            "Credit: Ryan, W.B.F., S.M. Carbotte, J.O. Coplan, S. O'Hara, A. Melkonian, "
            "R. Arko, R.A. Weissel, V. Ferrini, A. Goodwillie, F. Nitsche, J. Bonczkowski, "
            "and R. Zemsky (2009), Global Multi-Resolution Topography synthesis, "
            "Geochem. Geophys. Geosyst., 10, Q03014, doi: 10.1029/2008GC002332"
        )
        credit_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        credit_label.setWordWrap(True)  # Allow text to wrap to multiple lines
        credit_label.setStyleSheet("QLabel { color: gray; font-size: 9pt; padding: 5px; }")
        credit_label.setMaximumHeight(60)  # Limit height to save space
        right_layout.addWidget(credit_label)
        
        right_panel.setLayout(right_layout)
        
        # === FINAL LAYOUT ASSEMBLY ===
        
        # Add both panels to the main splitter
        main_splitter.addWidget(left_panel)
        main_splitter.addWidget(right_panel)
        main_splitter.setSizes([400, 600])  # Set initial split sizes (left: 400px, right: 600px)
        main_splitter.setStretchFactor(0, 0)  # Left panel should not stretch
        main_splitter.setStretchFactor(1, 1)  # Right panel should stretch to fill space
        
        # Create the main application layout
        main_layout = QVBoxLayout()
        main_layout.addWidget(main_splitter)
        
        # Set the main layout for the application window
        self.setLayout(main_layout)
        print("[DEBUG] About to set main layout")
        print("[DEBUG] Leaving init_ui")
        
        # === INITIALIZATION ===
        
        # Log the application startup
        self.log_message("GMRT Bathymetry Grid Downloader started")
        
        # Block signals to prevent multiple update_map_preview calls during initial setup
        self.west_spin.blockSignals(True)
        self.east_spin.blockSignals(True)
        self.south_spin.blockSignals(True)
        self.north_spin.blockSignals(True)
        # Set default coordinates for a sample area
        self.west_spin.setValue(-180.0)   # Western boundary
        self.east_spin.setValue(180.0)    # Eastern boundary
        self.south_spin.setValue(-85.0)   # Southern boundary (GMRT data limit)
        self.north_spin.setValue(85.0)    # Northern boundary (GMRT data limit)
        self.west_spin.blockSignals(False)
        self.east_spin.blockSignals(False)
        self.south_spin.blockSignals(False)
        self.north_spin.blockSignals(False)
        # Refresh tile availability and tooltips now that defaults are set
        try:
            self.update_tile_checkbox_availability()
        except Exception:
            pass
        # Call update_map_preview once after all values are set
        self.update_map_preview()
        
        # Set initial state of mosaic checkbox (disabled since tile dataset starts unchecked)
        self.mosaic_checkbox.setEnabled(False)
        # Set initial state of delete tiles checkbox (disabled since mosaic starts disabled)
        self.delete_tiles_checkbox.setEnabled(False)

    def update_map_preview(self):
        print("[DEBUG] Entering update_map_preview")
        """
        Update the map preview with the current coordinate settings.
        
        This method validates the coordinates, stops any existing map worker,
        creates a new worker thread to download the map image, and updates
        the UI to show the loading state.
        """
        # Validate coordinates first to ensure they make sense
        west = self.west_spin.value()
        east = self.east_spin.value()
        south = self.south_spin.value()
        north = self.north_spin.value()
        print(f"[DEBUG] update_map_preview: coords: W={west}, E={east}, S={south}, N={north}")
        
        # Check that coordinates form a valid rectangle
        if east <= west or north <= south:
            self.map_status_label.setText("Map: Invalid coordinates")
            return
        
        # Log the map preview request
        self.log_message(f"Requesting map preview: {west:.4f}°E to {east:.4f}°E, {south:.4f}°N to {north:.4f}°N")
        print("[DEBUG] update_map_preview: after validation and before worker start")
        
        # Do not start a new worker if the previous one is still running
        if self.current_map_worker and self.current_map_worker.isRunning():
            print("[DEBUG] update_map_preview: previous map worker still running, skipping new request")
            return
        
        # Create new map worker with current settings
        self.current_map_worker = MapWorker(
            west, east, south, north, 
            width=800,  # Fixed width for consistent preview quality
            mask=self.mask_checkbox.isChecked()  # Use current mask setting
        )
        
        # Connect worker signals to UI update methods
        self.current_map_worker.map_loaded.connect(self.on_map_loaded)
        self.current_map_worker.map_error.connect(self.on_map_error)
        
        # Update UI to show loading state
        self.map_status_label.setText("Map: Loading...")
        self.refresh_map_btn.setEnabled(False)  # Prevent multiple requests
        print("[DEBUG] update_map_preview: starting map worker")
        self.current_map_worker.start()  # Start the background download
        print("[DEBUG] update_map_preview: map worker started")
    
    def on_map_loaded(self, pixmap):
        """
        Handle successful map loading from the worker thread.
        
        This method is called when the map worker successfully downloads
        and processes the map image. It scales the image to fit the display
        area and updates the UI accordingly.
        
        Args:
            pixmap (QPixmap): The loaded map image
        """
        # Scale the pixmap to fit the label while maintaining aspect ratio
        scaled_pixmap = pixmap.scaled(
            self.map_widget.size(), 
            Qt.AspectRatioMode.KeepAspectRatio, 
            Qt.TransformationMode.SmoothTransformation
        )
        
        # Display the scaled image and update UI state
        self.map_widget.set_pixmap(scaled_pixmap)
        
        # Set the current bounds in the map widget for coordinate conversion
        west = self.west_spin.value()
        east = self.east_spin.value()
        south = self.south_spin.value()
        north = self.north_spin.value()
        self.map_widget.set_bounds(west, east, south, north)
        
        self.map_status_label.setText("Map: Loaded")
        self.refresh_map_btn.setEnabled(True)  # Re-enable the refresh button
        self.log_message("Map preview updated")
    
    def on_map_error(self, error_msg):
        """
        Handle map loading errors from the worker thread.
        
        This method is called when the map worker encounters an error
        during the download or processing of the map image.
        
        Args:
            error_msg (str): Description of the error that occurred
        """
        # Display error message in the map area
        self.map_widget.set_pixmap(QPixmap()) # Clear any previous image
        self.map_status_label.setText("Map: Error")
        self.refresh_map_btn.setEnabled(True)  # Re-enable the refresh button
        self.log_message(f"Map preview error: {error_msg}")

    def resizeEvent(self, event):
        """
        Handle window resize events to properly scale the map display.
        """
        super().resizeEvent(event)
        # If we have a valid pixmap, rescale it to fit the new size
        if (hasattr(self, 'map_widget') and 
            self.map_widget.pixmap and 
            not self.map_widget.pixmap.isNull()):
            
            pixmap = self.map_widget.pixmap
            scaled_pixmap = pixmap.scaled(
                self.map_widget.size(), 
                Qt.AspectRatioMode.KeepAspectRatio, 
                Qt.TransformationMode.SmoothTransformation
            )
            self.map_widget.set_pixmap(scaled_pixmap)

    def log_message(self, message):
        """Add a message to the log with timestamp"""
        timestamp = datetime.now().strftime("%H:%M:%S")
        log_entry = f"[{timestamp}] {message}"
        self.log_area.append(log_entry)
        # Auto-scroll to bottom
        scrollbar = self.log_area.verticalScrollBar()
        if scrollbar:
            scrollbar.setValue(scrollbar.maximum())

    def clear_log(self):
        """Clear the log area"""
        self.log_area.clear()
        self.log_message("Log cleared")


    def on_tile_checkbox_toggled(self, checked):
        """Enable/disable mosaic checkbox based on tile dataset checkbox"""
        self.mosaic_checkbox.setEnabled(checked)
        if not checked:
            # If tile dataset is unchecked, also uncheck mosaic tiles
            self.mosaic_checkbox.setChecked(False)

    def on_mosaic_checkbox_toggled(self, checked):
        """Enable/disable delete tiles checkbox based on mosaic checkbox"""
        self.delete_tiles_checkbox.setEnabled(checked)
        # Force a visual update and repaint
        self.delete_tiles_checkbox.update()
        self.delete_tiles_checkbox.repaint()
        # Don't automatically check/uncheck - let user control it

    def update_tile_checkbox_availability(self):
        """Enable/disable tile checkbox based on grid size"""
        try:
            west = self.west_spin.value()
            east = self.east_spin.value()
            south = self.south_spin.value()
            north = self.north_spin.value()
            
            # Calculate grid dimensions
            lon_span = east - west
            lat_span = north - south
            
            # Use configured tile size if available, otherwise default to 2.0 degrees
            try:
                tile_size = float(self.tile_size_combo.currentText())
            except Exception:
                tile_size = 2.0
            
            # Enable tiling if grid is larger than tile size in any dimension
            should_enable_tiling = (lon_span > tile_size) or (lat_span > tile_size)
            
            # Update checkbox state
            self.tile_checkbox.setEnabled(should_enable_tiling)
            
            # Automatically check the checkbox if area is larger than tile size
            if should_enable_tiling and not self.tile_checkbox.isChecked():
                self.tile_checkbox.setChecked(True)
                self.log_message(f"Tiling automatically enabled: Grid area ({lon_span:.1f}° x {lat_span:.1f}°) is larger than tile size ({tile_size} degrees)")
            
            # If tiling is disabled and currently checked, uncheck it
            if not should_enable_tiling and self.tile_checkbox.isChecked():
                self.tile_checkbox.setChecked(False)
                self.log_message(f"Tiling disabled: Grid area is smaller than tile size ({tile_size} degrees)")
            
            # Update tooltip based on availability
            if should_enable_tiling:
                self.tile_checkbox.setToolTip(
                    f"Break large dataset ({lon_span:.1f}° x {lat_span:.1f}°) into {tile_size}° tiles with configurable overlap"
                )
            else:
                self.tile_checkbox.setToolTip(
                    f"Tiling not available: Grid area ({lon_span:.1f}° x {lat_span:.1f}°) is smaller than tile size ({tile_size} degrees)"
                )
                
        except Exception as e:
            # If there's an error, enable the checkbox to avoid breaking the UI
            self.tile_checkbox.setEnabled(True)
            print(f"[DEBUG] Error updating tile checkbox availability: {e}")


    def check_large_grid_warning(self):
        """Check if user should be warned about large grids"""
        west = self.west_spin.value()
        east = self.east_spin.value()
        south = self.south_spin.value()
        north = self.north_spin.value()
        
        lon_span = abs(east - west)
        lat_span = abs(north - south)
        
        mres_value = float(self.mres_combo.currentText())
        is_high_res = mres_value <= 200  # Consider 100 or 200 as high resolution
        
        if is_high_res and (lon_span > 3 or lat_span > 3) and not self.tile_checkbox.isChecked():
            reply = QMessageBox.question(
                self, 
                "Large Grid Warning",
                f"Your grid spans {lon_span:.1f}° longitude and {lat_span:.1f}° latitude.\n"
                f"Downloading at {mres_value} meters/pixel resolution may result in very large files.\n\n"
                f"Consider enabling 'Tile Dataset for Download' to break this into smaller tiles.\n\n"
                f"Continue with single download?",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                QMessageBox.StandardButton.No
            )
            if reply == QMessageBox.StandardButton.Yes:
                self.log_message("User chose to continue with single download")
            else:
                self.log_message("User cancelled download")
            return reply == QMessageBox.StandardButton.Yes
        else:
            return True

    def calculate_overlap_from_resolution(self):
        """
        Calculate overlap in degrees based on cell resolution.
        Overlap is 2 grid cells worth of degrees.
        
        Returns:
            float: Overlap in degrees (2 cells worth)
        """
        mres_value = float(self.mres_combo.currentText())
        
        # Use exact overlap values as specified (2 cells worth in degrees)
        overlap_map = {
            800: 0.016,  # 800m grids: 0.016 degrees
            400: 0.008,  # 400m grids: 0.008 degrees
            200: 0.004,  # 200m grids: 0.004 degrees
            100: 0.002   # 100m grids: 0.002 degrees
        }
        
        # Get overlap from map, or calculate if not in map
        if mres_value in overlap_map:
            overlap_degrees = overlap_map[mres_value]
        else:
            # Fallback: calculate from meters (approximate)
            meters_per_degree = 111000.0
            cell_size_degrees = mres_value / meters_per_degree
            overlap_degrees = 2.0 * cell_size_degrees
        
        return overlap_degrees
    
    def generate_tiles(self, west, east, south, north, overlap=None):
        """
        Generate tile boundaries for a large grid.
        
        Args:
            west, east, south, north: Overall bounds for the grid
            overlap: Overlap in degrees (if None, calculated automatically from resolution)
        """
        tiles = []
        
        # Calculate overlap automatically from cell resolution if not provided
        if overlap is None:
            overlap = self.calculate_overlap_from_resolution()
            self.log_message(f"Auto-calculated overlap: {overlap:.6f} degrees (2 cells at {self.mres_combo.currentText()}m resolution)")
        
        # Calculate tile size from UI if available, else default to 2.0 degrees
        try:
            tile_size = float(self.tile_size_combo.currentText())
        except Exception:
            tile_size = 2.0
        
        # Generate longitude tiles (without overlap applied yet)
        lon_tiles = []
        current_lon = west
        iteration = 0
        
        while current_lon < east and iteration < 10:  # Safety limit
            tile_west = current_lon
            tile_east = min(current_lon + tile_size, east)
            
            # Only add tile if it has meaningful size (at least 0.1 degrees)
            if tile_east - tile_west >= 0.1:
                lon_tiles.append((tile_west, tile_east))
                # Move to next tile position (account for overlap)
                current_lon = tile_east - overlap
            else:
                break  # Exit the loop for tiny tiles
            
            iteration += 1
        
        # Generate latitude tiles (without overlap applied yet)
        lat_tiles = []
        current_lat = south
        iteration = 0
        
        while current_lat < north and iteration < 10:  # Safety limit
            tile_south = current_lat
            tile_north = min(current_lat + tile_size, north)
            
            # Only add tile if it has meaningful size (at least 0.1 degrees)
            if tile_north - tile_south >= 0.1:
                lat_tiles.append((tile_south, tile_north))
                # Move to next tile position (account for overlap)
                current_lat = tile_north - overlap
            else:
                break  # Exit the loop for tiny tiles
            
            iteration += 1
        
        # Generate all tile combinations with automatic overlap on all sides
        # Overlap is applied per-tile, respecting geographic boundaries
        for i, (tile_west, tile_east) in enumerate(lon_tiles):
            for j, (tile_south, tile_north) in enumerate(lat_tiles):
                # Start with original tile bounds
                padded_west = tile_west
                padded_east = tile_east
                padded_south = tile_south
                padded_north = tile_north
                
                # Add overlap to each edge, but respect geographic limits
                # West edge: only add overlap if not at -180°
                if padded_west > -180.0:
                    padded_west = max(padded_west - overlap, -180.0)
                
                # East edge: only add overlap if not at 180°
                if padded_east < 180.0:
                    padded_east = min(padded_east + overlap, 180.0)
                
                # South edge: only add overlap if not at -85° (GMRT data limit)
                if padded_south > -85.0:
                    padded_south = max(padded_south - overlap, -85.0)
                
                # North edge: only add overlap if not at 85° (GMRT data limit)
                if padded_north < 85.0:
                    padded_north = min(padded_north + overlap, 85.0)
                
                tiles.append({
                    'west': padded_west,
                    'east': padded_east,
                    'south': padded_south,
                    'north': padded_north,
                    'tile_id': f"tile_{i+1:02d}_{j+1:02d}"
                })
        
        return tiles

    def download_single_grid(self, params, filename, callback=None):
        """Download a single grid file using worker thread"""
        if self.current_worker and self.current_worker.isRunning():
            return False  # Already downloading
        
        # Log the request parameters for debugging
        param_str = ", ".join([f"{k}={v}" for k, v in params.items()])
        self.log_message(f"Making request: {param_str}")
        
        self.current_worker = DownloadWorker(params, filename)
        if callback:
            self.current_worker.finished.connect(callback)
        self.current_worker.start()
        return True

    def download_grid(self):
        # Immediate visual feedback
        self.download_btn.setText("Processing...")
        self.download_btn.setEnabled(False)
        self.status_label.setText("Processing download request...")
        self.status_label.repaint()
        
        # Test log message first
        try:
            self.log_message("=== DOWNLOAD BUTTON CLICKED ===")
        except Exception as e:
            self.status_label.setText(f"Log error: {str(e)}")
            self.download_btn.setText("Download Grid")
            self.download_btn.setEnabled(True)
            return
        
        try:
            self.log_message("Starting download process...")
        except Exception as e:
            self.status_label.setText(f"Log error: {str(e)}")
            self.download_btn.setText("Download Grid")
            self.download_btn.setEnabled(True)
            return
        
        try:
            self.log_message("Button clicked - processing request...")
        except Exception as e:
            self.status_label.setText(f"Log error: {str(e)}")
            self.download_btn.setText("Download Grid")
            self.download_btn.setEnabled(True)
            return
        
        # Get coordinate values
        west = self.west_spin.value()
        east = self.east_spin.value()
        south = self.south_spin.value()
        north = self.north_spin.value()
        
        # Validate coordinate order
        if east <= west:
            self.log_message(f"Error: East ({east}) must be greater than West ({west})")
            self.status_label.setText("Error: East must be greater than West")
            self.download_btn.setText("Download Grid")
            self.download_btn.setEnabled(True)
            QMessageBox.warning(self, "Invalid Coordinates", 
                              f"East coordinate ({east}) must be greater than West coordinate ({west}).\n\n"
                              f"Please correct the coordinates and try again.")
            return
        
        if north <= south:
            self.log_message(f"Error: North ({north}) must be greater than South ({south})")
            self.status_label.setText("Error: North must be greater than South")
            self.download_btn.setText("Download Grid")
            self.download_btn.setEnabled(True)
            QMessageBox.warning(self, "Invalid Coordinates", 
                              f"North coordinate ({north}) must be greater than South coordinate ({south}).\n\n"
                              f"Please correct the coordinates and try again.")
            return
        
        # Validate latitude range (GMRT data limit)
        if south < -85.0 or south > 85.0:
            self.log_message(f"Error: South latitude ({south}) must be between -85° and 85°")
            self.status_label.setText("Error: South latitude out of range")
            self.download_btn.setText("Download Grid")
            self.download_btn.setEnabled(True)
            QMessageBox.warning(self, "Invalid Coordinates", 
                              f"South latitude ({south}°) must be between -85° and 85° (GMRT data limit).\n\n"
                              f"Please correct the coordinates and try again.")
            return
        
        if north < -85.0 or north > 85.0:
            self.log_message(f"Error: North latitude ({north}) must be between -85° and 85°")
            self.status_label.setText("Error: North latitude out of range")
            self.download_btn.setText("Download Grid")
            self.download_btn.setEnabled(True)
            QMessageBox.warning(self, "Invalid Coordinates", 
                              f"North latitude ({north}°) must be between -85° and 85° (GMRT data limit).\n\n"
                              f"Please correct the coordinates and try again.")
            return
        
        self.log_message(f"Grid bounds: {west:.4f}°E to {east:.4f}°E, {south:.4f}°N to {north:.4f}°N")
        self.log_message("Coordinates validated successfully")
        
        # Log the grid download request details
        format_type = self.format_combo.currentText()
        layer_type_display = self.layer_combo.currentText()
        layer_type = self.get_layer_type()
        mres_value = self.mres_combo.currentText()
        
        self.log_message(f"Grid download request:")
        self.log_message(f"  Format: {format_type}")
        self.log_message(f"  Layer: {layer_type_display}")
        self.log_message(f"  Cell Resolution: {mres_value} meters/pixel")
        
        # Check for large grid warning
        try:
            if not self.check_large_grid_warning():
                self.log_message("Download cancelled by user")
                self.download_btn.setText("Download Grid")
                self.download_btn.setEnabled(True)
                self.status_label.setText("")
                return
        except Exception as e:
            self.status_label.setText(f"Warning check error: {str(e)}")
            self.download_btn.setText("Download Grid")
            self.download_btn.setEnabled(True)
            return

        format_type = self.format_combo.currentText()
        layer_type = self.get_layer_type()
        
        # Map format to file extension
        format_extensions = {
            "geotiff": ".tif",
            "netcdf": ".nc", 
            "esriascii": ".asc",
            "coards": ".grd"
        }
        
        file_ext = format_extensions.get(format_type, f".{format_type}")
        current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        layer_type_display = self.layer_combo.currentText()
        self.log_message(f"Format: {format_type}, Layer: {layer_type_display}, Cell Resolution: {self.mres_combo.currentText()} meters/pixel")
        self.log_message("Parameters validated successfully")

        if self.tile_checkbox.isChecked():
            # Tiled download
            # Overlap is now calculated automatically from cell resolution
            tiles = self.generate_tiles(west, east, south, north)
            self.log_message(f"Generated {len(tiles)} tiles")
            
            if len(tiles) == 1:
                # Only one tile needed, download as single file
                self.log_message("Single tile detected, downloading as regular file")
                self.download_single_tile(tiles[0], format_type, layer_type, file_ext, current_time)
            else:
                # Multiple tiles - ask for directory
                dir_path = QFileDialog.getExistingDirectory(self, "Select Directory for Tiled Files", self.last_download_dir)
                if not dir_path:
                    self.log_message("Directory selection cancelled")
                    self.download_btn.setText("Download Grid")
                    self.download_btn.setEnabled(True)
                    self.status_label.setText("")
                    return
                self.save_last_download_dir(dir_path)
                self.log_message(f"Selected directory: {dir_path}")
                
                self.status_label.setText(f"Downloading {len(tiles)} tiles...")
                self.status_label.repaint()
                
                # Start sequential tile downloads
                self.tiles_to_download = tiles
                self.current_tile_index = 0
                self.download_dir = dir_path
                self.format_type = format_type
                self.layer_type = layer_type
                self.file_ext = file_ext
                self.current_time = current_time
                self.success_count = 0
                self.downloaded_tile_files = []  # Reset tile files list
                
                self.download_btn.setEnabled(False)  # Disable button during download
                self.download_next_tile()
        else:
            # Single download
            self.log_message("Single file download")
            self.log_message("Opening directory selection dialog...")
            dir_path = QFileDialog.getExistingDirectory(self, "Select Directory for Grid File", self.last_download_dir)
            if not dir_path:
                self.log_message("Directory selection cancelled")
                self.download_btn.setText("Download Grid")
                self.download_btn.setEnabled(True)
                self.status_label.setText("")
                return
            self.save_last_download_dir(dir_path)
            self.log_message(f"Selected directory: {dir_path}")
            
            suggested_name = f"gmrt_{layer_type}_{current_time}{file_ext}"
            file_name = os.path.join(dir_path, suggested_name)

            params = {
                "west": west,
                "east": east,
                "south": south,
                "north": north,
                "format": format_type,
                "layer": layer_type,
                "mresolution": float(self.mres_combo.currentText())
            }

            self.status_label.setText("Downloading...")
            self.status_label.repaint()
            
            self.download_btn.setEnabled(False)  # Disable button during download
            self.download_single_grid(params, file_name, self.on_single_download_finished)
            
    def download_next_tile(self):
        """Download the next tile in the sequence"""
        if self.current_tile_index >= len(self.tiles_to_download):
            return  # All tiles downloaded

        tile = self.tiles_to_download[self.current_tile_index]
        # Use lower left coordinates instead of date/time
        ll_lon = f"{tile['west']:.3f}".replace('-', 'm').replace('.', 'p')
        ll_lat = f"{tile['south']:.3f}".replace('-', 'm').replace('.', 'p')
        tile_filename = f"gmrt_{self.layer_type}_{ll_lon}_{ll_lat}_{self.current_tile_index + 1:03d}{self.file_ext}"
        tile_path = os.path.join(self.download_dir, tile_filename)
        
        self.log_message(f"Starting download of tile {self.current_tile_index + 1}/{len(self.tiles_to_download)}: {tile_filename}")
        
        params = {
            "west": tile['west'],
            "east": tile['east'],
            "south": tile['south'],
            "north": tile['north'],
            "format": self.format_type,
            "layer": self.layer_type,
            "mresolution": float(self.mres_combo.currentText())
        }
        
        self.status_label.setText(f"Downloading tile {self.current_tile_index + 1}/{len(self.tiles_to_download)}: {tile_filename}")
        self.status_label.repaint()
        
        self.download_single_grid(params, tile_path, self.on_tile_download_finished)

    def split_grid_file(self, filename, format_type):
        """
        Split a grid file into topography (>=0) and bathymetry (<0) files for supported formats.
        Appends _topo and _bathy to the base filename.
        """
        import os
        base, ext = os.path.splitext(filename)
        topo_file = base + '_topo' + ext
        bathy_file = base + '_bathy' + ext
        try:
            deleted_files = []
            if format_type == 'geotiff' and rasterio is not None:
                with rasterio.open(filename) as src:
                    data = src.read(1)
                    profile = src.profile
                    topo_data = np.where(data >= 0, data, np.nan)
                    bathy_data = np.where(data < 0, data, np.nan)
                    profile.update(dtype=rasterio.float32, nodata=np.nan)
                    with rasterio.open(topo_file, 'w', **profile) as dst:
                        dst.write(topo_data.astype(np.float32), 1)
                    with rasterio.open(bathy_file, 'w', **profile) as dst:
                        dst.write(bathy_data.astype(np.float32), 1)
                for f, arr, label in [(topo_file, topo_data, 'topo'), (bathy_file, bathy_data, 'bathy')]:
                    if np.isnan(arr).all() or os.path.getsize(f) == 0:
                        try:
                            os.remove(f)
                            deleted_files.append(f)
                            self.log_message(f"Deleted empty {label} GeoTIFF: {os.path.basename(f)}")
                        except Exception as e:
                            self.log_message(f"Failed to delete empty {label} GeoTIFF: {os.path.basename(f)}: {e}")
                self.log_message(f"Split GeoTIFF: {os.path.basename(topo_file)}, {os.path.basename(bathy_file)}")
            elif format_type in ['esriascii', 'coards']:
                with open(filename, 'r') as f:
                    header = []
                    data_lines = []
                    for line in f:
                        if any(x in line.lower() for x in ['ncols', 'nrows', 'xllcorner', 'yllcorner', 'cellsize', 'nodata_value']):
                            header.append(line)
                        else:
                            data_lines.append(line)
                data = np.genfromtxt(data_lines)
                topo_data = np.where(data >= 0, data, np.nan)
                bathy_data = np.where(data < 0, data, np.nan)
                with open(topo_file, 'w') as f:
                    for h in header:
                        f.write(h)
                    np.savetxt(f, topo_data, fmt='%.6f')
                with open(bathy_file, 'w') as f:
                    for h in header:
                        f.write(h)
                    np.savetxt(f, bathy_data, fmt='%.6f')
                for f, arr, label in [(topo_file, topo_data, 'topo'), (bathy_file, bathy_data, 'bathy')]:
                    if np.isnan(arr).all() or os.path.getsize(f) == 0:
                        try:
                            os.remove(f)
                            deleted_files.append(f)
                            self.log_message(f"Deleted empty {label} ASCII grid: {os.path.basename(f)}")
                        except Exception as e:
                            self.log_message(f"Failed to delete empty {label} ASCII grid: {os.path.basename(f)}: {e}")
                self.log_message(f"Split ASCII grid: {os.path.basename(topo_file)}, {os.path.basename(bathy_file)}")
            elif format_type == 'netcdf' and netCDF4 is not None:
                ds = netCDF4.Dataset(filename, 'r')
                data_var = None
                for var in ds.variables:
                    if ds.variables[var].ndim == 2:
                        data_var = var
                        break
                if data_var is None:
                    self.log_message("No 2D variable found in NetCDF file for splitting.")
                    ds.close()
                    return
                data = ds.variables[data_var][:]
                topo_data = np.where(data >= 0, data, np.nan)
                bathy_data = np.where(data < 0, data, np.nan)
                topo_ds = netCDF4.Dataset(topo_file, 'w')
                for dim in ds.dimensions:
                    topo_ds.createDimension(dim, len(ds.dimensions[dim]))
                for var in ds.variables:
                    if var != data_var:
                        topo_ds.createVariable(var, ds.variables[var].datatype, ds.variables[var].dimensions)
                        topo_ds.variables[var][:] = ds.variables[var][:]
                topo_var = topo_ds.createVariable(data_var, 'f4', ds.variables[data_var].dimensions, fill_value=np.nan)
                topo_var[:] = topo_data
                topo_ds.close()
                bathy_ds = netCDF4.Dataset(bathy_file, 'w')
                for dim in ds.dimensions:
                    bathy_ds.createDimension(dim, len(ds.dimensions[dim]))
                for var in ds.variables:
                    if var != data_var:
                        bathy_ds.createVariable(var, ds.variables[var].datatype, ds.variables[var].dimensions)
                        bathy_ds.variables[var][:] = ds.variables[var][:]
                bathy_var = bathy_ds.createVariable(data_var, 'f4', ds.variables[data_var].dimensions, fill_value=np.nan)
                bathy_var[:] = bathy_data
                bathy_ds.close()
                ds.close()
                for f, arr, label in [(topo_file, topo_data, 'topo'), (bathy_file, bathy_data, 'bathy')]:
                    try:
                        with netCDF4.Dataset(f, 'r') as check_ds:
                            check_var = check_ds.variables[data_var][:]
                            if np.isnan(check_var).all() or os.path.getsize(f) == 0:
                                os.remove(f)
                                deleted_files.append(f)
                                self.log_message(f"Deleted empty {label} NetCDF: {os.path.basename(f)}")
                    except Exception as e:
                        self.log_message(f"Failed to check/delete empty {label} NetCDF: {os.path.basename(f)}: {e}")
                self.log_message(f"Split NetCDF: {os.path.basename(topo_file)}, {os.path.basename(bathy_file)}")
            elif format_type == 'netcdf':
                self.log_message("NetCDF4 not installed, cannot split NetCDF file.")
            else:
                self.log_message(f"Split not supported for format: {format_type}")
            # After splitting and empty checks, delete the original if split is enabled
            if os.path.exists(filename):
                try:
                    os.remove(filename)
                    self.log_message(f"Deleted original unsplit file: {os.path.basename(filename)} after splitting.")
                except Exception as e:
                    self.log_message(f"Failed to delete original unsplit file: {os.path.basename(filename)}: {e}")
        except Exception as e:
            self.log_message(f"Error splitting grid: {str(e)}")

    def on_single_download_finished(self, success, result):
        """Callback for single download completion"""
        self.download_btn.setEnabled(True)
        self.download_btn.setText("Download Grid")
        if success:
            self.log_message(f"Download completed successfully: {os.path.basename(result)}")
            self.status_label.setText(f"Download complete: {result}")
            # Split if requested
            format_type = self.format_combo.currentText()
            if self.split_checkbox.isChecked():
                self.split_grid_file(result, format_type)
            QMessageBox.information(self, "Success", f"Grid downloaded to:\n{result}")
        else:
            self.log_message(f"Download failed: {result}")
            self.status_label.setText("Download failed.")
            QMessageBox.critical(self, "Error", f"Failed to download grid:\n{result}")

    def on_tile_download_finished(self, success, result):
        """Callback for tile download completion"""
        try:
            if success:
                self.success_count += 1
                self.downloaded_tile_files.append(result)  # Track downloaded file
                self.log_message(f"Tile {self.current_tile_index + 1} downloaded successfully: {os.path.basename(result)}")
                # Don't split individual tiles if mosaicking is enabled
                if not self.mosaic_checkbox.isChecked():
                    format_type = self.format_type
                    if self.split_checkbox.isChecked():
                        self.split_grid_file(result, format_type)
            else:
                self.log_message(f"Tile {self.current_tile_index + 1} failed: {result}")
            
            # Move to next tile
            self.current_tile_index += 1
            
            # Schedule next download with a 2-second delay
            if self.current_tile_index < len(self.tiles_to_download):
                self.log_message(f"Waiting 2 seconds before downloading tile {self.current_tile_index + 1}")
                # Use QTimer.singleShot to ensure this runs on the main thread
                from PyQt6.QtCore import QTimer
                QTimer.singleShot(2000, self.download_next_tile)
            else:
                # All tiles downloaded, check if mosaicking is enabled
                self.log_message(f"All {len(self.tiles_to_download)} tiles completed. Success count: {self.success_count}")
                self.log_message(f"Downloaded tile files: {self.downloaded_tile_files}")
                
                if self.mosaic_checkbox.isChecked() and self.downloaded_tile_files:
                    self.log_message("All tiles downloaded, starting mosaicking process...")
                    # Use QTimer.singleShot to ensure this runs on the main thread
                    from PyQt6.QtCore import QTimer
                    QTimer.singleShot(3000, self.start_mosaicking)
                else:
                    # No mosaicking, finish normally
                    QTimer.singleShot(100, self.finish_tile_download)
                    
        except Exception as e:
            import traceback
            self.log_message(f"ERROR in on_tile_download_finished: {str(e)}")
            self.log_message(f"Traceback: {traceback.format_exc()}")
            from PyQt6.QtCore import QTimer
            QTimer.singleShot(100, self.finish_tile_download)
    
    def start_mosaicking(self):
        """Start the mosaicking process after a delay"""
        try:
            print("[DEBUG] === START_MOSAICKING CALLED ===")
            self.log_message("=== START_MOSAICKING CALLED ===")
            self.status_label.setText("Mosaicking tiles into single GeoTIFF...")
            self.status_label.repaint()
            print("[DEBUG] Status label updated")
            self.log_message("Starting mosaicking worker thread...")

            # Use rasterio for mosaicking
            print("[DEBUG] Using rasterio for mosaicking")
            self.log_message("Using rasterio for mosaicking")

            # Create and start the mosaic worker thread
            print(f"[DEBUG] Creating MosaicWorker with {len(self.downloaded_tile_files)} tiles")
            self.mosaic_worker = MosaicWorker(
                self.downloaded_tile_files,
                self.download_dir,
                self.layer_type,
                self.west_spin,
                self.south_spin,
                self.east_spin,
                self.north_spin,
                self.delete_tiles_checkbox,
                self.split_checkbox,
                self.format_type
            )
            print("[DEBUG] Connecting signals...")
            self.mosaic_worker.progress.connect(self.on_mosaic_progress)
            self.mosaic_worker.finished.connect(self.on_mosaic_finished)
            print("[DEBUG] Starting worker thread...")
            self.mosaic_worker.start()
            print("[DEBUG] Worker thread started successfully")
            
        except Exception as e:
            import traceback
            print(f"[DEBUG] ERROR in start_mosaicking: {str(e)}")
            print(f"[DEBUG] Traceback: {traceback.format_exc()}")
            self.log_message(f"ERROR in start_mosaicking: {str(e)}")
            self.log_message(f"Traceback: {traceback.format_exc()}")
            self.finish_tile_download()
    
    def on_mosaic_progress(self, message):
        """Handle progress updates from mosaic worker"""
        print(f"[DEBUG] Mosaic progress: {message}")
        self.log_message(f"Mosaic: {message}")
        self.status_label.setText(f"Mosaicking: {message}")
        self.status_label.repaint()
    
    def on_mosaic_finished(self, success, result):
        """Handle completion of mosaic worker"""
        try:
            print(f"[DEBUG] Mosaic worker finished - Success: {success}, Result: {result}")
            self.log_message(f"Mosaic worker finished - Success: {success}, Result: {result}")
            
            if success:
                print("[DEBUG] Mosaicking completed successfully")
                self.log_message("✓ Mosaicking completed successfully")
                self.status_label.setText("Mosaicking completed successfully")
                
                # Get the mosaic path from the worker
                if hasattr(self.mosaic_worker, 'mosaic_path') and self.mosaic_worker.mosaic_path:
                    mosaic_path = self.mosaic_worker.mosaic_path
                    print(f"[DEBUG] Mosaic file created: {mosaic_path}")
                    
                    # Delete individual tile files if enabled
                    if self.delete_tiles_checkbox.isChecked():
                        deleted_count = 0
                        for tile_file in self.downloaded_tile_files:
                            try:
                                if os.path.exists(tile_file):
                                    os.remove(tile_file)
                                    deleted_count += 1
                                    print(f"[DEBUG] Deleted tile: {os.path.basename(tile_file)}")
                            except Exception as e:
                                print(f"[DEBUG] Error deleting tile {tile_file}: {str(e)}")
                        
                        print(f"[DEBUG] Deleted {deleted_count} individual tile files")
                    
                    # Apply splitting to the mosaicked file if requested
                    if self.split_checkbox.isChecked():
                        print("[DEBUG] Applying split to mosaicked file...")
                        self.log_message("Applying split to mosaicked file...")
                        self.split_grid_file(mosaic_path, 'geotiff')
                    
                    # Finish with success message
                    self.download_btn.setEnabled(True)
                    self.download_btn.setText("Download Grid")
                    self.log_message(f"Mosaicking completed successfully: {os.path.basename(mosaic_path)}")
                    self.status_label.setText(f"Mosaic complete: {os.path.basename(mosaic_path)} in {self.download_dir}")
                    QMessageBox.information(self, "Success", f"Mosaicked {len(self.downloaded_tile_files)} tiles into:\n{mosaic_path}")
                else:
                    print("[DEBUG] ERROR: No mosaic path found")
                    self.log_message("ERROR: No mosaic path found")
                    self.finish_tile_download()
            else:
                print(f"[DEBUG] Mosaicking failed: {result}")
                self.log_message(f"✗ Mosaicking failed: {result}")
                self.status_label.setText("Mosaicking failed")
                self.download_btn.setText("Download Grid")
                self.download_btn.setEnabled(True)
                QMessageBox.critical(self, "Mosaicking Error", f"Mosaicking failed:\n{result}")
                self.finish_tile_download()
                
        except Exception as e:
            import traceback
            print(f"[DEBUG] ERROR in on_mosaic_finished: {str(e)}")
            print(f"[DEBUG] Traceback: {traceback.format_exc()}")
            self.log_message(f"ERROR in on_mosaic_finished: {str(e)}")
            self.log_message(f"Traceback: {traceback.format_exc()}")
            self.finish_tile_download()
    
    

    def finish_tile_download(self):
        """Finish the tile download process without mosaicking"""
        self.download_btn.setEnabled(True)
        self.download_btn.setText("Download Grid")
        self.log_message(f"All tiles completed: {self.success_count}/{len(self.tiles_to_download)} successful")
        self.status_label.setText(f"Download complete: {self.success_count}/{len(self.tiles_to_download)} tiles in {self.download_dir}")
        QMessageBox.information(self, "Success", f"Downloaded {self.success_count}/{len(self.tiles_to_download)} tiles to:\n{self.download_dir}")

    def validate_bathymetry_data(self, data):
        """
        Validate bathymetry/topography data and set unrealistic values to NaN.
        
        Args:
            data (numpy.ndarray): Input bathymetry/topography data
            
        Returns:
            numpy.ndarray: Data with unrealistic values set to NaN
        """
        import numpy as np
        
        # Define realistic bathymetry/topography limits for Earth
        # These are conservative limits based on known Earth topography
        max_elevation = 9000.0    # Mount Everest is ~8848m, use 9000m as upper limit
        min_elevation = -12000.0  # Mariana Trench is ~11000m, use 12000m as lower limit
        
        # Create a copy to avoid modifying the original
        validated_data = data.copy()
        
        # Set unrealistic values to nodata value
        nodata_value = -99999
        
        # Count original nodata values
        original_nodata = (validated_data == nodata_value).sum()
        
        # Values above max_elevation (too high)
        too_high = validated_data > max_elevation
        # Values below min_elevation (too deep)
        too_deep = validated_data < min_elevation
        
        # Set unrealistic values to nodata value (-99999)
        validated_data[too_high] = nodata_value
        validated_data[too_deep] = nodata_value
        
        # Count new nodata values
        new_nodata = (validated_data == nodata_value).sum()
        invalid_count = new_nodata - original_nodata
        
        if invalid_count > 0:
            self.log_message(f"Set {invalid_count} unrealistic values to nodata (-99999) (outside {min_elevation}m to {max_elevation}m range)")
        else:
            self.log_message("All bathymetry/topography values are within realistic range")
        
        return validated_data

    def trim_tile_overlap(self, dataset, overlap_degrees):
        """Trim overlap data from a tile based on the specified overlap amount"""
        if overlap_degrees <= 0:
            return dataset  # No trimming needed
        
        # Get the tile bounds
        bounds = dataset.bounds
        west, south, east, north = bounds
        
        # Calculate the trim amount in degrees
        trim_amount = overlap_degrees / 2  # Trim half the overlap from each side
        
        # Calculate trimmed bounds
        trimmed_west = west + trim_amount
        trimmed_east = east - trim_amount
        trimmed_south = south + trim_amount
        trimmed_north = north - trim_amount
        
        # Ensure we don't trim more than the tile size
        tile_width = east - west
        tile_height = north - south
        
        if trim_amount * 2 >= tile_width or trim_amount * 2 >= tile_height:
            # If overlap is too large, return the original dataset
            return dataset
        
        # Create a window for the trimmed area
        from rasterio.windows import from_bounds
        window = from_bounds(trimmed_west, trimmed_south, trimmed_east, trimmed_north, dataset.transform)
        
        # Read the trimmed data
        trimmed_data = dataset.read(window=window)
        
        # Create a new transform for the trimmed data
        from rasterio.transform import from_bounds
        trimmed_transform = from_bounds(trimmed_west, trimmed_south, trimmed_east, trimmed_north, 
                                      window.width, window.height)
        
        # Create a new dataset-like object with trimmed data
        class TrimmedDataset:
            def __init__(self, data, transform, profile):
                self.data = data
                self.transform = transform
                self.profile = profile
                self.bounds = (trimmed_west, trimmed_south, trimmed_east, trimmed_north)
                self.width = data.shape[2]
                self.height = data.shape[1]
            
            def read(self, **kwargs):
                return self.data
            
            def close(self):
                pass
        
        trimmed_profile = dataset.profile.copy()
        trimmed_profile.update({
            'height': window.height,
            'width': window.width,
            'transform': trimmed_transform
        })
        
        return TrimmedDataset(trimmed_data, trimmed_transform, trimmed_profile)


    def mosaic_tiles(self):
        """Mosaic all downloaded tiles into a single GeoTIFF file using rasterio only"""
        try:
            self.log_message("=== STARTING MOSAICKING PROCESS ===")
            self.status_label.setText("Starting mosaicking process...")
            self.status_label.repaint()
            
            if not self.downloaded_tile_files:
                self.log_message("No tiles to mosaic")
                self.finish_tile_download()
                return
            
            self.log_message(f"Mosaicking {len(self.downloaded_tile_files)} tiles...")
            self.log_message(f"Tile files: {self.downloaded_tile_files}")
            
            # Check if all tile files exist and are readable
            missing_files = []
            for f in self.downloaded_tile_files:
                if not os.path.exists(f):
                    missing_files.append(f)
                else:
                    # Check if file is readable
                    try:
                        with open(f, 'rb') as test_file:
                            test_file.read(1)
                        self.log_message(f"File {os.path.basename(f)} exists and is readable ({os.path.getsize(f)} bytes)")
                    except Exception as e:
                        self.log_message(f"File {os.path.basename(f)} exists but is not readable: {str(e)}")
                        missing_files.append(f)
            
            if missing_files:
                self.log_message(f"ERROR: Missing or unreadable tile files: {missing_files}")
                self.finish_tile_download()
                return
            
            if rasterio is None:
                self.log_message("Error: rasterio not available for mosaicking")
                self.finish_tile_download()
                return
            
            # Create output filename for mosaicked file
            current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
            mosaic_filename = f"gmrt_{self.layer_type}_mosaic_{current_time}.tif"
            mosaic_path = os.path.join(self.download_dir, mosaic_filename)
            self.log_message(f"Output mosaic path: {mosaic_path}")
            
            # Read all tile files and get their bounds
            datasets = []
            bounds_list = []
            cell_sizes = []
            
            self.log_message("Opening tile files with rasterio...")
            for i, tile_file in enumerate(self.downloaded_tile_files):
                if os.path.exists(tile_file):
                    try:
                        self.log_message(f"Opening tile {i+1}/{len(self.downloaded_tile_files)}: {os.path.basename(tile_file)}")
                        dataset = rasterio.open(tile_file)
                        datasets.append(dataset)
                        bounds_list.append(dataset.bounds)
                        
                        # Get cell size information
                        transform = dataset.transform
                        cell_size_x = abs(transform[0])  # Pixel width
                        cell_size_y = abs(transform[4])  # Pixel height
                        cell_sizes.append((cell_size_x, cell_size_y))
                        
                        self.log_message(f"Successfully loaded tile: {os.path.basename(tile_file)} (cell size: {cell_size_x:.2f}m x {cell_size_y:.2f}m)")
                    except Exception as e:
                        import traceback
                        self.log_message(f"Error reading tile {tile_file}: {str(e)}")
                        self.log_message(f"Traceback: {traceback.format_exc()}")
                        continue
                else:
                    self.log_message(f"Tile file does not exist: {tile_file}")
            
            if not datasets:
                self.log_message("No valid tiles found for mosaicking")
                self.finish_tile_download()
                return
            
            # Get the original requested bounds
            original_bounds = (self.west_spin.value(), self.south_spin.value(), 
                             self.east_spin.value(), self.north_spin.value())
            
            # Use rasterio for mosaicking
            self.log_message("Using rasterio for mosaicking...")
            try:
                self._mosaic_with_rasterio_fallback(datasets, mosaic_path, original_bounds)
            except Exception as mosaic_error:
                self.log_message(f"Rasterio mosaicking failed: {str(mosaic_error)}")
                self.log_message("Mosaicking failed completely")
                self.finish_tile_download()
                return
            
            self.log_message(f"Mosaicked file created: {os.path.basename(mosaic_path)}")
            
            # Close all datasets and clean up temporary files
            for dataset in datasets:
                try:
                    # Clean up temporary files if they exist (from NetCDF conversion)
                    temp_file = None
                    if hasattr(dataset, '_temp_file'):
                        temp_file = dataset._temp_file
                    dataset.close()
                    # Delete temporary file after closing
                    if temp_file and os.path.exists(temp_file):
                        try:
                            os.remove(temp_file)
                            print(f"[DEBUG] Cleaned up temporary file: {temp_file}")
                        except Exception as e:
                            print(f"[DEBUG] Could not delete temporary file {temp_file}: {e}")
                except Exception as e:
                    print(f"[DEBUG] Error closing dataset: {e}")
                    pass  # Ignore errors when closing
            
            # Delete individual tile files if enabled
            if self.delete_tiles_checkbox.isChecked():
                deleted_count = 0
                for tile_file in self.downloaded_tile_files:
                    try:
                        if os.path.exists(tile_file):
                            os.remove(tile_file)
                            deleted_count += 1
                            self.log_message(f"Deleted tile: {os.path.basename(tile_file)}")
                    except Exception as e:
                        self.log_message(f"Error deleting tile {tile_file}: {str(e)}")
                
                self.log_message(f"Deleted {deleted_count} individual tile files")
            else:
                self.log_message(f"Individual tile files preserved: {len(self.downloaded_tile_files)} tiles")
            
            # Apply splitting to the mosaicked file if requested
            if self.split_checkbox.isChecked():
                self.log_message("Applying split to mosaicked file...")
                self.split_grid_file(mosaic_path, 'geotiff')
            
            # Finish with success message
            self.download_btn.setEnabled(True)
            self.log_message(f"Mosaicking completed successfully: {os.path.basename(mosaic_path)}")
            self.status_label.setText(f"Mosaic complete: {os.path.basename(mosaic_path)} in {self.download_dir}")
            QMessageBox.information(self, "Success", f"Mosaicked {len(self.downloaded_tile_files)} tiles into:\n{mosaic_path}")
            
        except Exception as e:
            import traceback
            self.log_message(f"CRITICAL ERROR in mosaicking: {str(e)}")
            self.log_message(f"Traceback: {traceback.format_exc()}")
            self.log_message("Mosaicking failed - check error details above")
            
            # Try to provide helpful error message
            error_msg = str(e)
            if "Permission denied" in error_msg:
                error_msg += "\n\nPossible solutions:\n- Close any programs that might be using the files\n- Check file permissions\n- Try running as administrator"
            elif "No space left" in error_msg:
                error_msg += "\n\nPossible solutions:\n- Free up disk space\n- Choose a different output directory"
            elif "rasterio" in error_msg.lower():
                error_msg += "\n\nPossible solutions:\n- Install rasterio: pip install rasterio\n- Check if the tile files are corrupted"
            
            try:
                self.finish_tile_download()
                QMessageBox.critical(self, "Mosaicking Error", f"Mosaicking failed:\n{error_msg}")
            except Exception as ui_error:
                self.log_message(f"Could not show error dialog: {ui_error}")
                self.log_message("Application will continue but mosaicking failed")

    def download_single_tile(self, tile, format_type, layer_type, file_ext, current_time):
        """Download a single tile as a regular file"""
        dir_path = QFileDialog.getExistingDirectory(self, "Select Directory for Grid File", self.last_download_dir)
        if not dir_path:
            return
        self.save_last_download_dir(dir_path)
        
        suggested_name = f"gmrt_{layer_type}_{current_time}{file_ext}"
        file_name = os.path.join(dir_path, suggested_name)

        params = {
            "west": tile['west'],
            "east": tile['east'],
            "south": tile['south'],
            "north": tile['north'],
            "format": format_type,
            "layer": layer_type,
            "mresolution": float(self.mres_combo.currentText())
        }

        self.status_label.setText("Downloading...")
        self.status_label.repaint()
        
        self.download_btn.setEnabled(False)  # Disable button during download
        self.download_single_grid(params, file_name, self.on_single_download_finished)

    def on_rectangle_selected(self, bounds):
        print(f"[DEBUG] on_rectangle_selected called with bounds: {bounds}")
        # Block signals to prevent multiple update_map_preview calls
        self.west_spin.blockSignals(True)
        self.east_spin.blockSignals(True)
        self.south_spin.blockSignals(True)
        self.north_spin.blockSignals(True)

        self.west_spin.setValue(bounds[0])
        self.east_spin.setValue(bounds[1])
        self.south_spin.setValue(bounds[2])
        self.north_spin.setValue(bounds[3])

        self.west_spin.blockSignals(False)
        self.east_spin.blockSignals(False)
        self.south_spin.blockSignals(False)
        self.north_spin.blockSignals(False)

        print("[DEBUG] on_rectangle_selected: set spin boxes")
        self.map_widget.clear_selection()
        print("[DEBUG] on_rectangle_selected: cleared selection")
        self.draw_rect_btn.setChecked(False)
        print("[DEBUG] on_rectangle_selected: draw_rect_btn unchecked")
        self.update_map_preview() # Only one call now
        print("[DEBUG] on_rectangle_selected: called update_map_preview")
        self.log_message(f"Rectangle selected: {bounds}")

    def zoom_to_default(self):
        """
        Zoom the map to the starting map defaults.
        """
        # Block signals to prevent multiple map updates
        self.west_spin.blockSignals(True)
        self.east_spin.blockSignals(True)
        self.south_spin.blockSignals(True)
        self.north_spin.blockSignals(True)
        self.west_spin.setValue(-180.0)
        self.east_spin.setValue(180.0)
        self.south_spin.setValue(-85.0)
        self.north_spin.setValue(85.0)
        self.west_spin.blockSignals(False)
        self.east_spin.blockSignals(False)
        self.south_spin.blockSignals(False)
        self.north_spin.blockSignals(False)
        self.update_map_preview()
        self.log_message("Zoomed to starting map defaults")

class MapWidget(QWidget):
    """
    Custom widget for displaying map with rectangle drawing functionality.
    
    This widget extends QWidget to provide interactive map display with
    the ability to draw rectangles to select geographic areas.
    """

    # Signal to communicate selection bounds back to main window
    bounds_selected = pyqtSignal(tuple)  # (west, east, south, north)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.pixmap = None
        self.drawing_mode = False
        self.selection_rect = None
        self.drag_start = None
        self.current_bounds = None  # (west, east, south, north)
        self.image_rect = None  # Store the actual image rectangle within the widget
        self.cursor_pos = None  # Store current cursor position for coordinate display
        
        # Set up the widget
        self.setMinimumSize(600,400)
        self.setStyleSheet("QWidget { border: 1px solid gray; background-color: #f0f0f0; }")
        
        # Enable mouse tracking for coordinate display
        self.setMouseTracking(True)
        
    def set_pixmap(self, pixmap):
        """Set the map image and scale it to fit the widget"""
        self.pixmap = pixmap
        self.update()
        
    def set_bounds(self, west, east, south, north):
        """Set the geographic bounds for coordinate conversion"""
        self.current_bounds = (west, east, south, north)
        
    def enable_drawing(self, enabled):
        """Enable or disable drawing mode"""
        self.drawing_mode = enabled
        if not enabled:
            self.selection_rect = None
            self.drag_start = None
        self.update()
        
    def clear_selection(self):
        """Clear the current selection rectangle"""
        self.selection_rect = None
        self.drag_start = None
        self.update()
        
    def get_selection_bounds(self):
        if self.selection_rect is None or self.current_bounds is None or self.map_content_rect is None:
            return None
        west, east, south, north = self.current_bounds
        image_sel_rect = QRect(
            self.selection_rect.x() - self.map_content_rect.x(),
            self.selection_rect.y() - self.map_content_rect.y(),
            self.selection_rect.width(),
            self.selection_rect.height()
        )
        if (image_sel_rect.x() < 0 or image_sel_rect.y() < 0 or
            image_sel_rect.right() > self.map_content_rect.width() or
            image_sel_rect.bottom() > self.map_content_rect.height()):
            return None
        img_w = self.map_content_rect.width()
        img_h = self.map_content_rect.height()
        lon_per_pixel = (east - west) / (img_w - 1)
        sel_west = west + (image_sel_rect.x() * lon_per_pixel)
        sel_east = west + (image_sel_rect.x() + image_sel_rect.width()) * lon_per_pixel
        import math
        def lat_to_merc(lat):
            return math.log(math.tan(math.pi / 4 + math.radians(lat) / 2))
        def merc_to_lat(merc):
            return math.degrees(2 * math.atan(math.exp(merc)) - math.pi / 2)
        merc_north = lat_to_merc(north)
        merc_south = lat_to_merc(south)
        y0 = image_sel_rect.y()
        y1 = image_sel_rect.y() + image_sel_rect.height()
        merc_y0 = merc_north - (y0 / (img_h - 1)) * (merc_north - merc_south)
        merc_y1 = merc_north - (y1 / (img_h - 1)) * (merc_north - merc_south)
        sel_north = merc_to_lat(merc_y0)
        sel_south = merc_to_lat(merc_y1)
        return (sel_west, sel_east, sel_south, sel_north)
        
    def get_cursor_coordinates(self, mouse_pos):
        if self.current_bounds is None or self.map_content_rect is None:
            return None
        if not self.map_content_rect.contains(mouse_pos):
            return None
        west, east, south, north = self.current_bounds
        image_x = mouse_pos.x() - self.map_content_rect.x()
        image_y = mouse_pos.y() - self.map_content_rect.y()
        img_w = self.map_content_rect.width()
        img_h = self.map_content_rect.height()
        lon_per_pixel = (east - west) / (img_w - 1)
        longitude = west + (image_x * lon_per_pixel)
        def lat_to_merc(lat):
            import math
            return math.log(math.tan(math.pi / 4 + math.radians(lat) / 2))
        def merc_to_lat(merc):
            import math
            return math.degrees(2 * math.atan(math.exp(merc)) - math.pi / 2)
        merc_north = lat_to_merc(north)
        merc_south = lat_to_merc(south)
        merc_y = merc_north - (image_y / (img_h - 1)) * (merc_north - merc_south)
        latitude = merc_to_lat(merc_y)
        return (longitude, latitude)
        
    def mousePressEvent(self, event):
        if not self.drawing_mode or self.pixmap is None or self.map_content_rect is None:
            return
        if self.map_content_rect.contains(event.pos()):
            self.drag_start = event.pos()
            self.selection_rect = QRect(self.drag_start, self.drag_start)
            self.update()
        
    def mouseMoveEvent(self, event):
        self.cursor_pos = event.pos()
        self.update()
        if not self.drawing_mode or self.drag_start is None or self.pixmap is None or self.map_content_rect is None:
            return
        if not self.map_content_rect.contains(event.pos()):
            return
        constrained_pos = QPoint(
            max(self.map_content_rect.left(), min(event.pos().x(), self.map_content_rect.right())),
            max(self.map_content_rect.top(), min(event.pos().y(), self.map_content_rect.bottom()))
        )
        self.selection_rect = QRect(self.drag_start, constrained_pos).normalized()
        self.update()
        
    def mouseReleaseEvent(self, event):
        if not self.drawing_mode or self.drag_start is None or self.pixmap is None or self.map_content_rect is None:
            return
        if event.button() == Qt.MouseButton.LeftButton:
            constrained_pos = QPoint(
                max(self.map_content_rect.left(), min(event.pos().x(), self.map_content_rect.right())),
                max(self.map_content_rect.top(), min(event.pos().y(), self.map_content_rect.bottom()))
            )
            self.selection_rect = QRect(self.drag_start, constrained_pos).normalized()
            self.drag_start = None
            self.update()
            bounds = self.get_selection_bounds()
            if bounds:
                self.bounds_selected.emit(bounds)
                
    def paintEvent(self, event):
        """Custom paint event to draw the map and selection rectangle"""
        painter = QPainter(self)
        if self.pixmap and not self.pixmap.isNull():
            scaled_pixmap = self.pixmap.scaled(
                self.size(), 
                Qt.AspectRatioMode.KeepAspectRatio, 
                Qt.TransformationMode.SmoothTransformation
            )
            x = (self.width() - scaled_pixmap.width()) //2
            y = (self.height() - scaled_pixmap.height()) // 2
            painter.drawPixmap(x, y, scaled_pixmap)
            self.image_rect = QRect(x, y, scaled_pixmap.width(), scaled_pixmap.height())
            self.map_content_rect = QRect(
                self.image_rect.left() + 33,
                self.image_rect.top() + 13,
                self.image_rect.width() - 66,
                self.image_rect.height() - 26
            )
        else:
            painter.drawText(self.rect(), Qt.AlignmentFlag.AlignCenter, 
                        "Click 'Refresh Map' to load preview")
            self.image_rect = None
            self.map_content_rect = None
        if self.drawing_mode and self.selection_rect is not None:
            pen = QPen(QColor(255, 0, 0), 2)
            painter.setPen(pen)
            painter.drawRect(self.selection_rect)
            painter.setBrush(QColor(255, 0, 0, 50))
            painter.drawRect(self.selection_rect)
        if self.cursor_pos is not None and self.map_content_rect is not None and self.map_content_rect.contains(self.cursor_pos):
            coords = self.get_cursor_coordinates(self.cursor_pos)
            if coords:
                longitude, latitude = coords
                coord_text = f"Lon: {longitude:.4f}°\nLat: {latitude:.4f}°"
                font = painter.font()
                font.setPointSize(9)
                painter.setFont(font)
                text_rect = painter.boundingRect(QRect(), Qt.TextFlag.TextDontClip, coord_text)
                text_rect.setWidth(text_rect.width() + 10)
                text_rect.setHeight(text_rect.height() + 6)
                text_rect.moveTo(self.map_content_rect.left() + 10, self.map_content_rect.bottom() - text_rect.height() - 10)
                painter.setBrush(QColor(0, 0, 0, 180))
                painter.setPen(QPen(QColor(255, 255, 255)))
                painter.drawRect(text_rect)
                painter.setPen(QColor(255, 255, 255))
                painter.drawText(text_rect, Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignTop, coord_text)

def signal_handler(signum, frame):
    """Handle signals to catch crashes"""
    print(f"Received signal {signum}")
    import traceback
    traceback.print_stack(frame)
    sys.exit(1)

if __name__ == "__main__":
    try:
        import signal
        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)
        
        print("Starting GMRT Bathymetry Grid Downloader...")
        
        # Use rasterio for mosaicking
        print("[DEBUG] Using rasterio for mosaicking")
        
        app = QApplication(sys.argv)
        print("QApplication created successfully")
        
        win = GMRTGrabber()
        print("GMRTGrabber window created successfully")
        
        win.show()
        print("Window shown successfully")
        
        # Force the window to be visible and raised
        win.raise_()
        win.activateWindow()
        
        print("Starting application event loop...")
        sys.exit(app.exec())
    except Exception as e:
        print(f"Error starting application: {str(e)}")
        import traceback
        traceback.print_exc()
        input("Press Enter to exit...")
