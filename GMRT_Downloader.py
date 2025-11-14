"""
GMRT Bathymetry Grid Downloader

A PyQt6-based GUI application for downloading bathymetry data from the 
Global Multi-Resolution Topography (GMRT) synthesis.

Credit: Ryan, W.B.F., S.M. Carbotte, J.O. Coplan, S. O'Hara, A. Melkonian, R. Arko, R.A. Weissel, V. Ferrini, A. Goodwillie, F. Nitsche, J. Bonczkowski, and R. Zemsky (2009), Global Multi-Resolution Topography synthesis, Geochem. Geophys. Geosyst., 10, Q03014, doi: 10.1029/2008GC002332

Features:
- Interactive map preview using GMRT ImageServer
- Downloads in GeoTIFF format
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
import tempfile
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
                            
                            # Check if the dataset has valid geographic metadata
                            # If CRS is None or bounds are not geographic, we need to convert it
                            try:
                                src_bounds = dataset.bounds
                                src_crs = dataset.crs
                                needs_conversion = False
                                
                                if src_crs is None:
                                    print(f"[DEBUG] Rasterio source has no CRS, will use netCDF4 fallback")
                                    needs_conversion = True
                                elif not src_crs.is_geographic:
                                    print(f"[DEBUG] Rasterio source CRS is projected ({src_crs}), will use netCDF4 fallback")
                                    needs_conversion = True
                                elif src_crs.to_epsg() != 4326:
                                    print(f"[DEBUG] Rasterio source CRS is not EPSG:4326 ({src_crs}), will use netCDF4 fallback")
                                    needs_conversion = True
                                elif (src_bounds[0] < -200 or src_bounds[2] > 200 or
                                      src_bounds[1] < -100 or src_bounds[3] > 100):
                                    print(f"[DEBUG] Rasterio source has bad bounds: {src_bounds}, will use netCDF4 fallback")
                                    needs_conversion = True
                                
                                if needs_conversion:
                                    print(f"[DEBUG] Closing rasterio source and falling back to netCDF4 approach")
                                    dataset.close()
                                    # Fall through to netCDF4 conversion
                                    break
                                else:
                                    # Dataset is good, return it
                                    return dataset
                            except Exception as check_error:
                                print(f"[DEBUG] Error checking dataset metadata: {check_error}, will use netCDF4 fallback")
                                dataset.close()
                                break
                        except Exception as e3:
                            print(f"[DEBUG] Failed with pattern {gdal_path}: {e3}")
                            continue
                    
                    # If all GDAL patterns fail or produce invalid metadata, try using netCDF4 to read and create a temporary GeoTIFF
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
        
        # Check if this is GMRT's special 1D array format
        # GMRT NetCDF files have: z (1D array), dimension (grid size), x_range, y_range, spacing
        if 'z' in nc.variables and 'dimension' in nc.variables and 'x_range' in nc.variables and 'y_range' in nc.variables:
            print(f"[DEBUG] Detected GMRT 1D array format in _open_netcdf_with_netcdf4")
            try:
                # Get grid dimensions
                dimension = nc.variables['dimension'][:]
                if len(dimension) >= 2:
                    ncols = int(dimension[0])
                    nrows = int(dimension[1])
                else:
                    nc.close()
                    raise Exception(f"Invalid dimension array: {dimension}")
                
                print(f"[DEBUG] Grid dimensions: {ncols} x {nrows}")
                
                # Get coordinate ranges
                x_range = nc.variables['x_range'][:]
                y_range = nc.variables['y_range'][:]
                lon_min = float(x_range[0])
                lon_max = float(x_range[1])
                lat_min = float(y_range[0])
                lat_max = float(y_range[1])
                
                print(f"[DEBUG] Coordinate ranges: lon={lon_min:.6f} to {lon_max:.6f}, lat={lat_min:.6f} to {lat_max:.6f}")
                
                # Get the 1D data array
                z_data = nc.variables['z'][:]
                print(f"[DEBUG] Z data shape: {z_data.shape}, expected size: {nrows * ncols}")
                
                # Reshape to 2D (row-major, C order)
                if z_data.size == nrows * ncols:
                    data = z_data.reshape((nrows, ncols), order='C')
                    print(f"[DEBUG] Reshaped data to 2D: {data.shape}")
                else:
                    nc.close()
                    raise Exception(f"Data size mismatch: got {z_data.size}, expected {nrows * ncols}")
                
                # Create transform
                from rasterio.transform import from_bounds
                from rasterio.crs import CRS
                transform = from_bounds(lon_min, lat_min, lon_max, lat_max, ncols, nrows)
                print(f"[DEBUG] Transform: {transform}")
                
                # Use EPSG:4326 for GMRT data
                crs = CRS.from_epsg(4326)
                
                # Handle nodata
                nodata_value = None
                if 'z_range' in nc.variables:
                    nodata_value = -99999  # Common nodata value
                
                nc.close()
            except Exception as e:
                nc.close()
                raise Exception(f"Error processing GMRT 1D format: {e}")
        else:
            # Standard NetCDF format - find the data variable (usually 2D or 3D)
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
            
            # Handle nodata
            nodata_value = None
            if hasattr(nc.variables[data_var], '_FillValue'):
                nodata_value = nc.variables[data_var]._FillValue
            
            nc.close()
        
        # Create a temporary GeoTIFF
        temp_tiff = tempfile.NamedTemporaryFile(suffix='.tif', delete=False)
        temp_tiff.close()
        
        # Only use tiled format if dimensions are compatible
        use_tiled = (data.shape[0] % 16 == 0) and (data.shape[1] % 16 == 0)
        
        # Write data to temporary GeoTIFF
        profile = {
            'driver': 'GTiff',
            'height': data.shape[0],
            'width': data.shape[1],
            'count': 1,
            'dtype': data.dtype,
            'crs': crs,
            'transform': transform,
            'compress': 'lzw',
            'tiled': use_tiled
        }
        
        if nodata_value is not None:
            profile['nodata'] = nodata_value
        
        # If tiled, set appropriate block sizes
        if use_tiled:
            profile['blockxsize'] = min(512, data.shape[1])
            profile['blockysize'] = min(512, data.shape[0])
            profile['blockxsize'] = (profile['blockxsize'] // 16) * 16
            profile['blockysize'] = (profile['blockysize'] // 16) * 16
        
        with rasterio.open(temp_tiff.name, 'w', **profile) as dst:
            dst.write(data, 1)
        
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
                crs = dataset.crs
                print(f"[DEBUG] Tile {i+1} transform: {transform}")
                print(f"[DEBUG] Tile {i+1} bounds: {bounds}")
                print(f"[DEBUG] Tile {i+1} CRS: {crs}")
                print(f"[DEBUG] Tile {i+1} CRS is geographic: {crs.is_geographic if crs else 'None'}")
                print(f"[DEBUG] Tile {i+1} size: {dataset.width} x {dataset.height}")
                
                # INVESTIGATE CELL REGISTRATION: Check if bounds represent pixel centers or edges
                if i == 0:  # Only check first tile for registration type
                    from rasterio.transform import xy
                    # Get the first pixel center coordinates from the transform
                    first_pixel_center_lon, first_pixel_center_lat = xy(transform, 0, 0)
                    # Get the last pixel center coordinates
                    last_pixel_center_lon, last_pixel_center_lat = xy(transform, dataset.height - 1, dataset.width - 1)
                    
                    # Calculate cell sizes
                    cell_size_x = abs(transform[0])
                    cell_size_y = abs(transform[4])
                    
                    print(f"[DEBUG] === CELL REGISTRATION INVESTIGATION ===")
                    print(f"[DEBUG] Tile bounds (from dataset.bounds): west={bounds[0]:.9f}, east={bounds[2]:.9f}, south={bounds[1]:.9f}, north={bounds[3]:.9f}")
                    print(f"[DEBUG] First pixel center (from transform): lon={first_pixel_center_lon:.9f}, lat={first_pixel_center_lat:.9f}")
                    print(f"[DEBUG] Last pixel center (from transform): lon={last_pixel_center_lon:.9f}, lat={last_pixel_center_lat:.9f}")
                    print(f"[DEBUG] Cell size: {cell_size_x:.9f} x {cell_size_y:.9f} degrees")
                    
                    # Check if bounds represent pixel edges or centers
                    # If pixel-edge registration: first_pixel_center = bounds[0] + cell_size_x/2
                    # If pixel-center registration: first_pixel_center = bounds[0]
                    expected_edge_registration_lon = bounds[0] + cell_size_x / 2.0
                    expected_edge_registration_lat = bounds[3] - cell_size_y / 2.0  # North is top, so subtract
                    
                    lon_diff_edge = abs(first_pixel_center_lon - expected_edge_registration_lon)
                    lat_diff_edge = abs(first_pixel_center_lat - expected_edge_registration_lat)
                    lon_diff_center = abs(first_pixel_center_lon - bounds[0])
                    lat_diff_center = abs(first_pixel_center_lat - bounds[3])
                    
                    print(f"[DEBUG] If EDGE registration: first pixel center should be at lon={expected_edge_registration_lon:.9f}, lat={expected_edge_registration_lat:.9f}")
                    print(f"[DEBUG] Difference from edge registration: lon={lon_diff_edge:.9f}, lat={lat_diff_edge:.9f}")
                    print(f"[DEBUG] Difference from center registration: lon={lon_diff_center:.9f}, lat={lat_diff_center:.9f}")
                    
                    if lon_diff_edge < 1e-6 and lat_diff_edge < 1e-6:
                        print(f"[DEBUG] >>> TILE USES PIXEL-EDGE REGISTRATION (bounds = pixel edges) <<<")
                    elif lon_diff_center < 1e-6 and lat_diff_center < 1e-6:
                        print(f"[DEBUG] >>> TILE USES PIXEL-CENTER REGISTRATION (bounds = pixel centers) <<<")
                    else:
                        print(f"[DEBUG] >>> WARNING: Registration type unclear! Differences don't match either pattern <<<")
                    print(f"[DEBUG] ==========================================")
                
                # Validate bounds are reasonable for geographic CRS
                if crs and crs.is_geographic:
                    # Geographic bounds should be in degrees: lon -180 to 180, lat -90 to 90
                    if bounds[0] < -200 or bounds[2] > 200 or bounds[1] < -100 or bounds[3] > 100:
                        print(f"[DEBUG] WARNING: Tile {i+1} bounds look like they might be in wrong units!")
                        print(f"[DEBUG] Expected degrees (lon: -180 to 180, lat: -90 to 90)")
                        print(f"[DEBUG] Got: lon {bounds[0]:.2f} to {bounds[2]:.2f}, lat {bounds[1]:.2f} to {bounds[3]:.2f}")
                
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
                print(f"[DEBUG] === TILE {i+1} CELL SIZE: {cell_size_x:.9f}° x {cell_size_y:.9f}° (lon x lat) ===")
            
            # Find the finest resolution (smallest cell size)
            min_cell_x = min(cell_size[0] for cell_size in cell_sizes)
            min_cell_y = min(cell_size[1] for cell_size in cell_sizes)
            print(f"[DEBUG] Finest resolution: {min_cell_x:.9f} x {min_cell_y:.9f}")
            
            # Calculate union of tile bounds
            tile_min_x = min(bounds[0] for bounds in bounds_list)
            tile_min_y = min(bounds[1] for bounds in bounds_list)
            tile_max_x = max(bounds[2] for bounds in bounds_list)
            tile_max_y = max(bounds[3] for bounds in bounds_list)
            print(f"[DEBUG] Tile union bounds: min_x={tile_min_x:.6f}, max_x={tile_max_x:.6f}, min_y={tile_min_y:.6f}, max_y={tile_max_y:.6f}")
            
            # Get the first tile's transform to understand the grid alignment
            # We need to align the output transform to the same grid as the tiles
            first_tile_transform = datasets[0].transform
            from rasterio.transform import xy
            first_tile_first_pixel_lon, first_tile_first_pixel_lat = xy(first_tile_transform, 0, 0)
            
            print(f"[DEBUG] First tile's first pixel center: lon={first_tile_first_pixel_lon:.9f}, lat={first_tile_first_pixel_lat:.9f}")
            
            # Use original requested bounds for the output extent
            min_x, min_y, max_x, max_y = original_bounds
            print(f"[DEBUG] Using original requested bounds: min_x={min_x:.6f}, max_x={max_x:.6f}, min_y={min_y:.6f}, max_y={max_y:.6f}")
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
            
            # ALIGN OUTPUT TRANSFORM TO TILE GRID
            # Instead of using from_bounds which creates a transform aligned to the bounds,
            # we need to create a transform that aligns with the tile grid.
            # Calculate how many pixels from the first tile's first pixel to the requested west edge
            # The first tile's first pixel center is at first_tile_first_pixel_lon
            # We want the output's first pixel to align with the tile grid
            
            # Calculate the offset from the first tile's first pixel to the requested west edge
            # Then snap to the nearest pixel center on the tile grid
            offset_from_first_pixel = min_x - first_tile_first_pixel_lon
            pixels_offset = round(offset_from_first_pixel / min_cell_x)
            aligned_west = first_tile_first_pixel_lon + pixels_offset * min_cell_x
            
            # Do the same for latitude (north edge)
            offset_from_first_pixel_lat = max_y - first_tile_first_pixel_lat
            pixels_offset_lat = round(offset_from_first_pixel_lat / min_cell_y)
            aligned_north = first_tile_first_pixel_lat + pixels_offset_lat * min_cell_y
            
            # Calculate aligned east and south based on width/height
            aligned_east = aligned_west + width * min_cell_x
            aligned_south = aligned_north - height * min_cell_y
            
            print(f"[DEBUG] === OUTPUT TRANSFORM ALIGNMENT ===")
            print(f"[DEBUG] Requested bounds: west={min_x:.9f}, east={max_x:.9f}, south={min_y:.9f}, north={max_y:.9f}")
            print(f"[DEBUG] Aligned bounds: west={aligned_west:.9f}, east={aligned_east:.9f}, south={aligned_south:.9f}, north={aligned_north:.9f}")
            print(f"[DEBUG] Alignment offset: lon={aligned_west - min_x:.9f}, lat={aligned_north - max_y:.9f}")
            print(f"[DEBUG] ===================================")
            
            # Create the output transform using aligned bounds (aligned to tile grid)
            output_transform = rasterio.transform.from_bounds(aligned_west, aligned_south, aligned_east, aligned_north, width, height)
            
            # INVESTIGATE OUTPUT TRANSFORM REGISTRATION
            from rasterio.transform import xy
            output_first_pixel_center_lon, output_first_pixel_center_lat = xy(output_transform, 0, 0)
            output_last_pixel_center_lon, output_last_pixel_center_lat = xy(output_transform, height - 1, width - 1)
            
            print(f"[DEBUG] === OUTPUT TRANSFORM REGISTRATION INVESTIGATION ===")
            print(f"[DEBUG] Output bounds (from from_bounds): west={min_x:.9f}, east={max_x:.9f}, south={min_y:.9f}, north={max_y:.9f}")
            print(f"[DEBUG] Output first pixel center (from transform): lon={output_first_pixel_center_lon:.9f}, lat={output_first_pixel_center_lat:.9f}")
            print(f"[DEBUG] Output last pixel center (from transform): lon={output_last_pixel_center_lon:.9f}, lat={output_last_pixel_center_lat:.9f}")
            print(f"[DEBUG] Output cell size: {min_cell_x:.9f} x {min_cell_y:.9f} degrees")
            
            # from_bounds creates edge-registered transform
            expected_output_edge_lon = min_x + min_cell_x / 2.0
            expected_output_edge_lat = max_y - min_cell_y / 2.0
            
            output_lon_diff = abs(output_first_pixel_center_lon - expected_output_edge_lon)
            output_lat_diff = abs(output_first_pixel_center_lat - expected_output_edge_lat)
            
            print(f"[DEBUG] Expected first pixel center (edge registration): lon={expected_output_edge_lon:.9f}, lat={expected_output_edge_lat:.9f}")
            print(f"[DEBUG] Difference: lon={output_lon_diff:.9f}, lat={output_lat_diff:.9f}")
            if output_lon_diff < 1e-6 and output_lat_diff < 1e-6:
                print(f"[DEBUG] >>> OUTPUT USES PIXEL-EDGE REGISTRATION (as expected from from_bounds) <<<")
            else:
                print(f"[DEBUG] >>> WARNING: Output transform doesn't match expected edge registration! <<<")
            print(f"[DEBUG] =====================================================")
            
            # Initialize the output array with nodata values
            output_array = np.full((height, width), -99999, dtype=np.float32)
            
            # Process each dataset using rasterio.warp.reproject for proper alignment
            print("[DEBUG] Processing tiles with rasterio.warp.reproject for proper alignment...")
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
                src_crs = dataset.crs
                src_bounds = dataset.bounds
                
                print(f"[DEBUG] Tile {i+1} bounds: {src_bounds}")
                print(f"[DEBUG] Tile {i+1} data shape: {data.shape}")
                print(f"[DEBUG] Tile {i+1} transform: {src_transform}")
                
                # Use rasterio.warp.reproject to properly align and resample the tile
                # This handles all coordinate transformations correctly
                from rasterio.warp import reproject, Resampling
                
                # Create a temporary array for this tile's contribution to the output
                tile_output = np.full((height, width), -99999, dtype=np.float32)
                
                # Reproject the tile data to the output grid
                reproject(
                    source=data,
                    destination=tile_output,
                    src_transform=src_transform,
                    src_crs=src_crs,
                    dst_transform=output_transform,
                    dst_crs=datasets[0].crs,  # Use CRS from first dataset
                    resampling=Resampling.nearest,
                    src_nodata=-99999,
                    dst_nodata=-99999
                )
                
                # Merge data using maximum (shallowest) values
                valid_mask = (tile_output != -99999) & ~np.isnan(tile_output) & ~np.isinf(tile_output)
                
                if np.any(valid_mask):
                    # Only update where source data is valid and shallower (more positive/less negative)
                    update_mask = valid_mask & (
                        (output_array == -99999) | 
                        (tile_output > output_array)
                    )
                    output_array[update_mask] = tile_output[update_mask]
                    
                    valid_count = np.sum(valid_mask)
                    print(f"[DEBUG] Tile {i+1}: Updated {valid_count} valid pixels")
                else:
                    print(f"[DEBUG] Tile {i+1}: No valid data to merge")
            
            # Final data validation
            print("[DEBUG] Performing final data validation...")
            output_array = self.validate_bathymetry_data(output_array)
            
            # No need to crop since we already used original_bounds for the output transform
            # The output array is already aligned to the requested bounds
            
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


def convert_netcdf_to_geotiff(netcdf_path, output_path):
    """
    Convert NetCDF file to GeoTIFF format.
    
    Args:
        netcdf_path (str): Path to input NetCDF file
        output_path (str): Path to output GeoTIFF file
        
    Returns:
        bool: True if successful, False otherwise
    """
    if rasterio is None:
        return False
    
    try:
        # Try multiple approaches to open NetCDF with rasterio
        src = None
        
        # Approach 1: Try opening directly
        try:
            src = rasterio.open(netcdf_path)
            print(f"[DEBUG] Successfully opened NetCDF with direct rasterio.open")
        except Exception as e1:
            print(f"[DEBUG] Direct rasterio.open failed: {e1}")
            pass
        
        # Approach 2: Try with NETCDF driver
        if src is None:
            try:
                src = rasterio.open(netcdf_path, driver='NETCDF')
                print(f"[DEBUG] Successfully opened NetCDF with NETCDF driver")
            except Exception as e2:
                print(f"[DEBUG] NETCDF driver failed: {e2}")
                pass
        
        # Approach 3: Try with GDAL virtual dataset patterns
        if src is None:
            gdal_patterns = [
                f'NETCDF:"{netcdf_path}":z',
                f'NETCDF:"{netcdf_path}":elevation',
                f'NETCDF:"{netcdf_path}":topo',
                f'NETCDF:"{netcdf_path}":bathy',
                f'NETCDF:"{netcdf_path}"',
            ]
            for pattern in gdal_patterns:
                try:
                    src = rasterio.open(pattern)
                    print(f"[DEBUG] Successfully opened NetCDF with pattern: {pattern}")
                    break
                except Exception as e3:
                    print(f"[DEBUG] Pattern {pattern} failed: {e3}")
                    continue
        
        # Check if rasterio source has valid bounds/CRS before using it
        if src is not None:
            try:
                src_bounds = src.bounds
                src_crs = src.crs
                # Check if we need to fallback due to bad bounds/CRS
                needs_fallback = False
                if src_crs is None:
                    print(f"[DEBUG] Rasterio source has no CRS, will use netCDF4 fallback")
                    needs_fallback = True
                elif not src_crs.is_geographic:
                    print(f"[DEBUG] Rasterio source CRS is projected ({src_crs}), will use netCDF4 fallback")
                    needs_fallback = True
                elif src_crs.to_epsg() != 4326:
                    print(f"[DEBUG] Rasterio source CRS is not EPSG:4326 ({src_crs}), will use netCDF4 fallback")
                    needs_fallback = True
                elif (src_bounds[0] < -200 or src_bounds[2] > 200 or 
                      src_bounds[1] < -100 or src_bounds[3] > 100):
                    print(f"[DEBUG] Rasterio source has bad bounds: {src_bounds}, will use netCDF4 fallback")
                    needs_fallback = True
                
                if needs_fallback:
                    print(f"[DEBUG] Closing rasterio source and falling back to netCDF4 approach")
                    src.close()
                    src = None
            except Exception as e:
                print(f"[DEBUG] Error checking rasterio source bounds/CRS: {e}, will use netCDF4 fallback")
                # If we can't check, close and fallback
                if src:
                    src.close()
                src = None
        
        # Approach 4: Try with netCDF4 library and convert directly to GeoTIFF
        if src is None and netCDF4 is not None:
            try:
                print(f"[DEBUG] Attempting netCDF4-based conversion")
                # Open with netCDF4 and find the data variable
                with netCDF4.Dataset(netcdf_path, 'r') as nc:
                    # Debug: Print all variables
                    print(f"[DEBUG] NetCDF variables: {list(nc.variables.keys())}")
                    print(f"[DEBUG] NetCDF dimensions: {list(nc.dimensions.keys())}")
                    
                    # Check if this is GMRT's special 1D array format
                    # GMRT NetCDF files have: z (1D array), dimension (grid size), x_range, y_range, spacing
                    if 'z' in nc.variables and 'dimension' in nc.variables and 'x_range' in nc.variables and 'y_range' in nc.variables:
                        print(f"[DEBUG] Detected GMRT 1D array format")
                        try:
                            # Get grid dimensions
                            dimension = nc.variables['dimension'][:]
                            if len(dimension) >= 2:
                                ncols = int(dimension[0])
                                nrows = int(dimension[1])
                            else:
                                print(f"[DEBUG] Invalid dimension array: {dimension}")
                                return False
                            
                            print(f"[DEBUG] Grid dimensions: {ncols} x {nrows}")
                            
                            # Get coordinate ranges
                            x_range = nc.variables['x_range'][:]
                            y_range = nc.variables['y_range'][:]
                            lon_min = float(x_range[0])
                            lon_max = float(x_range[1])
                            lat_min = float(y_range[0])
                            lat_max = float(y_range[1])
                            
                            print(f"[DEBUG] Coordinate ranges: lon={lon_min:.6f} to {lon_max:.6f}, lat={lat_min:.6f} to {lat_max:.6f}")
                            
                            # Get the 1D data array
                            z_data = nc.variables['z'][:]
                            print(f"[DEBUG] Z data shape: {z_data.shape}, expected size: {nrows * ncols}")
                            
                            # Reshape to 2D (note: GMRT may store row-major or column-major)
                            # Try both orientations
                            if z_data.size == nrows * ncols:
                                # Reshape to 2D - try row-major first (C order)
                                data = z_data.reshape((nrows, ncols), order='C')
                                print(f"[DEBUG] Reshaped data to 2D: {data.shape}")
                            else:
                                print(f"[DEBUG] Data size mismatch: got {z_data.size}, expected {nrows * ncols}")
                                return False
                            
                            # Create transform
                            from rasterio.transform import from_bounds
                            transform = from_bounds(lon_min, lat_min, lon_max, lat_max, ncols, nrows)
                            print(f"[DEBUG] Transform: {transform}")
                            
                            # Only use tiled format if dimensions are compatible
                            use_tiled = (nrows % 16 == 0) and (ncols % 16 == 0)
                            
                            # Create profile
                            profile = {
                                'driver': 'GTiff',
                                'height': nrows,
                                'width': ncols,
                                'count': 1,
                                'dtype': data.dtype,
                                'crs': 'EPSG:4326',
                                'transform': transform,
                                'compress': 'lzw',
                                'tiled': use_tiled
                            }
                            
                            # If tiled, set appropriate block sizes
                            if use_tiled:
                                profile['blockxsize'] = min(512, ncols)
                                profile['blockysize'] = min(512, nrows)
                                profile['blockxsize'] = (profile['blockxsize'] // 16) * 16
                                profile['blockysize'] = (profile['blockysize'] // 16) * 16
                            
                            # Handle nodata - check z_range or use default
                            if 'z_range' in nc.variables:
                                z_range = nc.variables['z_range'][:]
                                # Often nodata is indicated by a specific value in z_range
                                # For now, use a common nodata value
                                profile['nodata'] = -99999
                            else:
                                profile['nodata'] = -99999
                            
                            # Write to GeoTIFF
                            print(f"[DEBUG] Writing GeoTIFF to: {output_path}")
                            with rasterio.open(output_path, 'w', **profile) as dst:
                                dst.write(data, 1)
                            
                            print(f"[DEBUG] Successfully converted GMRT NetCDF to GeoTIFF")
                            return True
                        except Exception as e:
                            print(f"[DEBUG] Error processing GMRT 1D format: {e}")
                            import traceback
                            print(f"[DEBUG] Traceback: {traceback.format_exc()}")
                            return False
                    
                    # Find the 2D data variable (standard NetCDF format)
                    data_var = None
                    for var_name in nc.variables:
                        var = nc.variables[var_name]
                        if len(var.dimensions) >= 2:
                            data_var = var_name
                            print(f"[DEBUG] Found 2D data variable: {data_var}")
                            print(f"[DEBUG] Variable dimensions: {var.dimensions}")
                            break
                    
                    if data_var is None:
                        print(f"[DEBUG] No 2D variable found in NetCDF file")
                        print(f"[DEBUG] All variables: {[(name, len(nc.variables[name].dimensions)) for name in nc.variables]}")
                        return False
                    
                    # Get the data
                    data = nc.variables[data_var][:]
                    print(f"[DEBUG] Data shape: {data.shape}, dtype: {data.dtype}")
                    if len(data.shape) == 3:
                        data = data[0, :, :]
                        print(f"[DEBUG] Reduced 3D to 2D, new shape: {data.shape}")
                    
                    # Get coordinates
                    dims = nc.variables[data_var].dimensions
                    print(f"[DEBUG] Data variable dimensions: {dims}")
                    lat_var = None
                    lon_var = None
                    
                    # Try to find coordinate variables
                    for dim in dims:
                        if dim in nc.variables:
                            var = nc.variables[dim]
                            if hasattr(var, 'standard_name'):
                                std_name = var.standard_name.lower()
                                print(f"[DEBUG] Dimension {dim} has standard_name: {std_name}")
                                if 'lat' in std_name:
                                    lat_var = dim
                                elif 'lon' in std_name:
                                    lon_var = dim
                    
                    # Fallback to common names - check dimension names first
                    if lat_var is None:
                        # Check if dimension names themselves are coordinates
                        for dim in dims:
                            if dim.lower() in ['lat', 'latitude', 'y']:
                                if dim in nc.variables:
                                    lat_var = dim
                                    break
                        # If not found, check all variables
                        if lat_var is None:
                            for name in ['lat', 'latitude', 'y']:
                                if name in nc.variables:
                                    lat_var = name
                                    break
                    
                    if lon_var is None:
                        for dim in dims:
                            if dim.lower() in ['lon', 'longitude', 'x']:
                                if dim in nc.variables:
                                    lon_var = dim
                                    break
                        if lon_var is None:
                            for name in ['lon', 'longitude', 'x']:
                                if name in nc.variables:
                                    lon_var = name
                                    break
                    
                    print(f"[DEBUG] Found lat_var: {lat_var}, lon_var: {lon_var}")
                    
                    if lat_var is None or lon_var is None:
                        print(f"[DEBUG] Could not find coordinate variables")
                        print(f"[DEBUG] Available variables: {list(nc.variables.keys())}")
                        print(f"[DEBUG] Data dimensions: {dims}")
                        return False
                    
                    lats = nc.variables[lat_var][:]
                    lons = nc.variables[lon_var][:]
                    print(f"[DEBUG] Lats shape: {lats.shape}, Lons shape: {lons.shape}")
                    print(f"[DEBUG] Lat range: {float(lats.min())} to {float(lats.max())}")
                    print(f"[DEBUG] Lon range: {float(lons.min())} to {float(lons.max())}")
                    
                    # Calculate transform
                    nrows, ncols = data.shape
                    lat_min = float(lats.min())
                    lat_max = float(lats.max())
                    lon_min = float(lons.min())
                    lon_max = float(lons.max())
                    
                    print(f"[DEBUG] Data shape: {nrows} rows x {ncols} cols")
                    print(f"[DEBUG] Bounds: lon={lon_min:.6f} to {lon_max:.6f}, lat={lat_min:.6f} to {lat_max:.6f}")
                    
                    # Create transform
                    from rasterio.transform import from_bounds
                    transform = from_bounds(lon_min, lat_min, lon_max, lat_max, ncols, nrows)
                    print(f"[DEBUG] Transform: {transform}")
                    
                    # Only use tiled format if dimensions are compatible
                    # Tiled TIFF requires block dimensions to be multiples of 16
                    use_tiled = (nrows % 16 == 0) and (ncols % 16 == 0)
                    
                    # Create profile
                    profile = {
                        'driver': 'GTiff',
                        'height': nrows,
                        'width': ncols,
                        'count': 1,
                        'dtype': data.dtype,
                        'crs': 'EPSG:4326',
                        'transform': transform,
                        'compress': 'lzw',
                        'tiled': use_tiled
                    }
                    
                    # If tiled, set appropriate block sizes
                    if use_tiled:
                        profile['blockxsize'] = min(512, ncols)
                        profile['blockysize'] = min(512, nrows)
                        # Ensure block sizes are multiples of 16
                        profile['blockxsize'] = (profile['blockxsize'] // 16) * 16
                        profile['blockysize'] = (profile['blockysize'] // 16) * 16
                    
                    # Handle nodata
                    if hasattr(nc.variables[data_var], '_FillValue'):
                        nodata = float(nc.variables[data_var]._FillValue)
                        profile['nodata'] = nodata
                        print(f"[DEBUG] Using nodata value: {nodata}")
                    
                    # Write to GeoTIFF
                    print(f"[DEBUG] Writing GeoTIFF to: {output_path}")
                    with rasterio.open(output_path, 'w', **profile) as dst:
                        dst.write(data, 1)
                    
                    print(f"[DEBUG] Successfully converted NetCDF to GeoTIFF")
                    return True
            except Exception as e:
                print(f"[DEBUG] Error converting NetCDF with netCDF4: {e}")
                import traceback
                print(f"[DEBUG] Traceback: {traceback.format_exc()}")
                return False
        
        # If we got a rasterio source, use it
        if src is not None:
            try:
                # Read the data
                data = src.read(1)
                
                # Get bounds and CRS from source
                src_bounds = src.bounds
                src_crs = src.crs
                src_transform = src.transform
                
                print(f"[DEBUG] Source bounds: {src_bounds}")
                print(f"[DEBUG] Source CRS: {src_crs}")
                print(f"[DEBUG] Source transform: {src_transform}")
                
                # Ensure CRS is EPSG:4326 (WGS84) for geographic data
                # If source CRS is None or different, use EPSG:4326
                output_crs = rasterio.crs.CRS.from_epsg(4326)
                needs_reproject = False
                
                if src_crs is None:
                    print(f"[DEBUG] Source CRS is None, using EPSG:4326 and recalculating transform")
                    needs_reproject = True
                elif not src_crs.is_geographic:
                    print(f"[DEBUG] Warning: Source CRS is projected ({src_crs}), using EPSG:4326 and recalculating transform")
                    needs_reproject = True
                elif src_crs.to_epsg() != 4326:
                    print(f"[DEBUG] Source CRS is geographic but not EPSG:4326 ({src_crs}), using EPSG:4326")
                    needs_reproject = True
                else:
                    # Check if bounds look reasonable for geographic coordinates
                    # Geographic bounds should be: lon -180 to 180, lat -90 to 90
                    if (src_bounds[0] < -200 or src_bounds[2] > 200 or 
                        src_bounds[1] < -100 or src_bounds[3] > 100):
                        print(f"[DEBUG] Warning: Bounds look wrong for geographic CRS, recalculating transform")
                        print(f"[DEBUG] Bounds: {src_bounds}")
                        needs_reproject = True
                
                # Only use tiled format if dimensions are compatible
                # Tiled TIFF requires block dimensions to be multiples of 16
                height, width = data.shape
                use_tiled = (height % 16 == 0) and (width % 16 == 0)
                
                # If we need to recalculate transform, we need to get bounds from NetCDF metadata
                # For now, if bounds look wrong, try to use the transform as-is but with correct CRS
                # The issue is that we don't have the original geographic bounds from the NetCDF
                # when opened via GDAL pattern. We should fall back to netCDF4 approach if this happens.
                if needs_reproject:
                    print(f"[DEBUG] Cannot recalculate transform without original bounds, falling back to netCDF4 approach")
                    src.close()
                    # Fall through to netCDF4 approach
                    src = None
                else:
                    # Create profile with correct CRS and transform
                    profile = {
                        'driver': 'GTiff',
                        'height': height,
                        'width': width,
                        'count': 1,
                        'dtype': data.dtype,
                        'crs': output_crs,
                        'transform': src_transform,
                        'compress': 'lzw',
                        'tiled': use_tiled
                    }
                    
                    # Copy nodata if present
                    if src.nodata is not None:
                        profile['nodata'] = src.nodata
                    
                    # If tiled, set appropriate block sizes
                    if use_tiled:
                        # Use standard block sizes that are multiples of 16
                        profile['blockxsize'] = min(512, width)
                        profile['blockysize'] = min(512, height)
                        # Ensure block sizes are multiples of 16
                        profile['blockxsize'] = (profile['blockxsize'] // 16) * 16
                        profile['blockysize'] = (profile['blockysize'] // 16) * 16
                    
                    print(f"[DEBUG] Writing GeoTIFF with tiled={use_tiled}, shape={data.shape}, CRS={output_crs}")
                    print(f"[DEBUG] Output bounds will be: {src_bounds}")
                    # Write to GeoTIFF
                    with rasterio.open(output_path, 'w', **profile) as dst:
                        dst.write(data, 1)
                    src.close()
                    print(f"[DEBUG] Successfully wrote GeoTIFF")
                    return True
            except Exception as e:
                print(f"[DEBUG] Error writing GeoTIFF from rasterio source: {e}")
                import traceback
                print(f"[DEBUG] Traceback: {traceback.format_exc()}")
                if src:
                    src.close()
                return False
        
        print(f"[DEBUG] Could not open NetCDF file with any method")
        return False
        
    except Exception as e:
        print(f"[DEBUG] Error converting NetCDF to GeoTIFF: {e}")
        import traceback
        print(f"[DEBUG] Traceback: {traceback.format_exc()}")
        return False


def convert_geotiff_to_esri_ascii(geotiff_path, output_path):
    """
    Convert GeoTIFF file directly to ESRI ASCII grid format.
    
    Args:
        geotiff_path (str): Path to input GeoTIFF file
        output_path (str): Path to output ESRI ASCII file
        
    Returns:
        bool: True if successful, False otherwise
    """
    if rasterio is None:
        print(f"[DEBUG] Cannot convert GeoTIFF to ESRI ASCII: rasterio not available")
        return False
    
    try:
        with rasterio.open(geotiff_path) as src:
            data = src.read(1)  # Read first band
            transform = src.transform
            bounds = src.bounds
            nodata = src.nodata
            
            # Get corner coordinates
            xllcorner = bounds.left
            yllcorner = bounds.bottom
            ncols = data.shape[1]  # width (east-west)
            nrows = data.shape[0]  # height (north-south)
            
            # Calculate cell size from actual bounds and dimensions
            # ESRI ASCII uses a single cellsize value, so we calculate it from the actual extent
            # For geographic coordinates, X and Y cell sizes may differ, so we calculate both
            cell_size_x = (bounds.right - bounds.left) / ncols
            cell_size_y = (bounds.top - bounds.bottom) / nrows
            
            # ESRI ASCII format uses a single cellsize for both X and Y dimensions
            # Since east-west bounds are correct, use cell_size_x
            # However, if cells are rectangular, using X cell size for Y dimension will cause stretching
            # Adjust yllcorner so that yllcorner + nrows * cell_size_x = bounds.top (correct north extent)
            cell_size = cell_size_x
            
            # Calculate what yllcorner should be to match the actual north bound when using X cell size
            # ESRI ASCII calculates: north_extent = yllcorner + nrows * cellsize
            # We want: north_extent = bounds.top
            # So: yllcorner = bounds.top - nrows * cell_size
            yllcorner_adjusted = bounds.top - nrows * cell_size
            
            print(f"[DEBUG] ESRI ASCII conversion: calculated cell_size_x={cell_size_x:.9f}°, cell_size_y={cell_size_y:.9f}°")
            print(f"[DEBUG] Using cell_size={cell_size:.9f}° for ESRI ASCII (using X cell size since east-west is correct)")
            print(f"[DEBUG] Original yllcorner={yllcorner:.9f}°, adjusted yllcorner={yllcorner_adjusted:.9f}°")
            print(f"[DEBUG] Calculated north extent with adjusted yllcorner: {yllcorner_adjusted + nrows * cell_size:.9f}°, actual bounds.top: {bounds.top:.9f}°")
            
            yllcorner = yllcorner_adjusted
            
            # Replace NaN and nodata values with ESRI ASCII nodata value BEFORE any transformations
            nodata_value = -9999
            if nodata is not None:
                data = np.where(data == nodata, nodata_value, data)
            data = np.where(np.isnan(data), nodata_value, data)
            data = np.where(np.isinf(data), nodata_value, data)
            
            # ESRI ASCII format expects:
            # - First row in file = northernmost row (top)
            # - Each row = west to east (left to right)
            # - Data written top to bottom (north to south)
            # Rasterio data: shape (height, width) = (nrows, ncols)
            # - First row (index 0) = northernmost row
            # - Each row = west to east
            # This matches ESRI ASCII format, so write as-is (no flip, no transpose)
            
            # Write ESRI ASCII format (explicitly use ASCII/UTF-8 encoding for text)
            with open(output_path, 'w', encoding='ascii', errors='replace') as f:
                f.write(f"ncols {ncols}\n")
                f.write(f"nrows {nrows}\n")
                f.write(f"xllcorner {xllcorner:.6f}\n")
                f.write(f"yllcorner {yllcorner:.6f}\n")
                f.write(f"cellsize {cell_size:.6f}\n")
                f.write(f"NODATA_value {nodata_value}\n")
                np.savetxt(f, data, fmt='%.6f')
        
        print(f"[DEBUG] Successfully converted GeoTIFF to ESRI ASCII: {output_path}")
        return True
    except Exception as e:
        print(f"[DEBUG] Error converting GeoTIFF to ESRI ASCII: {e}")
        import traceback
        print(f"[DEBUG] Traceback: {traceback.format_exc()}")
        return False


def convert_netcdf_to_esri_ascii(netcdf_path, output_path):
    """
    Convert NetCDF file to ESRI ASCII grid format.
    This function is kept for backward compatibility or if needed for NetCDF inputs.
    
    Args:
        netcdf_path (str): Path to input NetCDF file
        output_path (str): Path to output ESRI ASCII file
        
    Returns:
        bool: True if successful, False otherwise
    """
    if rasterio is None or netCDF4 is None:
        return False
    
    try:
        # Try opening with rasterio first
        try:
            with rasterio.open(netcdf_path) as src:
                data = src.read(1)
                transform = src.transform
                bounds = src.bounds
                crs = src.crs
                
                # Calculate cell size
                cell_size = abs(transform[0])
                xllcorner = bounds.left
                yllcorner = bounds.bottom
                ncols = data.shape[1]
                nrows = data.shape[0]
                
        except:
            # Fallback to netCDF4
            with netCDF4.Dataset(netcdf_path, 'r') as nc:
                # Find data variable
                data_var = None
                for var_name in nc.variables:
                    var = nc.variables[var_name]
                    if len(var.dimensions) >= 2:
                        data_var = var_name
                        break
                
                if data_var is None:
                    return False
                
                data = nc.variables[data_var][:]
                if len(data.shape) == 3:
                    data = data[0, :, :]
                
                # Get coordinates
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
                
                # Fallback names
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
                    return False
                
                lats = nc.variables[lat_var][:]
                lons = nc.variables[lon_var][:]
                
                nrows, ncols = data.shape
                cell_size = abs(float(lons[1] - lons[0])) if len(lons) > 1 else 0.01
                xllcorner = float(lons.min())
                yllcorner = float(lats.min())
        
        # Flip data vertically (ESRI ASCII uses bottom-to-top)
        data = np.flipud(data)
        
        # Replace NaN with nodata value
        nodata_value = -9999
        data = np.where(np.isnan(data), nodata_value, data)
        
        # Write ESRI ASCII format (explicitly use ASCII/UTF-8 encoding for text)
        with open(output_path, 'w', encoding='ascii', errors='replace') as f:
            f.write(f"ncols {ncols}\n")
            f.write(f"nrows {nrows}\n")
            f.write(f"xllcorner {xllcorner:.6f}\n")
            f.write(f"yllcorner {yllcorner:.6f}\n")
            f.write(f"cellsize {cell_size:.6f}\n")
            f.write(f"NODATA_value {nodata_value}\n")
            np.savetxt(f, data, fmt='%.6f')
        
        return True
    except Exception as e:
        print(f"[DEBUG] Error converting NetCDF to ESRI ASCII: {e}")
        import traceback
        print(f"[DEBUG] Traceback: {traceback.format_exc()}")
        return False


def convert_netcdf_to_coards(netcdf_path, output_path):
    """
    Convert NetCDF file to COARDS format (which is also NetCDF-based).
    COARDS (Cooperative Ocean/Atmosphere Research Data Service) is a NetCDF convention.
    This ensures the file follows COARDS conventions.
    
    Args:
        netcdf_path (str): Path to input NetCDF file
        output_path (str): Path to output COARDS file
        
    Returns:
        bool: True if successful, False otherwise
    """
    if netCDF4 is None:
        return False
    
    try:
        # Open source NetCDF
        with netCDF4.Dataset(netcdf_path, 'r') as src:
            # Create output COARDS-compliant NetCDF
            with netCDF4.Dataset(output_path, 'w', format='NETCDF4') as dst:
                # Copy dimensions
                for dim_name, dim in src.dimensions.items():
                    if dim.isunlimited():
                        dst.createDimension(dim_name, None)
                    else:
                        dst.createDimension(dim_name, len(dim))
                
                # Copy variables
                for var_name, var in src.variables.items():
                    # Extract _FillValue if it exists (must be set at creation time)
                    fill_value = None
                    if '_FillValue' in var.ncattrs():
                        fill_value = var.getncattr('_FillValue')
                    
                    # Create variable with fill_value if it exists
                    if fill_value is not None:
                        dst_var = dst.createVariable(var_name, var.datatype, var.dimensions, fill_value=fill_value)
                    else:
                        dst_var = dst.createVariable(var_name, var.datatype, var.dimensions)
                    
                    # Copy data
                    dst_var[:] = var[:]
                    
                    # Copy attributes (skip _FillValue as it's already set)
                    for attr_name in var.ncattrs():
                        if attr_name != '_FillValue':  # Skip _FillValue as it's set at creation
                            dst_var.setncattr(attr_name, var.getncattr(attr_name))
                    
                    # Ensure COARDS compliance for coordinate variables
                    if var_name.lower() in ['lon', 'longitude', 'x']:
                        if 'units' not in var.ncattrs():
                            dst_var.units = 'degrees_east'
                        if 'long_name' not in var.ncattrs():
                            dst_var.long_name = 'longitude'
                    elif var_name.lower() in ['lat', 'latitude', 'y']:
                        if 'units' not in var.ncattrs():
                            dst_var.units = 'degrees_north'
                        if 'long_name' not in var.ncattrs():
                            dst_var.long_name = 'latitude'
                
                # Copy global attributes
                for attr_name in src.ncattrs():
                    dst.setncattr(attr_name, src.getncattr(attr_name))
                
                # Add COARDS convention attribute
                if 'Conventions' in dst.ncattrs():
                    conventions = dst.getncattr('Conventions')
                    if 'COARDS' not in str(conventions):
                        dst.Conventions = f"{conventions}, COARDS"
                else:
                    dst.Conventions = 'COARDS'
        
        print(f"[DEBUG] Successfully converted NetCDF to COARDS: {output_path}")
        return True
    except Exception as e:
        print(f"[DEBUG] Error converting NetCDF to COARDS: {e}")
        import traceback
        print(f"[DEBUG] Traceback: {traceback.format_exc()}")
        return False


def convert_geotiff_to_coards(geotiff_path, output_path):
    """
    Convert GeoTIFF file to COARDS format.
    First converts to NetCDF, then ensures COARDS compliance.
    
    Args:
        geotiff_path (str): Path to input GeoTIFF file
        output_path (str): Path to output COARDS file
        
    Returns:
        bool: True if successful, False otherwise
    """
    if rasterio is None or netCDF4 is None:
        print(f"[DEBUG] Cannot convert GeoTIFF to COARDS: rasterio or netCDF4 not available")
        return False
    
    try:
        # First convert GeoTIFF to NetCDF
        import tempfile
        temp_netcdf = tempfile.NamedTemporaryFile(suffix='.nc', delete=False)
        temp_netcdf_path = temp_netcdf.name
        temp_netcdf.close()
        
        try:
            # Convert GeoTIFF to NetCDF
            if not convert_geotiff_to_netcdf(geotiff_path, temp_netcdf_path):
                print(f"[DEBUG] Failed to convert GeoTIFF to NetCDF for COARDS conversion")
                return False
            
            # Convert NetCDF to COARDS
            if not convert_netcdf_to_coards(temp_netcdf_path, output_path):
                print(f"[DEBUG] Failed to convert NetCDF to COARDS")
                return False
            
            print(f"[DEBUG] Successfully converted GeoTIFF to COARDS: {output_path}")
            return True
        finally:
            # Clean up temporary NetCDF file
            try:
                if os.path.exists(temp_netcdf_path):
                    os.remove(temp_netcdf_path)
            except Exception as e:
                print(f"[DEBUG] Warning: Could not delete temporary NetCDF file: {e}")
                
    except Exception as e:
        print(f"[DEBUG] Error converting GeoTIFF to COARDS: {e}")
        import traceback
        print(f"[DEBUG] Traceback: {traceback.format_exc()}")
        return False


def convert_geotiff_to_netcdf(geotiff_path, output_path):
    """
    Convert GeoTIFF file to NetCDF format.
    
    Args:
        geotiff_path (str): Path to input GeoTIFF file
        output_path (str): Path to output NetCDF file
        
    Returns:
        bool: True if successful, False otherwise
    """
    if rasterio is None or netCDF4 is None:
        print(f"[DEBUG] Cannot convert GeoTIFF to NetCDF: rasterio or netCDF4 not available")
        return False
    
    try:
        # Open GeoTIFF with rasterio
        with rasterio.open(geotiff_path) as src:
            data = src.read(1)  # Read first band
            transform = src.transform
            crs = src.crs
            nodata = src.nodata
            
            # Get dimensions
            height, width = data.shape
            
            # Get bounds
            bounds = src.bounds
            lon_min = bounds.left
            lon_max = bounds.right
            lat_min = bounds.bottom
            lat_max = bounds.top
            
            # Calculate cell sizes
            cell_size_lon = (lon_max - lon_min) / width
            cell_size_lat = (lat_max - lat_min) / height
            
            # Create coordinate arrays
            lons = np.linspace(lon_min + cell_size_lon/2, lon_max - cell_size_lon/2, width)
            lats = np.linspace(lat_min + cell_size_lat/2, lat_max - cell_size_lat/2, height)
            
            # Create NetCDF file
            with netCDF4.Dataset(output_path, 'w', format='NETCDF4') as nc:
                # Create dimensions
                nc.createDimension('lon', width)
                nc.createDimension('lat', height)
                
                # Create coordinate variables
                lon_var = nc.createVariable('lon', 'f4', ('lon',))
                lat_var = nc.createVariable('lat', 'f4', ('lat',))
                lon_var[:] = lons
                lat_var[:] = lats
                
                # Set coordinate attributes
                lon_var.units = 'degrees_east'
                lon_var.long_name = 'longitude'
                lon_var.standard_name = 'longitude'
                lat_var.units = 'degrees_north'
                lat_var.long_name = 'latitude'
                lat_var.standard_name = 'latitude'
                
                # Create data variable with fill_value set at creation time
                fill_value = float(nodata) if nodata is not None else -99999.0
                data_var = nc.createVariable('z', 'f4', ('lat', 'lon'), fill_value=fill_value)
                data_var[:] = data
                data_var.units = 'meters'
                data_var.long_name = 'elevation'
                data_var.standard_name = 'height'
                
                # Set global attributes
                nc.title = 'Bathymetry/Topography Data'
                nc.source = 'GMRT GridServer'
                nc.Conventions = 'CF-1.6'
                if crs:
                    nc.geospatial_lat_min = float(lat_min)
                    nc.geospatial_lat_max = float(lat_max)
                    nc.geospatial_lon_min = float(lon_min)
                    nc.geospatial_lon_max = float(lon_max)
                    if crs.to_epsg() == 4326:
                        nc.geospatial_crs = 'EPSG:4326'
        
        print(f"[DEBUG] Successfully converted GeoTIFF to NetCDF: {output_path}")
        return True
    except Exception as e:
        print(f"[DEBUG] Error converting GeoTIFF to NetCDF: {e}")
        import traceback
        print(f"[DEBUG] Traceback: {traceback.format_exc()}")
        return False


class DownloadWorker(QThread):
    """
    Worker thread for downloading bathymetry data files from GMRT GridServer.
    
    This class runs in a separate thread to prevent the GUI from freezing
    during large file downloads. It handles streaming downloads and provides
    detailed error messages for various failure scenarios. It downloads
    directly as GeoTIFF format.
    """
    # Signals to communicate with the main thread
    finished = pyqtSignal(bool, str)  # (success, filename/error_message)
    
    def __init__(self, params, filename, requested_format=None):
        """
        Initialize the download worker with request parameters and target filename.
        
        Args:
            params (dict): Parameters for the GMRT GridServer API request (format will be set to 'geotiff')
            filename (str): Full path where the final file should be saved
            requested_format (str): Not used (kept for compatibility, downloads as GeoTIFF)
        """
        super().__init__()
        # Always download as GeoTIFF
        self.params = params.copy()
        self.params['format'] = 'geotiff'
        self.filename = filename  # Final output filename
        self.requested_format = requested_format  # Kept for compatibility but not used
    
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
                    # Download to temporary GeoTIFF file first
                    temp_geotiff = tempfile.NamedTemporaryFile(suffix='.tif', delete=False)
                    temp_geotiff_path = temp_geotiff.name
                    temp_geotiff.close()
                    
                    total_bytes = 0
                    with open(temp_geotiff_path, 'wb') as f:
                        for chunk in r.iter_content(chunk_size=8192):  # 8KB chunks
                            if chunk:
                                f.write(chunk)
                                total_bytes += len(chunk)
                    
                    print(f"[DownloadWorker] GeoTIFF download completed successfully ({total_bytes} bytes)")
                    
                    # Check if temp file exists and has content
                    if not os.path.exists(temp_geotiff_path):
                        error_msg = f"Temporary GeoTIFF file not found: {temp_geotiff_path}"
                        print(f"[DownloadWorker] {error_msg}")
                        self.finished.emit(False, error_msg)
                        return
                    
                    file_size = os.path.getsize(temp_geotiff_path)
                    print(f"[DownloadWorker] Temp GeoTIFF file size: {file_size} bytes")
                    if file_size == 0:
                        error_msg = f"Downloaded GeoTIFF file is empty"
                        print(f"[DownloadWorker] {error_msg}")
                        try:
                            os.remove(temp_geotiff_path)
                        except:
                            pass
                        self.finished.emit(False, error_msg)
                        return
                    
                    # Move the downloaded GeoTIFF file to the final location
                    try:
                        if temp_geotiff_path != self.filename:
                            shutil.move(temp_geotiff_path, self.filename)
                        else:
                            # Already at correct location
                            pass
                    except Exception as e:
                        print(f"[DownloadWorker] Error moving GeoTIFF file: {e}")
                        # Try copy instead
                        try:
                            shutil.copy2(temp_geotiff_path, self.filename)
                            os.remove(temp_geotiff_path)
                        except Exception as e2:
                            error_msg = f"Error saving GeoTIFF file: {e2}"
                            print(f"[DownloadWorker] {error_msg}")
                            self.finished.emit(False, error_msg)
                            return
                    
                    # Download completed successfully
                    print(f"[DownloadWorker] Successfully downloaded GeoTIFF file")
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
        
        # Timer for debouncing map preview updates when typing coordinates
        self.map_preview_timer = QTimer()
        self.map_preview_timer.setSingleShot(True)  # Only fire once
        self.map_preview_timer.timeout.connect(self.update_map_preview)
        print("[DEBUG] Initialized map preview debounce timer")
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
        self.west_spin.valueChanged.connect(self.on_coordinate_changed)  # Debounced update
        aoi_form.addRow("West (min lon)", self.west_spin)

        # Eastern boundary (maximum longitude)
        self.east_spin = QDoubleSpinBox()
        self.east_spin.setRange(-180, 180)  # Valid longitude range
        self.east_spin.setDecimals(6)       # 6 decimal places for precision
        self.east_spin.valueChanged.connect(self.on_coordinate_changed)  # Debounced update
        aoi_form.addRow("East (max lon)", self.east_spin)

        # Southern boundary (minimum latitude)
        self.south_spin = QDoubleSpinBox()
        self.south_spin.setRange(-85, 85)   # Valid latitude range (GMRT data limit)
        self.south_spin.setDecimals(6)      # 6 decimal places for precision
        self.south_spin.valueChanged.connect(self.on_coordinate_changed)  # Debounced update
        aoi_form.addRow("South (min lat)", self.south_spin)

        # Northern boundary (maximum latitude)
        self.north_spin = QDoubleSpinBox()
        self.north_spin.setRange(-85, 85)   # Valid latitude range (GMRT data limit)
        self.north_spin.setDecimals(6)      # 6 decimal places for precision
        self.north_spin.valueChanged.connect(self.on_coordinate_changed)  # Debounced update
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
            "GeoTIFF",           # GeoTIFF - Raster format with embedded georeference
            "NetCDF",            # NetCDF - Scientific data format with metadata
            "NetCDF (COARDS)"    # COARDS - NetCDF convention for oceanographic/atmospheric data
        ])
        grid_form.addRow("Output Format", self.format_combo)

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
        grid_form.addRow("GMRT Source", self.layer_combo)

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

        # Tiling is now automatic when area exceeds 2 degrees in either dimension
        # Tile size is always 2 degrees (no UI parameter needed)

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
        # Call update_map_preview once after all values are set
        self.update_map_preview()
        

    def on_coordinate_changed(self):
        """
        Handle coordinate value changes with debouncing.
        Restarts the timer so update_map_preview is only called after user stops typing.
        """
        # Stop the timer if it's running and restart it
        # This ensures update_map_preview is only called 500ms after the user stops typing
        self.map_preview_timer.stop()
        self.map_preview_timer.start(500)  # 500ms delay

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
        
        # Tile size is always 2.0 degrees
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

    def download_single_grid(self, params, filename, callback=None, requested_format=None):
        """
        Download a single grid file using worker thread.
        Downloads directly as GeoTIFF format.
        
        Args:
            params: Parameters for API request (format will be set to 'geotiff')
            filename: Final output filename
            callback: Callback function when download completes
            requested_format: Not used (kept for compatibility, downloads as GeoTIFF)
        """
        if self.current_worker and self.current_worker.isRunning():
            return False  # Already downloading
        
        # Log the request parameters for debugging
        param_str = ", ".join([f"{k}={v}" for k, v in params.items()])
        self.log_message(f"Making request (will download as GeoTIFF): {param_str}")
        
        self.current_worker = DownloadWorker(params, filename, requested_format)
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
        format_type_display = self.format_combo.currentText()
        # Normalize format type for internal use
        # Handle "NetCDF (COARDS)" -> "coards", "NetCDF" -> "netcdf", "GeoTIFF" -> "geotiff"
        format_type_lower = format_type_display.lower()
        if "coards" in format_type_lower:
            format_type = "coards"
        elif "netcdf" in format_type_lower:
            format_type = "netcdf"
        else:
            format_type = format_type_lower
        layer_type_display = self.layer_combo.currentText()
        layer_type = self.get_layer_type()
        mres_value = self.mres_combo.currentText()
        
        self.log_message(f"Grid download request:")
        self.log_message(f"  Format: {format_type_display}")
        self.log_message(f"  Layer: {layer_type_display}")
        self.log_message(f"  Cell Resolution: {mres_value} meters/pixel")
        layer_type = self.get_layer_type()
        
        # Map format to file extension (downloads are always GeoTIFF, but final format may differ)
        format_extensions = {
            "geotiff": ".tif",
            "netcdf": ".nc",
            "coards": ".grd"
        }
        file_ext = format_extensions.get(format_type, ".tif")
        current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        layer_type_display = self.layer_combo.currentText()
        self.log_message(f"Format: {format_type}, Layer: {layer_type_display}, Cell Resolution: {self.mres_combo.currentText()} meters/pixel")
        self.log_message("Parameters validated successfully")

        # Automatically determine if tiling is needed based on 2-degree threshold
        lon_span = abs(east - west)
        lat_span = abs(north - south)
        needs_tiling = (lat_span > 2.0) or (lon_span > 2.0)
        
        if needs_tiling:
            self.log_message(f"Area exceeds 2-degree threshold (lat: {lat_span:.2f}°, lon: {lon_span:.2f}°), automatically tiling download")
            # Tiled download - automatically mosaic and delete tiles
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
                "layer": layer_type,
                "mresolution": float(self.mres_combo.currentText())
            }

            self.status_label.setText("Downloading...")
            self.status_label.repaint()
            
            self.download_btn.setEnabled(False)  # Disable button during download
            self.download_single_grid(params, file_name, self.on_single_download_finished, format_type)
            
    def download_next_tile(self):
        """Download the next tile in the sequence"""
        if self.current_tile_index >= len(self.tiles_to_download):
            return  # All tiles downloaded

        tile = self.tiles_to_download[self.current_tile_index]
        # Use lower left coordinates instead of date/time
        ll_lon = f"{tile['west']:.3f}".replace('-', 'm').replace('.', 'p')
        ll_lat = f"{tile['south']:.3f}".replace('-', 'm').replace('.', 'p')
        # Tiles always use .tif extension (they remain as GeoTIFF)
        # Only the final mosaic will be converted to the output format
        tile_ext = '.tif'
        tile_filename = f"gmrt_{self.layer_type}_{ll_lon}_{ll_lat}_{self.current_tile_index + 1:03d}{tile_ext}"
        tile_path = os.path.join(self.download_dir, tile_filename)
        
        self.log_message(f"Starting download of tile {self.current_tile_index + 1}/{len(self.tiles_to_download)}: {tile_filename}")
        
        params = {
            "west": tile['west'],
            "east": tile['east'],
            "south": tile['south'],
            "north": tile['north'],
            "layer": self.layer_type,
            "mresolution": float(self.mres_combo.currentText())
        }
        
        self.status_label.setText(f"Downloading tile {self.current_tile_index + 1}/{len(self.tiles_to_download)}: {tile_filename}")
        self.status_label.repaint()
        
        self.download_single_grid(params, tile_path, self.on_tile_download_finished, self.format_type)

    def split_grid_file(self, filename, format_type):
        """
        Split a grid file into topography (>=0) and bathymetry (<0) files for supported formats.
        Appends _topo and _bathy to the base filename.
        For NetCDF format, splits GeoTIFF first, then converts to NetCDF.
        """
        import os
        base, ext = os.path.splitext(filename)
        
        # Determine output extension based on format
        if format_type == 'netcdf':
            output_ext = '.nc'
        elif format_type == 'esriascii':
            output_ext = '.asc'
        elif format_type == 'coards':
            output_ext = '.grd'
        else:
            output_ext = ext  # Default to input extension (GeoTIFF)
        
        topo_file = base + '_topo' + output_ext
        bathy_file = base + '_bathy' + output_ext
        try:
            deleted_files = []
            
            # For NetCDF and COARDS formats, we need to split as GeoTIFF first, then convert
            if format_type == 'netcdf' and rasterio is not None:
                # Split as GeoTIFF first
                temp_topo_tif = base + '_topo_temp.tif'
                temp_bathy_tif = base + '_bathy_temp.tif'
                
                with rasterio.open(filename) as src:
                    data = src.read(1)
                    profile = src.profile
                    topo_data = np.where(data >= 0, data, np.nan)
                    bathy_data = np.where(data < 0, data, np.nan)
                    profile.update(dtype=rasterio.float32, nodata=np.nan)
                    with rasterio.open(temp_topo_tif, 'w', **profile) as dst:
                        dst.write(topo_data.astype(np.float32), 1)
                    with rasterio.open(temp_bathy_tif, 'w', **profile) as dst:
                        dst.write(bathy_data.astype(np.float32), 1)
                
                # Check if files are empty and convert to NetCDF
                topo_data_check = np.where(np.isnan(topo_data), 0, topo_data)
                bathy_data_check = np.where(np.isnan(bathy_data), 0, bathy_data)
                
                if not np.isnan(topo_data).all() and os.path.getsize(temp_topo_tif) > 0:
                    if convert_geotiff_to_netcdf(temp_topo_tif, topo_file):
                        self.log_message(f"Split topography to NetCDF: {os.path.basename(topo_file)}")
                    else:
                        self.log_message(f"Failed to convert topography to NetCDF")
                else:
                    try:
                        os.remove(temp_topo_tif)
                        deleted_files.append(temp_topo_tif)
                        self.log_message(f"Deleted empty topography GeoTIFF")
                    except:
                        pass
                
                if not np.isnan(bathy_data).all() and os.path.getsize(temp_bathy_tif) > 0:
                    if convert_geotiff_to_netcdf(temp_bathy_tif, bathy_file):
                        self.log_message(f"Split bathymetry to NetCDF: {os.path.basename(bathy_file)}")
                    else:
                        self.log_message(f"Failed to convert bathymetry to NetCDF")
                else:
                    try:
                        os.remove(temp_bathy_tif)
                        deleted_files.append(temp_bathy_tif)
                        self.log_message(f"Deleted empty bathymetry GeoTIFF")
                    except:
                        pass
                
                # Clean up temporary GeoTIFF files
                for temp_file in [temp_topo_tif, temp_bathy_tif]:
                    try:
                        if os.path.exists(temp_file):
                            os.remove(temp_file)
                    except:
                        pass
            
            # For COARDS format, split as GeoTIFF first, then convert to COARDS
            elif format_type == 'coards' and rasterio is not None:
                # Split as GeoTIFF first
                temp_topo_tif = base + '_topo_temp.tif'
                temp_bathy_tif = base + '_bathy_temp.tif'
                
                with rasterio.open(filename) as src:
                    data = src.read(1)
                    profile = src.profile
                    topo_data = np.where(data >= 0, data, np.nan)
                    bathy_data = np.where(data < 0, data, np.nan)
                    profile.update(dtype=rasterio.float32, nodata=np.nan)
                    with rasterio.open(temp_topo_tif, 'w', **profile) as dst:
                        dst.write(topo_data.astype(np.float32), 1)
                    with rasterio.open(temp_bathy_tif, 'w', **profile) as dst:
                        dst.write(bathy_data.astype(np.float32), 1)
                
                # Check if files are empty and convert to COARDS
                if not np.isnan(topo_data).all() and os.path.getsize(temp_topo_tif) > 0:
                    if convert_geotiff_to_coards(temp_topo_tif, topo_file):
                        self.log_message(f"Split topography to COARDS: {os.path.basename(topo_file)}")
                    else:
                        self.log_message(f"Failed to convert topography to COARDS")
                else:
                    try:
                        os.remove(temp_topo_tif)
                        deleted_files.append(temp_topo_tif)
                        self.log_message(f"Deleted empty topography GeoTIFF")
                    except:
                        pass
                
                if not np.isnan(bathy_data).all() and os.path.getsize(temp_bathy_tif) > 0:
                    if convert_geotiff_to_coards(temp_bathy_tif, bathy_file):
                        self.log_message(f"Split bathymetry to COARDS: {os.path.basename(bathy_file)}")
                    else:
                        self.log_message(f"Failed to convert bathymetry to COARDS")
                else:
                    try:
                        os.remove(temp_bathy_tif)
                        deleted_files.append(temp_bathy_tif)
                        self.log_message(f"Deleted empty bathymetry GeoTIFF")
                    except:
                        pass
                
                # Clean up temporary GeoTIFF files
                for temp_file in [temp_topo_tif, temp_bathy_tif]:
                    try:
                        if os.path.exists(temp_file):
                            os.remove(temp_file)
                    except:
                        pass
                        
            elif format_type == 'geotiff' and rasterio is not None:
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
            # Normalize format type for internal use
            format_type_display = self.format_combo.currentText()
            format_type_lower = format_type_display.lower()
            if "coards" in format_type_lower:
                format_type = "coards"
            elif "netcdf" in format_type_lower:
                format_type = "netcdf"
            else:
                format_type = format_type_lower
            
            # Split if requested (this handles conversion to NetCDF/COARDS if format is NetCDF/COARDS)
            if self.split_checkbox.isChecked():
                self.split_grid_file(result, format_type)
            # Convert to NetCDF if format is NetCDF and split is not enabled
            elif format_type == 'netcdf':
                base, ext = os.path.splitext(result)
                # If filename already has .nc extension, use a temp file for conversion
                if ext.lower() == '.nc':
                    temp_tif = base + '_temp.tif'
                    try:
                        # Rename the downloaded file to .tif temporarily
                        os.rename(result, temp_tif)
                        netcdf_file = result  # Use original filename
                        if convert_geotiff_to_netcdf(temp_tif, netcdf_file):
                            self.log_message(f"Converted to NetCDF: {os.path.basename(netcdf_file)}")
                            # Delete the temporary GeoTIFF file
                            try:
                                os.remove(temp_tif)
                            except Exception as e:
                                self.log_message(f"Failed to delete temporary GeoTIFF file: {e}")
                            result = netcdf_file
                        else:
                            # Conversion failed, restore original file
                            os.rename(temp_tif, result)
                            self.log_message(f"Failed to convert to NetCDF, keeping GeoTIFF file")
                    except Exception as e:
                        self.log_message(f"Error during NetCDF conversion: {e}")
                else:
                    # Filename has .tif extension, convert normally
                    netcdf_file = base + '.nc'
                    if convert_geotiff_to_netcdf(result, netcdf_file):
                        self.log_message(f"Converted to NetCDF: {os.path.basename(netcdf_file)}")
                        # Delete the temporary GeoTIFF file
                        try:
                            os.remove(result)
                            self.log_message(f"Deleted temporary GeoTIFF file")
                        except Exception as e:
                            self.log_message(f"Failed to delete temporary GeoTIFF file: {e}")
                        result = netcdf_file  # Update result to point to NetCDF file
                    else:
                        self.log_message(f"Failed to convert to NetCDF, keeping GeoTIFF file")
            # Convert to COARDS if format is COARDS and split is not enabled
            elif format_type == 'coards':
                base, ext = os.path.splitext(result)
                # If filename already has .grd extension, use a temp file for conversion
                if ext.lower() == '.grd':
                    temp_tif = base + '_temp.tif'
                    try:
                        os.rename(result, temp_tif)
                        coards_file = result
                        if convert_geotiff_to_coards(temp_tif, coards_file):
                            self.log_message(f"Converted to COARDS: {os.path.basename(coards_file)}")
                            try:
                                os.remove(temp_tif)
                            except Exception as e:
                                self.log_message(f"Failed to delete temporary GeoTIFF file: {e}")
                            result = coards_file
                        else:
                            os.rename(temp_tif, result)
                            self.log_message(f"Failed to convert to COARDS, keeping GeoTIFF file")
                    except Exception as e:
                        self.log_message(f"Error during COARDS conversion: {e}")
                else:
                    coards_file = base + '.grd'
                    if convert_geotiff_to_coards(result, coards_file):
                        self.log_message(f"Converted to COARDS: {os.path.basename(coards_file)}")
                        try:
                            os.remove(result)
                            self.log_message(f"Deleted temporary GeoTIFF file")
                        except Exception as e:
                            self.log_message(f"Failed to delete temporary GeoTIFF file: {e}")
                        result = coards_file
                    else:
                        self.log_message(f"Failed to convert to COARDS, keeping GeoTIFF file")
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
                # Tiles remain as GeoTIFF and only the final mosaic will be converted to the output format
                # No need to convert individual tiles since they will be automatically mosaicked
            else:
                self.log_message(f"Tile {self.current_tile_index + 1} failed: {result}")
            
            # Move to next tile
            self.current_tile_index += 1
            
            # Schedule next download with a 2-second delay
            if self.current_tile_index < len(self.tiles_to_download):
                self.log_message(f"Waiting 2 seconds before downloading tile {self.current_tile_index + 1}")
                # Use QTimer.singleShot to ensure this runs on the main thread
                QTimer.singleShot(2000, self.download_next_tile)
            else:
                # All tiles downloaded, automatically start mosaicking
                self.log_message(f"All {len(self.tiles_to_download)} tiles completed. Success count: {self.success_count}")
                self.log_message(f"Downloaded tile files: {self.downloaded_tile_files}")
                
                if self.downloaded_tile_files:
                    self.log_message("All tiles downloaded, automatically starting mosaicking process...")
                    # Use QTimer.singleShot to ensure this runs on the main thread
                    QTimer.singleShot(3000, self.start_mosaicking)
                else:
                    # No tiles downloaded, finish normally
                    QTimer.singleShot(100, self.finish_tile_download)
                    
        except Exception as e:
            import traceback
            self.log_message(f"ERROR in on_tile_download_finished: {str(e)}")
            self.log_message(f"Traceback: {traceback.format_exc()}")
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
            # Note: delete_tiles_checkbox is None since tiles are always deleted after mosaicking
            print(f"[DEBUG] Creating MosaicWorker with {len(self.downloaded_tile_files)} tiles")
            self.mosaic_worker = MosaicWorker(
                self.downloaded_tile_files,
                self.download_dir,
                self.layer_type,
                self.west_spin,
                self.south_spin,
                self.east_spin,
                self.north_spin,
                None,  # delete_tiles_checkbox - always delete tiles after mosaicking
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
                    
                    # Always delete individual tile files after mosaicking
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
                    
                    # Get format type
                    format_type = self.format_type
                    
                    # Apply splitting to the mosaicked file if requested
                    if self.split_checkbox.isChecked():
                        print("[DEBUG] Applying split to mosaicked file...")
                        self.log_message("Applying split to mosaicked file...")
                        self.split_grid_file(mosaic_path, format_type)
                    # Convert to NetCDF if format is NetCDF and split is not enabled
                    elif format_type == 'netcdf':
                        print("[DEBUG] Converting mosaicked file to NetCDF...")
                        self.log_message("Converting mosaicked file to NetCDF...")
                        base, ext = os.path.splitext(mosaic_path)
                        # If filename already has .nc extension, use a temp file for conversion
                        if ext.lower() == '.nc':
                            temp_tif = base + '_temp.tif'
                            try:
                                os.rename(mosaic_path, temp_tif)
                                netcdf_file = mosaic_path
                                if convert_geotiff_to_netcdf(temp_tif, netcdf_file):
                                    self.log_message(f"Converted mosaic to NetCDF: {os.path.basename(netcdf_file)}")
                                    try:
                                        os.remove(temp_tif)
                                    except Exception as e:
                                        self.log_message(f"Failed to delete temporary GeoTIFF mosaic: {e}")
                                    mosaic_path = netcdf_file
                                else:
                                    os.rename(temp_tif, mosaic_path)
                                    self.log_message(f"Failed to convert mosaic to NetCDF, keeping GeoTIFF file")
                            except Exception as e:
                                self.log_message(f"Error during mosaic NetCDF conversion: {e}")
                        else:
                            netcdf_file = base + '.nc'
                            if convert_geotiff_to_netcdf(mosaic_path, netcdf_file):
                                self.log_message(f"Converted mosaic to NetCDF: {os.path.basename(netcdf_file)}")
                                try:
                                    os.remove(mosaic_path)
                                    self.log_message(f"Deleted temporary GeoTIFF mosaic file")
                                except Exception as e:
                                    self.log_message(f"Failed to delete temporary GeoTIFF mosaic: {e}")
                                mosaic_path = netcdf_file
                            else:
                                self.log_message(f"Failed to convert mosaic to NetCDF, keeping GeoTIFF file")
                    # Convert to COARDS if format is COARDS and split is not enabled
                    elif format_type == 'coards':
                        print("[DEBUG] Converting mosaicked file to COARDS...")
                        self.log_message("Converting mosaicked file to COARDS...")
                        base, ext = os.path.splitext(mosaic_path)
                        # If filename already has .grd extension, use a temp file for conversion
                        if ext.lower() == '.grd':
                            temp_tif = base + '_temp.tif'
                            try:
                                os.rename(mosaic_path, temp_tif)
                                coards_file = mosaic_path
                                if convert_geotiff_to_coards(temp_tif, coards_file):
                                    self.log_message(f"Converted mosaic to COARDS: {os.path.basename(coards_file)}")
                                    try:
                                        os.remove(temp_tif)
                                    except Exception as e:
                                        self.log_message(f"Failed to delete temporary GeoTIFF mosaic: {e}")
                                    mosaic_path = coards_file
                                else:
                                    os.rename(temp_tif, mosaic_path)
                                    self.log_message(f"Failed to convert mosaic to COARDS, keeping GeoTIFF file")
                            except Exception as e:
                                self.log_message(f"Error during mosaic COARDS conversion: {e}")
                        else:
                            coards_file = base + '.grd'
                            if convert_geotiff_to_coards(mosaic_path, coards_file):
                                self.log_message(f"Converted mosaic to COARDS: {os.path.basename(coards_file)}")
                                try:
                                    os.remove(mosaic_path)
                                    self.log_message(f"Deleted temporary GeoTIFF mosaic file")
                                except Exception as e:
                                    self.log_message(f"Failed to delete temporary GeoTIFF mosaic: {e}")
                                mosaic_path = coards_file
                            else:
                                self.log_message(f"Failed to convert mosaic to COARDS, keeping GeoTIFF file")
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
            
            # Always delete individual tile files after mosaicking
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
            "layer": layer_type,
            "mresolution": float(self.mres_combo.currentText())
        }

        self.status_label.setText("Downloading...")
        self.status_label.repaint()
        
        self.download_btn.setEnabled(False)  # Disable button during download
        self.download_single_grid(params, file_name, self.on_single_download_finished, format_type)

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
