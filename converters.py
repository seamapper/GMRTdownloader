# Copyright (c) 2026 Paul Johnson
# SPDX-License-Identifier: MIT

"""
Format conversion functions for GMRT bathymetry data (GeoTIFF, NetCDF, COARDS, ESRI ASCII).
"""

import os
import tempfile
import numpy as np

try:
    import rasterio
except ImportError:
    rasterio = None
try:
    import netCDF4
except ImportError:
    netCDF4 = None


def convert_netcdf_to_geotiff(netcdf_path, output_path):
    """Convert NetCDF file to GeoTIFF format. Returns True if successful."""
    if rasterio is None:
        return False
    try:
        src = None
        try:
            src = rasterio.open(netcdf_path)
        except Exception:
            pass
        if src is None:
            try:
                src = rasterio.open(netcdf_path, driver='NETCDF')
            except Exception:
                pass
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
                    break
                except Exception:
                    continue
        if src is not None:
            try:
                src_bounds = src.bounds
                src_crs = src.crs
                needs_fallback = False
                if src_crs is None or not src_crs.is_geographic or src_crs.to_epsg() != 4326:
                    needs_fallback = True
                elif (src_bounds[0] < -200 or src_bounds[2] > 200 or src_bounds[1] < -100 or src_bounds[3] > 100):
                    needs_fallback = True
                if needs_fallback:
                    src.close()
                    src = None
            except Exception:
                if src:
                    src.close()
                src = None
        if src is None and netCDF4 is not None:
            try:
                with netCDF4.Dataset(netcdf_path, 'r') as nc:
                    if 'z' in nc.variables and 'dimension' in nc.variables and 'x_range' in nc.variables and 'y_range' in nc.variables:
                        try:
                            dimension = nc.variables['dimension'][:]
                            if len(dimension) < 2:
                                return False
                            ncols = int(dimension[0])
                            nrows = int(dimension[1])
                            x_range = nc.variables['x_range'][:]
                            y_range = nc.variables['y_range'][:]
                            lon_min, lon_max = float(x_range[0]), float(x_range[1])
                            lat_min, lat_max = float(y_range[0]), float(y_range[1])
                            z_data = nc.variables['z'][:]
                            if z_data.size != nrows * ncols:
                                return False
                            data = z_data.reshape((nrows, ncols), order='C')
                            from rasterio.transform import from_bounds
                            transform = from_bounds(lon_min, lat_min, lon_max, lat_max, ncols, nrows)
                            use_tiled = (nrows % 16 == 0) and (ncols % 16 == 0)
                            profile = {
                                'driver': 'GTiff', 'height': nrows, 'width': ncols, 'count': 1,
                                'dtype': data.dtype, 'crs': 'EPSG:4326', 'transform': transform,
                                'compress': 'lzw', 'tiled': use_tiled, 'nodata': -99999
                            }
                            if use_tiled:
                                profile['blockxsize'] = (min(512, ncols) // 16) * 16
                                profile['blockysize'] = (min(512, nrows) // 16) * 16
                            with rasterio.open(output_path, 'w', **profile) as dst:
                                dst.write(data, 1)
                            return True
                        except Exception:
                            return False
                    data_var = None
                    for var_name in nc.variables:
                        if len(nc.variables[var_name].dimensions) >= 2:
                            data_var = var_name
                            break
                    if data_var is None:
                        return False
                    data = nc.variables[data_var][:]
                    if len(data.shape) == 3:
                        data = data[0, :, :]
                    dims = nc.variables[data_var].dimensions
                    lat_var = lon_var = None
                    for dim in dims:
                        if dim in nc.variables and hasattr(nc.variables[dim], 'standard_name'):
                            sn = nc.variables[dim].standard_name.lower()
                            if 'lat' in sn:
                                lat_var = dim
                            elif 'lon' in sn:
                                lon_var = dim
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
                    lat_min, lat_max = float(lats.min()), float(lats.max())
                    lon_min, lon_max = float(lons.min()), float(lons.max())
                    from rasterio.transform import from_bounds
                    transform = from_bounds(lon_min, lat_min, lon_max, lat_max, ncols, nrows)
                    use_tiled = (nrows % 16 == 0) and (ncols % 16 == 0)
                    profile = {
                        'driver': 'GTiff', 'height': nrows, 'width': ncols, 'count': 1,
                        'dtype': data.dtype, 'crs': 'EPSG:4326', 'transform': transform,
                        'compress': 'lzw', 'tiled': use_tiled
                    }
                    if use_tiled:
                        profile['blockxsize'] = (min(512, ncols) // 16) * 16
                        profile['blockysize'] = (min(512, nrows) // 16) * 16
                    if hasattr(nc.variables[data_var], '_FillValue'):
                        profile['nodata'] = float(nc.variables[data_var]._FillValue)
                    with rasterio.open(output_path, 'w', **profile) as dst:
                        dst.write(data, 1)
                    return True
            except Exception:
                return False
        if src is not None:
            try:
                data = src.read(1)
                src_bounds, src_crs, src_transform = src.bounds, src.crs, src.transform
                output_crs = rasterio.crs.CRS.from_epsg(4326)
                needs_reproject = (src_crs is None or not src_crs.is_geographic or src_crs.to_epsg() != 4326 or
                    (src_bounds[0] < -200 or src_bounds[2] > 200 or src_bounds[1] < -100 or src_bounds[3] > 100))
                height, width = data.shape
                use_tiled = (height % 16 == 0) and (width % 16 == 0)
                if needs_reproject:
                    src.close()
                    return False
                profile = {
                    'driver': 'GTiff', 'height': height, 'width': width, 'count': 1,
                    'dtype': data.dtype, 'crs': output_crs, 'transform': src_transform,
                    'compress': 'lzw', 'tiled': use_tiled
                }
                if src.nodata is not None:
                    profile['nodata'] = src.nodata
                if use_tiled:
                    profile['blockxsize'] = (min(512, width) // 16) * 16
                    profile['blockysize'] = (min(512, height) // 16) * 16
                with rasterio.open(output_path, 'w', **profile) as dst:
                    dst.write(data, 1)
                src.close()
                return True
            except Exception:
                if src:
                    src.close()
                return False
        return False
    except Exception:
        return False


def convert_geotiff_to_esri_ascii(geotiff_path, output_path):
    """Convert GeoTIFF file to ESRI ASCII grid format. Returns True if successful."""
    if rasterio is None:
        return False
    try:
        with rasterio.open(geotiff_path) as src:
            data = src.read(1)
            bounds = src.bounds
            nodata = src.nodata
            ncols = data.shape[1]
            nrows = data.shape[0]
            cell_size_x = (bounds.right - bounds.left) / ncols
            cell_size = cell_size_x
            yllcorner = bounds.top - nrows * cell_size
            nodata_value = -9999
            if nodata is not None:
                data = np.where(data == nodata, nodata_value, data)
            data = np.where(np.isnan(data), nodata_value, data)
            data = np.where(np.isinf(data), nodata_value, data)
            with open(output_path, 'w', encoding='ascii', errors='replace') as f:
                f.write(f"ncols {ncols}\n")
                f.write(f"nrows {nrows}\n")
                f.write(f"xllcorner {bounds.left:.6f}\n")
                f.write(f"yllcorner {yllcorner:.6f}\n")
                f.write(f"cellsize {cell_size:.6f}\n")
                f.write(f"NODATA_value {nodata_value}\n")
                np.savetxt(f, data, fmt='%.6f')
        return True
    except Exception:
        return False


def convert_netcdf_to_esri_ascii(netcdf_path, output_path):
    """Convert NetCDF file to ESRI ASCII grid format. Returns True if successful."""
    if rasterio is None or netCDF4 is None:
        return False
    try:
        try:
            with rasterio.open(netcdf_path) as src:
                data = src.read(1)
                bounds = src.bounds
                cell_size = abs(src.transform[0])
                xllcorner = bounds.left
                yllcorner = bounds.bottom
                ncols = data.shape[1]
                nrows = data.shape[0]
        except Exception:
            with netCDF4.Dataset(netcdf_path, 'r') as nc:
                data_var = next((n for n, v in nc.variables.items() if len(v.dimensions) >= 2), None)
                if data_var is None:
                    return False
                data = nc.variables[data_var][:]
                if len(data.shape) == 3:
                    data = data[0, :, :]
                dims = nc.variables[data_var].dimensions
                lat_var = next((d for d in dims if d in nc.variables and 'lat' in getattr(nc.variables[d], 'standard_name', '')), None) or next((n for n in ['lat', 'latitude', 'y'] if n in nc.variables), None)
                lon_var = next((d for d in dims if d in nc.variables and 'lon' in getattr(nc.variables[d], 'standard_name', '')), None) or next((n for n in ['lon', 'longitude', 'x'] if n in nc.variables), None)
                if lat_var is None or lon_var is None:
                    return False
                lats = nc.variables[lat_var][:]
                lons = nc.variables[lon_var][:]
                nrows, ncols = data.shape
                cell_size = abs(float(lons[1] - lons[0])) if len(lons) > 1 else 0.01
                xllcorner = float(lons.min())
                yllcorner = float(lats.min())
        data = np.flipud(data)
        nodata_value = -9999
        data = np.where(np.isnan(data), nodata_value, data)
        with open(output_path, 'w', encoding='ascii', errors='replace') as f:
            f.write(f"ncols {ncols}\n")
            f.write(f"nrows {nrows}\n")
            f.write(f"xllcorner {xllcorner:.6f}\n")
            f.write(f"yllcorner {yllcorner:.6f}\n")
            f.write(f"cellsize {cell_size:.6f}\n")
            f.write(f"NODATA_value {nodata_value}\n")
            np.savetxt(f, data, fmt='%.6f')
        return True
    except Exception:
        return False


def convert_netcdf_to_coards(netcdf_path, output_path):
    """Convert NetCDF file to COARDS format. Returns True if successful."""
    if netCDF4 is None:
        return False
    try:
        with netCDF4.Dataset(netcdf_path, 'r') as src:
            with netCDF4.Dataset(output_path, 'w', format='NETCDF4') as dst:
                for dim_name, dim in src.dimensions.items():
                    dst.createDimension(dim_name, None if dim.isunlimited() else len(dim))
                for var_name, var in src.variables.items():
                    fill_value = var.getncattr('_FillValue') if '_FillValue' in var.ncattrs() else None
                    dst_var = dst.createVariable(var_name, var.datatype, var.dimensions, fill_value=fill_value) if fill_value is not None else dst.createVariable(var_name, var.datatype, var.dimensions)
                    dst_var[:] = var[:]
                    for attr_name in var.ncattrs():
                        if attr_name != '_FillValue':
                            dst_var.setncattr(attr_name, var.getncattr(attr_name))
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
                for attr_name in src.ncattrs():
                    dst.setncattr(attr_name, src.getncattr(attr_name))
                if 'Conventions' in dst.ncattrs():
                    conventions = dst.getncattr('Conventions')
                    if 'COARDS' not in str(conventions):
                        dst.Conventions = f"{conventions}, COARDS"
                else:
                    dst.Conventions = 'COARDS'
        return True
    except Exception:
        return False


def convert_geotiff_to_coards(geotiff_path, output_path):
    """Convert GeoTIFF file to COARDS format (via NetCDF). Returns True if successful."""
    if rasterio is None or netCDF4 is None:
        return False
    try:
        temp_netcdf = tempfile.NamedTemporaryFile(suffix='.nc', delete=False)
        temp_netcdf_path = temp_netcdf.name
        temp_netcdf.close()
        try:
            if not convert_geotiff_to_netcdf(geotiff_path, temp_netcdf_path):
                return False
            if not convert_netcdf_to_coards(temp_netcdf_path, output_path):
                return False
            return True
        finally:
            try:
                if os.path.exists(temp_netcdf_path):
                    os.remove(temp_netcdf_path)
            except Exception:
                pass
    except Exception:
        return False


def convert_geotiff_to_netcdf(geotiff_path, output_path):
    """Convert GeoTIFF file to NetCDF format. Returns True if successful."""
    if rasterio is None or netCDF4 is None:
        return False
    try:
        with rasterio.open(geotiff_path) as src:
            data = src.read(1)
            transform = src.transform
            crs = src.crs
            nodata = src.nodata
            height, width = data.shape
            bounds = src.bounds
            lon_min, lon_max = bounds.left, bounds.right
            lat_min, lat_max = bounds.bottom, bounds.top
            cell_size_lon = (lon_max - lon_min) / width
            cell_size_lat = (lat_max - lat_min) / height
            lons = np.linspace(lon_min + cell_size_lon/2, lon_max - cell_size_lon/2, width)
            lats = np.linspace(lat_min + cell_size_lat/2, lat_max - cell_size_lat/2, height)
            with netCDF4.Dataset(output_path, 'w', format='NETCDF4') as nc:
                nc.createDimension('lon', width)
                nc.createDimension('lat', height)
                lon_var = nc.createVariable('lon', 'f4', ('lon',))
                lat_var = nc.createVariable('lat', 'f4', ('lat',))
                lon_var[:] = lons
                lat_var[:] = lats
                lon_var.units = 'degrees_east'
                lon_var.long_name = 'longitude'
                lon_var.standard_name = 'longitude'
                lat_var.units = 'degrees_north'
                lat_var.long_name = 'latitude'
                lat_var.standard_name = 'latitude'
                fill_value = float(nodata) if nodata is not None else -99999.0
                data_var = nc.createVariable('z', 'f4', ('lat', 'lon'), fill_value=fill_value)
                data_var[:] = data
                data_var.units = 'meters'
                data_var.long_name = 'elevation'
                data_var.standard_name = 'height'
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
        return True
    except Exception:
        return False
