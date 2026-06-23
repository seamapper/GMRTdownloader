# Copyright (c) 2026 Paul Johnson
# SPDX-License-Identifier: MIT

"""
Configuration and constants for GMRT Bathymetry Grid Downloader.
"""

import math
import os
from datetime import datetime
from urllib.parse import urlencode

# Version (match last uncommented __version__ in release)
__version__ = "2026.03"  
# GMRT API endpoints
GMRT_URL = "https://www.gmrt.org/services/GridServer"  # For downloading bathymetry data
GMRT_IMAGE_URL = "https://www.gmrt.org/services/ImageServer"  # For map preview images
GMRT_LOG_FILENAME = "gmrt_log.txt"

# Map preview (ImageServer)
MAP_PREVIEW_REQUEST_TIMEOUT = 30   # seconds per HTTP request
MAP_PREVIEW_MAX_RETRIES = 2        # retries after a failed attempt (3 tries total)
MAP_PREVIEW_RETRY_DELAY_MS = 5000  # wait before each retry
MAP_PREVIEW_WATCHDOG_MS = 35000    # no response watchdog (timeout + buffer)

# Tiled grid download
MAX_TILES_PER_DOWNLOAD = 253     # refuse download when tiling would exceed this count
TILE_DOWNLOAD_MAX_RETRIES = 2      # retries per tile after failure (3 attempts total)
TILE_DOWNLOAD_MISSING_PASS_MAX = 1  # full passes over failed tiles after initial pass
TILE_DOWNLOAD_RETRY_DELAY_MS = 5000  # wait before retrying a failed tile
TILE_DOWNLOAD_INTER_TILE_DELAY_MS = 2000  # wait before starting the next tile
TILE_DOWNLOAD_REQUEST_TIMEOUT = 120  # seconds per tile HTTP request
TILE_DOWNLOAD_WATCHDOG_MS = 130000  # fail tile if still running after this (timeout + buffer)

# Resolution-dependent tile footprint (120 m baseline = 2° per side)
TILE_SIZE_BASE_MRES = 120.0
TILE_SIZE_BASE_DEGREES = 2.0


def tile_size_degrees_for_mres(mres: float) -> float:
    """Tile side length in degrees (60 m -> 1°, 120 m -> 2°, 240 m -> 4°, etc.)."""
    return TILE_SIZE_BASE_DEGREES * (float(mres) / TILE_SIZE_BASE_MRES)


def needs_tiling_for_spans(lon_span: float, lat_span: float, mres: float) -> bool:
    """True when lon_span + lat_span exceeds two tile sides at this resolution."""
    tile_size = tile_size_degrees_for_mres(mres)
    return (lon_span + lat_span) > (2.0 * tile_size)


MIN_TILE_STRIP_DEGREES = 0.1


def build_degree_strips(
    span_start: float, span_end: float, tile_size: float, overlap: float
) -> list[tuple[float, float]]:
    """Return (start, end) pairs for tile strips along one axis."""
    if span_end <= span_start:
        return []
    strips: list[tuple[float, float]] = []
    current = span_start
    step = max(tile_size - overlap, 0.01)
    max_iters = max(1, int(math.ceil((span_end - span_start) / step)) + 2)
    iteration = 0
    while current < span_end and iteration < max_iters:
        tile_end = min(current + tile_size, span_end)
        if tile_end - current >= MIN_TILE_STRIP_DEGREES:
            strips.append((current, tile_end))
            current = tile_end - overlap
        else:
            break
        iteration += 1
    return strips


def format_gmrt_grid_request_url(params: dict) -> str:
    """Full GridServer GET URL (always requests GeoTIFF, matching DownloadWorker)."""
    request_params = {**params, "format": "geotiff"}
    return f"{GMRT_URL}?{urlencode(request_params)}"


def append_gmrt_log(output_dir: str, line: str) -> None:
    """Append a timestamped line to gmrt_log.txt in the output directory."""
    if not output_dir:
        return
    try:
        os.makedirs(output_dir, exist_ok=True)
        log_path = os.path.join(output_dir, GMRT_LOG_FILENAME)
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        with open(log_path, "a", encoding="utf-8") as log_file:
            log_file.write(f"[{timestamp}] {line}\n")
    except OSError:
        pass


def init_gmrt_log_session(output_dir: str, header: str) -> None:
    """Start a new download session block in gmrt_log.txt."""
    append_gmrt_log(output_dir, "=" * 72)
    append_gmrt_log(output_dir, header)
