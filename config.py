# Copyright (c) 2026 Paul Johnson
# SPDX-License-Identifier: MIT

"""
Configuration and constants for GMRT Bathymetry Grid Downloader.
"""

# Version (match last uncommented __version__ in release)
# __version__ = "2025.05"  # Cleaned up the UI
# __version__ = "2025.06"  # Streamlined download process by removing tiling options
__version__ = "2026.02"  # Refactored the code to use the main.py file
# GMRT API endpoints
GMRT_URL = "https://www.gmrt.org/services/GridServer"  # For downloading bathymetry data
GMRT_IMAGE_URL = "https://www.gmrt.org/services/ImageServer"  # For map preview images

# Map preview (ImageServer)
MAP_PREVIEW_REQUEST_TIMEOUT = 30   # seconds per HTTP request
MAP_PREVIEW_MAX_RETRIES = 2        # retries after a failed attempt (3 tries total)
MAP_PREVIEW_RETRY_DELAY_MS = 5000  # wait before each retry
MAP_PREVIEW_WATCHDOG_MS = 35000    # no response watchdog (timeout + buffer)
