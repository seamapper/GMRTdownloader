"""
Configuration and constants for GMRT Bathymetry Grid Downloader.
"""

# Version (match last uncommented __version__ in release)
# __version__ = "2025.05"  # Cleaned up the UI
__version__ = "2025.06"  # Streamlined download process by removing tiling options

# GMRT API endpoints
GMRT_URL = "https://www.gmrt.org/services/GridServer"  # For downloading bathymetry data
GMRT_IMAGE_URL = "https://www.gmrt.org/services/ImageServer"  # For map preview images
