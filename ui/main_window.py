# Copyright (c) 2026 Paul Johnson
# SPDX-License-Identifier: MIT

import os
import json
import math
from datetime import datetime
from PyQt6.QtWidgets import (
    QApplication, QWidget, QLabel, QLineEdit, QPushButton,
    QVBoxLayout, QHBoxLayout, QFileDialog, QMessageBox, QDoubleSpinBox,
    QGroupBox, QFormLayout, QCheckBox, QTextEdit, QSplitter, QFrame, QSizePolicy
)
from PyQt6.QtCore import Qt, QTimer, QThread, pyqtSignal, QUrl, QRect, QPoint
from PyQt6.QtGui import QPixmap, QImage, QPainter, QPen, QColor, QIcon, QPalette
import numpy as np
try:
    import rasterio
except ImportError:
    rasterio = None

from config import (
    __version__,
    GMRT_LOG_FILENAME,
    MAX_TILES_PER_DOWNLOAD,
    MAP_PREVIEW_MAX_RETRIES,
    MAP_PREVIEW_RETRY_DELAY_MS,
    MAP_PREVIEW_WATCHDOG_MS,
    TILE_DOWNLOAD_MAX_RETRIES,
    TILE_DOWNLOAD_RETRY_DELAY_MS,
    TILE_DOWNLOAD_INTER_TILE_DELAY_MS,
    TILE_DOWNLOAD_WATCHDOG_MS,
    tile_size_degrees_for_mres,
    needs_tiling_for_spans,
    build_degree_strips,
    format_gmrt_grid_request_url,
    init_gmrt_log_session,
)
from workers import MapWorker, MosaicWorker, DownloadWorker
from converters import convert_geotiff_to_netcdf, convert_geotiff_to_coards
from ui.map_widget import MapWidget
from ui.form_selector import FormSelector
from ui.download_progress_dialog import DownloadProgressDialog

# Activity log colors
_LOG_COLOR_PROBLEM = "#c45c00"

# Substrings that mark a problem line in the activity log (case-insensitive)
_LOG_PROBLEM_MARKERS = (
    "error",
    "failed",
    "failure",
    "timeout",
    "timed out",
    "invalid",
    "✗",
    "no data returned",
    "connection error",
    "request too large",
    "retries exhausted",
    "critical error",
    "could not",
    "not available",
    "missing or unreadable",
    "skipping",
    "empty response",
    "empty image",
    "watchdog",
)


def _project_root():
    """Project root (parent of ui/). Used for media and config paths."""
    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


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
        super().__init__()
        self.setWindowTitle(f"GMRT Bathymetry Grid Downloader v{__version__} - pjohnson@ccom.unh.edu")
        # Set window icon
        icon_path = os.path.join(_project_root(), "media", "GMRT-logo2020.ico")
        if os.path.exists(icon_path):
            self.setWindowIcon(QIcon(icon_path))
        else:
            pass
        # Data structures for managing tiled downloads
        self.tiles_to_download = []        # List of tile boundaries to download
        self.current_tile_index = 0        # Current tile being downloaded
        self.download_dir = ""             # Directory for saving downloaded files
        self.downloaded_tile_files = []    # List of successfully downloaded tile files
        # Timer for managing sequential tile downloads
        self.download_timer = QTimer()
        self.download_timer.timeout.connect(self.download_next_tile)
        
        # Timer for debouncing map preview updates when typing coordinates
        self.map_preview_timer = QTimer()
        self.map_preview_timer.setSingleShot(True)  # Only fire once
        self.map_preview_timer.timeout.connect(self.update_map_preview)
        # Map preview retry / watchdog (timeouts and failed loads)
        self._map_preview_retry_count = 0
        self._map_preview_request_id = 0
        self._active_map_request_id = 0
        self.map_preview_retry_timer = QTimer()
        self.map_preview_retry_timer.setSingleShot(True)
        self.map_preview_retry_timer.timeout.connect(self._retry_map_preview)
        self.map_preview_watchdog_timer = QTimer()
        self.map_preview_watchdog_timer.setSingleShot(True)
        self.map_preview_watchdog_timer.timeout.connect(self._on_map_preview_watchdog)
        # Configuration and state management
        self.config_file = os.path.join(_project_root(), "gmrtgrab_config.json")
        self.last_download_dir = self.load_last_download_dir()
        # Area-of-interest history (up to 10 extents): back/forward stacks
        self.aoi_past = []   # AOIs you can go back to (Zoom Previous)
        self.aoi_future = [] # AOIs you can go forward to (Zoom Next)
        # Worker threads for background operations
        self.current_worker = None         # Current download worker
        self.current_map_worker = None     # Current map preview worker
        self.download_progress = None      # Progress dialog during download/mosaic
        self._download_progress_mode = None  # 'single', 'tiles', or 'mosaic'
        self._tile_retry_count = 0         # retries for the current tile (tiled downloads)
        self.tile_download_watchdog = QTimer()
        self.tile_download_watchdog.setSingleShot(True)
        self.tile_download_watchdog.timeout.connect(self._on_tile_download_watchdog)
        # Initialize the user interface
        self.init_ui()
        # Set default window size
        self.resize(1200, 800)

    def load_last_download_dir(self):
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
                with open(self.config_file, 'r') as f:
                    config = json.load(f)
                    last_dir = config.get('last_download_dir', '')
                    # Verify the directory still exists
                    if last_dir and os.path.exists(last_dir):
                        return last_dir
        except Exception as e:
            # If any error occurs (file doesn't exist, invalid JSON, etc.),
            # we'll use the default directory
            pass
        
        # Default to user's home directory
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

        # Area of Interest subgroup — plus layout: North top, West/East middle, South bottom
        aoi_group = QGroupBox("Area of Interest")
        aoi_layout = QVBoxLayout()

        # Northern boundary (top, centered)
        self.north_spin = QDoubleSpinBox()
        self.north_spin.setRange(-85, 85)
        self.north_spin.setDecimals(6)
        self.north_spin.valueChanged.connect(self.on_coordinate_changed)
        self.north_spin.valueChanged.connect(self.update_estimated_pixel_count)
        north_row = QHBoxLayout()
        north_row.addStretch()
        north_row.addWidget(QLabel("North"))
        north_row.addWidget(self.north_spin)
        north_row.addStretch()
        aoi_layout.addLayout(north_row)

        # West (left) and East (right) on same row
        self.west_spin = QDoubleSpinBox()
        self.west_spin.setRange(-180, 180)
        self.west_spin.setDecimals(6)
        self.west_spin.valueChanged.connect(self.on_coordinate_changed)
        self.west_spin.valueChanged.connect(self.update_estimated_pixel_count)
        self.east_spin = QDoubleSpinBox()
        self.east_spin.setRange(-180, 180)
        self.east_spin.setDecimals(6)
        self.east_spin.valueChanged.connect(self.on_coordinate_changed)
        self.east_spin.valueChanged.connect(self.update_estimated_pixel_count)
        we_row = QHBoxLayout()
        we_row.addWidget(QLabel("West"))
        we_row.addWidget(self.west_spin)
        we_row.addStretch()
        we_row.addWidget(QLabel("East"))
        we_row.addWidget(self.east_spin)
        aoi_layout.addLayout(we_row)

        # Southern boundary (bottom, centered)
        self.south_spin = QDoubleSpinBox()
        self.south_spin.setRange(-85, 85)
        self.south_spin.setDecimals(6)
        self.south_spin.valueChanged.connect(self.on_coordinate_changed)
        self.south_spin.valueChanged.connect(self.update_estimated_pixel_count)
        south_row = QHBoxLayout()
        south_row.addStretch()
        south_row.addWidget(QLabel("South"))
        south_row.addWidget(self.south_spin)
        south_row.addStretch()
        aoi_layout.addLayout(south_row)

        # Estimated pixel count (bounds × cell resolution)
        aoi_info_policy = QSizePolicy(
            QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Minimum
        )
        aoi_info_align = (
            Qt.AlignmentFlag.AlignHCenter | Qt.AlignmentFlag.AlignTop
        )
        self.estimated_pixels_label = QLabel("Est. pixels: —")
        self.estimated_pixels_label.setWordWrap(False)
        self.estimated_pixels_label.setSizePolicy(aoi_info_policy)
        self.estimated_pixels_label.setAlignment(aoi_info_align)
        self.estimated_pixels_label.setStyleSheet(self._secondary_label_style())
        aoi_layout.addWidget(self.estimated_pixels_label)
        self.estimated_tiles_label = QLabel("")
        self.estimated_tiles_label.setWordWrap(False)
        self.estimated_tiles_label.setSizePolicy(aoi_info_policy)
        self.estimated_tiles_label.setAlignment(aoi_info_align)
        self.estimated_tiles_label.setStyleSheet(self._secondary_label_style())
        self.estimated_tiles_label.hide()
        aoi_layout.addWidget(self.estimated_tiles_label)
        self.estimated_tile_warning_label = QLabel("")
        self.estimated_tile_warning_label.setWordWrap(True)
        self.estimated_tile_warning_label.setSizePolicy(aoi_info_policy)
        self.estimated_tile_warning_label.setAlignment(aoi_info_align)
        self.estimated_tile_warning_label.setStyleSheet(
            f"color: {_LOG_COLOR_PROBLEM}; font-size: 9pt;"
        )
        self.estimated_tile_warning_label.hide()
        aoi_layout.addWidget(self.estimated_tile_warning_label)

        aoi_group.setLayout(aoi_layout)
        form_layout.addRow(aoi_group)

        # === OUTPUT FORMAT AND RESOLUTION CONTROLS ===

        # Grid Parameters subgroup inside Download Parameters
        grid_group = QGroupBox("Grid Parameters")
        grid_form = QFormLayout()
        
        # Output format selection
        # Different formats are suitable for different applications
        self.format_combo = FormSelector()
        self.format_combo.addItems([
            "GeoTIFF",           # GeoTIFF - Raster format with embedded georeference
            "NetCDF",            # NetCDF - Scientific data format with metadata
            "NetCDF (COARDS)"    # COARDS - NetCDF convention for oceanographic/atmospheric data
        ])
        grid_form.addRow("Output Format", self.format_combo)

        # Cell resolution selection
        # Allows users to specify meter-per-pixel resolution
        self.mres_combo = FormSelector()
        self.mres_combo.addItems(["60", "120", "240", "480", "960"])
        self.mres_combo.setCurrentIndex(3)  # 480 m/pixel
        self.mres_combo.currentIndexChanged.connect(self._on_mres_changed)
        grid_form.addRow("Cell Resolution (meters/pixel)", self.mres_combo)

        # Data layer selection
        # Different layers provide different types of bathymetry data
        self.layer_combo = FormSelector()
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
        self.log_area.setStyleSheet("")
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

        # Drawing / zoom controls
        draw_controls_layout = QHBoxLayout()

        self.zoom_previous_btn = QPushButton("Zoom to Previous")
        self.zoom_previous_btn.setToolTip("Zoom to the previous Area of Interest (up to 10 steps)")
        self.zoom_previous_btn.clicked.connect(self.zoom_to_previous)
        self.zoom_previous_btn.setEnabled(False)  # No history at startup
        draw_controls_layout.addWidget(self.zoom_previous_btn)

        self.zoom_default_btn = QPushButton("Zoom to Defaults")
        self.zoom_default_btn.setToolTip("Zoom to starting map defaults")
        self.zoom_default_btn.clicked.connect(self.zoom_to_default)
        draw_controls_layout.addWidget(self.zoom_default_btn)

        self.zoom_next_btn = QPushButton("Zoom to Next")
        self.zoom_next_btn.setToolTip("Zoom to the next Area of Interest after using Zoom to Previous")
        self.zoom_next_btn.clicked.connect(self.zoom_to_next)
        self.zoom_next_btn.setEnabled(False)
        draw_controls_layout.addWidget(self.zoom_next_btn)
        
        map_layout.addLayout(draw_controls_layout)
        
        map_group.setLayout(map_layout)
        right_layout.addWidget(map_group)
        
        # === CREDIT LINE ===
        # Required attribution for GMRT data usage
        # This citation must be included when using GMRT data
        credit_group = QGroupBox("Credit")
        credit_group.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        credit_group.setStyleSheet("QGroupBox { padding-top: 2px; }")
        credit_layout = QVBoxLayout()
        credit_layout.setContentsMargins(2, 2, 2, 2)  # Minimal margins
        credit_layout.setSpacing(0)  # No spacing
        credit_label = QLabel(
            "Ryan, W.B.F., S.M. Carbotte, J.O. Coplan, S. O'Hara, A. Melkonian, "
            "R. Arko, R.A. Weissel, V. Ferrini, A. Goodwillie, F. Nitsche, J. Bonczkowski, "
            "and R. Zemsky (2009), Global Multi-Resolution Topography synthesis, "
            "Geochem. Geophys. Geosyst., 10, Q03014, doi: 10.1029/2008GC002332"
        )
        credit_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        credit_label.setWordWrap(True)  # Allow text to wrap to multiple lines
        credit_label.setStyleSheet(self._secondary_label_style(padding="0px"))
        credit_label.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        credit_layout.addWidget(credit_label)
        credit_group.setLayout(credit_layout)
        right_layout.addWidget(credit_group)
        
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
        
        # === INITIALIZATION ===
        
        # Log the application startup
        self.log_message("GMRT Bathymetry Grid Downloader started")
        self.log_message("Tip: To select an Area of Interest, move the mouse over the map (crosshair cursor) and left-drag to draw a red rectangle; the North/West/East/South fields and estimated pixels will update automatically.")
        
        # Block signals to prevent multiple update_map_preview calls during initial setup
        self.west_spin.blockSignals(True)
        self.east_spin.blockSignals(True)
        self.south_spin.blockSignals(True)
        self.north_spin.blockSignals(True)
        # Set default coordinates for a sample area
        self.west_spin.setValue(-180.0)   # Western boundary
        self.east_spin.setValue(180.0)    # Eastern boundary
        self.south_spin.setValue(-83.0)   # Southern boundary (default)
        self.north_spin.setValue(85.0)    # Northern boundary (GMRT data limit)
        self.west_spin.blockSignals(False)
        self.east_spin.blockSignals(False)
        self.south_spin.blockSignals(False)
        self.north_spin.blockSignals(False)
        # Call update_map_preview once after all values are set
        self.update_map_preview()
        self.update_estimated_pixel_count()

    def on_coordinate_changed(self):
        """
        Handle coordinate value changes with debouncing.
        Restarts the timer so update_map_preview is only called after user stops typing.
        """
        # Stop the timer if it's running and restart it
        # This ensures update_map_preview is only called 500ms after the user stops typing
        self.map_preview_timer.stop()
        self.map_preview_timer.start(500)  # 500ms delay

    def _on_mres_changed(self, index: int) -> None:
        """Handle cell resolution changes (index-based for reliable macOS updates)."""
        if index >= 0:
            self.update_estimated_pixel_count()

    @staticmethod
    def _secondary_label_style(padding: str = "") -> str:
        """Muted label style that follows the system palette."""
        pal = QApplication.palette()
        color = pal.color(QPalette.ColorRole.PlaceholderText).name()
        pad = f" padding: {padding};" if padding else ""
        return f"QLabel {{ color: {color}; font-size: 9pt;{pad} }}"

    @staticmethod
    def _log_text_color(is_problem: bool) -> str:
        if is_problem:
            return _LOG_COLOR_PROBLEM
        return QApplication.palette().color(QPalette.ColorRole.Text).name()

    def _compute_tile_counts(
        self, west: float, east: float, south: float, north: float, cell_m: float
    ) -> tuple[int, int, int, float, bool]:
        """
        Return (n_lon, n_lat, n_tiles, tile_size_deg, needs_tiling) for the current AOI.
        """
        lon_span = east - west
        lat_span = north - south
        tile_size = tile_size_degrees_for_mres(cell_m)
        if not needs_tiling_for_spans(lon_span, lat_span, cell_m):
            return 1, 1, 1, tile_size, False
        overlap = self.calculate_overlap_from_resolution()
        n_lon = len(build_degree_strips(west, east, tile_size, overlap))
        n_lat = len(build_degree_strips(south, north, tile_size, overlap))
        return n_lon, n_lat, n_lon * n_lat, tile_size, True

    def _sync_download_button_for_tile_limit(self) -> None:
        """Enable Download only when tile count is within the allowed maximum."""
        if self.download_btn.text() != "Download Grid":
            return
        if self.current_worker and self.current_worker.isRunning():
            return
        try:
            west = self.west_spin.value()
            east = self.east_spin.value()
            south = self.south_spin.value()
            north = self.north_spin.value()
            if east <= west or north <= south:
                self.download_btn.setEnabled(True)
                return
            cell_m = float(self.mres_combo.currentText())
            if cell_m <= 0:
                self.download_btn.setEnabled(True)
                return
            _, _, n_tiles, _, _ = self._compute_tile_counts(
                west, east, south, north, cell_m
            )
            self.download_btn.setEnabled(n_tiles <= MAX_TILES_PER_DOWNLOAD)
        except Exception:
            self.download_btn.setEnabled(True)

    def update_estimated_pixel_count(self):
        """Update the estimated pixel count from current bounds and cell resolution."""
        try:
            west = self.west_spin.value()
            east = self.east_spin.value()
            south = self.south_spin.value()
            north = self.north_spin.value()
            if east <= west or north <= south:
                self.estimated_pixels_label.setText("Est. pixels: —")
                self.estimated_tiles_label.hide()
                self.estimated_tile_warning_label.hide()
                self._sync_download_button_for_tile_limit()
                return
            cell_m = float(self.mres_combo.currentText())
            if cell_m <= 0:
                self.estimated_pixels_label.setText("Est. pixels: —")
                self.estimated_tiles_label.hide()
                self.estimated_tile_warning_label.hide()
                self._sync_download_button_for_tile_limit()
                return
            # Approx meters per degree: lat ~111320 m/deg; lon at center lat = 111320*cos(center_lat)
            center_lat_rad = math.radians((north + south) / 2.0)
            m_per_deg_lon = 111320.0 * math.cos(center_lat_rad)
            m_per_deg_lat = 111320.0
            width_m = (east - west) * m_per_deg_lon
            height_m = (north - south) * m_per_deg_lat
            width_px = int(round(width_m / cell_m))
            height_px = int(round(height_m / cell_m))
            total = width_px * height_px
            n_lon, n_lat, n_tiles, tile_size, needs_tiling = self._compute_tile_counts(
                west, east, south, north, cell_m
            )
            self.estimated_pixels_label.setText(
                f"Est. pixels: {width_px:,} × {height_px:,} ({total:,} total)"
            )
            if needs_tiling:
                self.estimated_tiles_label.setText(
                    f"Tiles: {n_lon}×{n_lat} ({n_tiles} total) @ {tile_size:g}°"
                )
                self.estimated_tiles_label.show()
            else:
                self.estimated_tiles_label.setText(
                    f"Single download ({tile_size:g}° tile)"
                )
                self.estimated_tiles_label.show()
            if needs_tiling and n_tiles > MAX_TILES_PER_DOWNLOAD:
                self.estimated_tile_warning_label.setText(
                    f"Over {MAX_TILES_PER_DOWNLOAD} tiles ({n_tiles}): "
                    "reduce area or resolution."
                )
                self.estimated_tile_warning_label.show()
                self.estimated_tile_warning_label.adjustSize()
            else:
                self.estimated_tile_warning_label.hide()
            self._sync_download_button_for_tile_limit()
        except Exception:
            self.estimated_pixels_label.setText("Est. pixels: —")
            self.estimated_tiles_label.hide()
            self.estimated_tile_warning_label.hide()
            self._sync_download_button_for_tile_limit()

    def _cancel_map_preview_retry(self):
        """Stop pending map preview retries and watchdog."""
        self.map_preview_retry_timer.stop()
        self.map_preview_watchdog_timer.stop()

    def _disconnect_map_worker(self):
        """Detach signals from the current map worker (abandon hung requests)."""
        worker = self.current_map_worker
        if worker is None:
            return
        try:
            worker.map_loaded.disconnect(self.on_map_loaded)
        except (TypeError, RuntimeError):
            pass
        try:
            worker.map_error.disconnect(self.on_map_error)
        except (TypeError, RuntimeError):
            pass
        self.current_map_worker = None

    def _schedule_map_preview_retry(self, reason: str):
        """Queue another map preview attempt after a delay."""
        if self._map_preview_retry_count >= MAP_PREVIEW_MAX_RETRIES:
            self.map_status_label.setText("Map: Error (retries exhausted)")
            self.log_message("Map preview failed after all retry attempts")
            self.refresh_map_btn.setEnabled(True)
            return

        self._map_preview_retry_count += 1
        delay_s = MAP_PREVIEW_RETRY_DELAY_MS // 1000
        self.map_status_label.setText(
            f"Map: Retrying in {delay_s}s ({self._map_preview_retry_count}/{MAP_PREVIEW_MAX_RETRIES})..."
        )
        self.log_message(
            f"Map preview {reason}; retry {self._map_preview_retry_count}/"
            f"{MAP_PREVIEW_MAX_RETRIES} in {delay_s}s"
        )
        self.refresh_map_btn.setEnabled(True)
        self.map_preview_retry_timer.start(MAP_PREVIEW_RETRY_DELAY_MS)

    def _retry_map_preview(self):
        """Retry loading the map for the current area of interest."""
        self.update_map_preview(is_retry=True)

    def _on_map_preview_watchdog(self):
        """Reload if a preview request produces no result within the watchdog window."""
        if self._active_map_request_id == 0:
            return
        if not self.map_status_label.text().startswith("Map: Loading"):
            return
        self.log_message("Map preview watchdog: no response in time")
        self._disconnect_map_worker()
        self._schedule_map_preview_retry("timed out")

    def update_map_preview(self, is_retry=False):
        """
        Update the map preview with the current coordinate settings.
        
        This method validates the coordinates, stops any existing map worker,
        creates a new worker thread to download the map image, and updates
        the UI to show the loading state.
        """
        self._cancel_map_preview_retry()
        if not is_retry:
            self._map_preview_retry_count = 0

        # Validate coordinates first to ensure they make sense
        west = self.west_spin.value()
        east = self.east_spin.value()
        south = self.south_spin.value()
        north = self.north_spin.value()
        
        # Check that coordinates form a valid rectangle
        if east <= west or north <= south:
            self.map_status_label.setText("Map: Invalid coordinates")
            return
        
        # Log the map preview request
        attempt = self._map_preview_retry_count + 1
        if is_retry:
            self.log_message(
                f"Retrying map preview (attempt {attempt}): "
                f"{west:.4f}°E to {east:.4f}°E, {south:.4f}°N to {north:.4f}°N"
            )
        else:
            self.log_message(
                f"Requesting map preview: {west:.4f}°E to {east:.4f}°E, "
                f"{south:.4f}°N to {north:.4f}°N"
            )
        
        # Do not start a new worker if the previous one is still running
        if self.current_map_worker and self.current_map_worker.isRunning():
            return

        self._map_preview_request_id += 1
        request_id = self._map_preview_request_id
        self._active_map_request_id = request_id
        
        # Create new map worker with current settings
        self.current_map_worker = MapWorker(
            west, east, south, north, 
            width=800,  # Fixed width for consistent preview quality
            mask=self.mask_checkbox.isChecked()  # Use current mask setting
        )
        self.current_map_worker.request_id = request_id
        
        # Connect worker signals to UI update methods
        self.current_map_worker.map_loaded.connect(self.on_map_loaded)
        self.current_map_worker.map_error.connect(self.on_map_error)
        
        # Update UI to show loading state
        self.map_status_label.setText("Map: Loading...")
        self.refresh_map_btn.setEnabled(False)  # Prevent multiple requests
        self.map_preview_watchdog_timer.start(MAP_PREVIEW_WATCHDOG_MS)
        self.current_map_worker.start()  # Start the background download
    
    def on_map_loaded(self, pixmap, request_id=None):
        """
        Handle successful map loading from the worker thread.
        
        This method is called when the map worker successfully downloads
        and processes the map image. It scales the image to fit the display
        area and updates the UI accordingly.
        
        Args:
            pixmap (QPixmap): The loaded map image
            request_id: Active preview request id (stale responses are ignored)
        """
        if request_id is None and self.current_map_worker is not None:
            request_id = getattr(self.current_map_worker, "request_id", None)
        if request_id is not None and request_id != self._active_map_request_id:
            return

        self._cancel_map_preview_retry()

        if pixmap.isNull() or pixmap.width() == 0 or pixmap.height() == 0:
            self._schedule_map_preview_retry("received empty image")
            return

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
        self._map_preview_retry_count = 0
        self.log_message("Map preview updated")
    
    def on_map_error(self, error_msg, request_id=None):
        """
        Handle map loading errors from the worker thread.
        
        This method is called when the map worker encounters an error
        during the download or processing of the map image.
        
        Args:
            error_msg (str): Description of the error that occurred
            request_id: Active preview request id (stale responses are ignored)
        """
        if request_id is None and self.current_map_worker is not None:
            request_id = getattr(self.current_map_worker, "request_id", None)
        if request_id is not None and request_id != self._active_map_request_id:
            return

        self._cancel_map_preview_retry()
        self.log_message(f"Map preview error: {error_msg}")

        # Keep the previous image if any; schedule a retry for the same AOI
        self.map_status_label.setText("Map: Error — retrying...")
        self._schedule_map_preview_retry(error_msg)

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

    @staticmethod
    def _escape_log_html(text: str) -> str:
        return (
            text.replace("&", "&amp;")
            .replace("<", "&lt;")
            .replace(">", "&gt;")
        )

    @classmethod
    def _is_problem_log_message(cls, message: str) -> bool:
        lower = message.lower()
        return any(marker in lower for marker in _LOG_PROBLEM_MARKERS)

    def log_message(self, message, *, problem: bool | None = None):
        """Add a message to the log with timestamp; problems are shown in orange."""
        timestamp = datetime.now().strftime("%H:%M:%S")
        is_problem = self._is_problem_log_message(message) if problem is None else problem
        color = self._log_text_color(is_problem)
        safe_message = self._escape_log_html(message)
        self.log_area.append(
            f'<span style="color:{color};">[{timestamp}] {safe_message}</span>'
        )
        # Auto-scroll to bottom
        scrollbar = self.log_area.verticalScrollBar()
        if scrollbar:
            scrollbar.setValue(scrollbar.maximum())

    def clear_log(self):
        """Clear the log area"""
        self.log_area.clear()
        self.log_message("Log cleared")

    def _tile_download_counts(self) -> tuple[int, int, int]:
        """Return (expected, successful, failed) tile counts for the current run."""
        expected = len(self.tiles_to_download)
        successful = getattr(self, "success_count", 0)
        failed = max(0, expected - successful)
        return expected, successful, failed

    def _all_tiles_downloaded(self) -> bool:
        expected, successful, _ = self._tile_download_counts()
        return expected > 0 and successful >= expected

    def _log_incomplete_tiles_error(self, context: str = "") -> None:
        """Log an error when one or more tiles failed to download."""
        expected, successful, failed = self._tile_download_counts()
        if failed <= 0:
            return
        prefix = f"{context}: " if context else ""
        self.log_message(
            f"{prefix}Not all tiles were downloaded — {successful}/{expected} succeeded, "
            f"{failed} failed. The result may contain gaps."
        )

    def _get_current_aoi(self):
        """Return current Area of Interest as (west, east, south, north)."""
        return (
            float(self.west_spin.value()),
            float(self.east_spin.value()),
            float(self.south_spin.value()),
            float(self.north_spin.value()),
        )

    def _push_aoi_to_history(self, aoi):
        """
        Push an AOI into the 'past' stack, keeping at most 10 entries
        and clearing the 'future' stack (invalidating forward history).
        """
        if aoi is None:
            return
        if len(self.aoi_past) > 0 and self.aoi_past[-1] == aoi:
            return
        self.aoi_past.append(aoi)
        if len(self.aoi_past) > 10:
            self.aoi_past.pop(0)
        # New navigation path invalidates forward history
        self.aoi_future.clear()
        if getattr(self, "zoom_previous_btn", None) is not None:
            self.zoom_previous_btn.setEnabled(bool(self.aoi_past))
        if getattr(self, "zoom_next_btn", None) is not None:
            self.zoom_next_btn.setEnabled(bool(self.aoi_future))

    def zoom_to_previous(self):
        """Zoom to the most recent previous Area of Interest (back)."""
        if not self.aoi_past:
            self.log_message("No previous Area of Interest in history")
            return
        # Current AOI becomes part of the 'future' stack
        current = self._get_current_aoi()
        if current is not None:
            self.aoi_future.append(current)
            if len(self.aoi_future) > 10:
                self.aoi_future.pop(0)
        prev_west, prev_east, prev_south, prev_north = self.aoi_past.pop()

        # Block signals while applying previous AOI
        self.west_spin.blockSignals(True)
        self.east_spin.blockSignals(True)
        self.south_spin.blockSignals(True)
        self.north_spin.blockSignals(True)

        self.west_spin.setValue(prev_west)
        self.east_spin.setValue(prev_east)
        self.south_spin.setValue(prev_south)
        self.north_spin.setValue(prev_north)

        self.west_spin.blockSignals(False)
        self.east_spin.blockSignals(False)
        self.south_spin.blockSignals(False)
        self.north_spin.blockSignals(False)

        # Update map and pixel estimate after programmatic changes
        self.update_map_preview()
        self.update_estimated_pixel_count()
        self.log_message(
            f"Zoomed to previous AOI: West={prev_west:.4f}, East={prev_east:.4f}, "
            f"South={prev_south:.4f}, North={prev_north:.4f}"
        )

        # Update buttons based on history
        if getattr(self, "zoom_previous_btn", None) is not None:
            self.zoom_previous_btn.setEnabled(bool(self.aoi_past))
        if getattr(self, "zoom_next_btn", None) is not None:
            self.zoom_next_btn.setEnabled(bool(self.aoi_future))

    def zoom_to_next(self):
        """Zoom to the next Area of Interest (forward), if available."""
        if not self.aoi_future:
            self.log_message("No next Area of Interest in history")
            return
        # Current AOI becomes part of the 'past' stack
        current = self._get_current_aoi()
        if current is not None:
            self.aoi_past.append(current)
            if len(self.aoi_past) > 10:
                self.aoi_past.pop(0)
        next_west, next_east, next_south, next_north = self.aoi_future.pop()

        self.west_spin.blockSignals(True)
        self.east_spin.blockSignals(True)
        self.south_spin.blockSignals(True)
        self.north_spin.blockSignals(True)

        self.west_spin.setValue(next_west)
        self.east_spin.setValue(next_east)
        self.south_spin.setValue(next_south)
        self.north_spin.setValue(next_north)

        self.west_spin.blockSignals(False)
        self.east_spin.blockSignals(False)
        self.south_spin.blockSignals(False)
        self.north_spin.blockSignals(False)

        self.update_map_preview()
        self.update_estimated_pixel_count()
        self.log_message(
            f"Zoomed to next AOI: West={next_west:.4f}, East={next_east:.4f}, "
            f"South={next_south:.4f}, North={next_north:.4f}"
        )

        # Update buttons based on history
        if getattr(self, "zoom_previous_btn", None) is not None:
            self.zoom_previous_btn.setEnabled(bool(self.aoi_past))
        if getattr(self, "zoom_next_btn", None) is not None:
            self.zoom_next_btn.setEnabled(bool(self.aoi_future))

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
            960: 0.0192,  # 960m grids: 0.0192 degrees
            480: 0.0096,  # 480m grids: 0.0096 degrees
            240: 0.0048,  # 240m grids: 0.0048 degrees
            120: 0.0024,  # 120m grids: 0.0024 degrees
            60: 0.0012    # 60m grids: 0.0012 degrees
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

        mres_value = float(self.mres_combo.currentText())
        tile_size = tile_size_degrees_for_mres(mres_value)
        self.log_message(
            f"Tile size at {mres_value:.0f} m resolution: {tile_size:g}° × {tile_size:g}°"
        )

        lon_tiles = build_degree_strips(west, east, tile_size, overlap)
        lat_tiles = build_degree_strips(south, north, tile_size, overlap)
        
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

    def _open_download_progress(self, title: str, message: str = "", maximum: int = 100) -> None:
        if self.download_progress is None:
            self.download_progress = DownloadProgressDialog(self)
        self.download_progress.begin(title, message, maximum)

    def _update_download_progress(
        self,
        value: int | None = None,
        message: str | None = None,
        *,
        maximum: int | None = None,
        indeterminate: bool = False,
    ) -> None:
        if self.download_progress is not None:
            self.download_progress.update_progress(
                value, message, maximum=maximum, indeterminate=indeterminate
            )

    def _close_download_progress(self) -> None:
        if self.download_progress is not None:
            self.download_progress.close()
            self.download_progress = None
        self._download_progress_mode = None

    def _disconnect_download_worker_signals(self) -> None:
        worker = self.current_worker
        if worker is None:
            return
        try:
            worker.progress.disconnect(self._on_download_byte_progress)
        except (TypeError, RuntimeError):
            pass
        try:
            worker.finished.disconnect(self.on_tile_download_finished)
        except (TypeError, RuntimeError):
            pass
        try:
            worker.finished.disconnect(self.on_single_download_finished)
        except (TypeError, RuntimeError):
            pass

    def _stop_tile_download_watchdog(self) -> None:
        self.tile_download_watchdog.stop()

    def _start_tile_download_watchdog(self) -> None:
        self._stop_tile_download_watchdog()
        self.tile_download_watchdog.start(TILE_DOWNLOAD_WATCHDOG_MS)

    def _on_tile_download_watchdog(self) -> None:
        """Abort a tile download that has not finished within the watchdog window."""
        worker = self.current_worker
        if worker is None or not worker.isRunning():
            return
        self.log_message("Tile download watchdog: request exceeded time limit")
        self._disconnect_download_worker_signals()
        worker.terminate()
        worker.wait(3000)
        self.current_worker = None
        self.on_tile_download_finished(
            False, "Request Timeout - The server took too long to respond"
        )

    def _on_download_byte_progress(self, received: int, total: int) -> None:
        if self.download_progress is None:
            return
        if received > 0 and self.download_progress.progress_bar.maximum() == 0:
            self.download_progress.progress_bar.setRange(0, 100)
        if self._download_progress_mode == "tiles" and self.tiles_to_download:
            n = len(self.tiles_to_download)
            i = self.current_tile_index
            if total > 0:
                frac = (i + received / total) / n
                pct = min(99, int(frac * 100))
                mb_done = received / (1024 * 1024)
                mb_total = total / (1024 * 1024)
                msg = f"Tile {i + 1}/{n}: {mb_done:.1f} / {mb_total:.1f} MB"
            else:
                pct = min(99, int(i / n * 100))
                msg = f"Tile {i + 1}/{n}: downloading..."
            self._update_download_progress(pct, msg)
        elif self._download_progress_mode == "single":
            if total > 0:
                pct = min(99, int(received * 100 / total))
                mb_done = received / (1024 * 1024)
                mb_total = total / (1024 * 1024)
                self._update_download_progress(
                    pct, f"Downloading: {mb_done:.1f} / {mb_total:.1f} MB ({pct}%)"
                )
            else:
                self._update_download_progress(
                    message="Downloading grid from GMRT server..."
                )

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
            self.log_message("Waiting for previous download worker to finish...")
            if not self.current_worker.wait(5000):
                self.log_message("Previous download worker did not finish; stopping it")
                self._disconnect_download_worker_signals()
                self.current_worker.terminate()
                self.current_worker.wait(3000)
            self.current_worker = None

        dest_dir = os.path.dirname(os.path.abspath(filename))
        if dest_dir:
            os.makedirs(dest_dir, exist_ok=True)

        # Log the request parameters for debugging
        param_str = ", ".join([f"{k}={v}" for k, v in params.items()])
        request_url = format_gmrt_grid_request_url(params)
        self.log_message(f"Making request (will download as GeoTIFF): {param_str}")
        self.log_message(f"GMRT URL: {request_url}")
        self.log_message(f"Saving to: {filename}")
        if dest_dir:
            self.log_message(
                f"Request log: {os.path.join(dest_dir, GMRT_LOG_FILENAME)}"
            )

        self.current_worker = DownloadWorker(
            params, filename, requested_format, log_dir=dest_dir
        )
        self.current_worker.progress.connect(
            self._on_download_byte_progress, Qt.ConnectionType.QueuedConnection
        )
        if callback:
            self.current_worker.finished.connect(
                callback, Qt.ConnectionType.QueuedConnection
            )
        self.current_worker.start()
        if self._download_progress_mode == "tiles":
            self._start_tile_download_watchdog()
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
            self._sync_download_button_for_tile_limit()
            return
        
        try:
            self.log_message("Starting download process...")
        except Exception as e:
            self.status_label.setText(f"Log error: {str(e)}")
            self.download_btn.setText("Download Grid")
            self._sync_download_button_for_tile_limit()
            return
        
        try:
            self.log_message("Button clicked - processing request...")
        except Exception as e:
            self.status_label.setText(f"Log error: {str(e)}")
            self.download_btn.setText("Download Grid")
            self._sync_download_button_for_tile_limit()
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
            self._sync_download_button_for_tile_limit()
            QMessageBox.warning(self, "Invalid Coordinates", 
                              f"East coordinate ({east}) must be greater than West coordinate ({west}).\n\n"
                              f"Please correct the coordinates and try again.")
            return
        
        if north <= south:
            self.log_message(f"Error: North ({north}) must be greater than South ({south})")
            self.status_label.setText("Error: North must be greater than South")
            self.download_btn.setText("Download Grid")
            self._sync_download_button_for_tile_limit()
            QMessageBox.warning(self, "Invalid Coordinates", 
                              f"North coordinate ({north}) must be greater than South coordinate ({south}).\n\n"
                              f"Please correct the coordinates and try again.")
            return
        
        # Validate latitude range (GMRT data limit)
        if south < -85.0 or south > 85.0:
            self.log_message(f"Error: South latitude ({south}) must be between -85° and 85°")
            self.status_label.setText("Error: South latitude out of range")
            self.download_btn.setText("Download Grid")
            self._sync_download_button_for_tile_limit()
            QMessageBox.warning(self, "Invalid Coordinates", 
                              f"South latitude ({south}°) must be between -85° and 85° (GMRT data limit).\n\n"
                              f"Please correct the coordinates and try again.")
            return
        
        if north < -85.0 or north > 85.0:
            self.log_message(f"Error: North latitude ({north}) must be between -85° and 85°")
            self.status_label.setText("Error: North latitude out of range")
            self.download_btn.setText("Download Grid")
            self._sync_download_button_for_tile_limit()
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

        lon_span = abs(east - west)
        lat_span = abs(north - south)
        mres_value = float(self.mres_combo.currentText())
        tile_size = tile_size_degrees_for_mres(mres_value)
        span_sum_limit = 2.0 * tile_size
        needs_tiling = needs_tiling_for_spans(lon_span, lat_span, mres_value)

        n_lon, n_lat, n_tiles, _, _ = self._compute_tile_counts(
            west, east, south, north, mres_value
        )
        if n_tiles > MAX_TILES_PER_DOWNLOAD:
            self.log_message(
                f"ERROR: Tile count {n_tiles} exceeds maximum of "
                f"{MAX_TILES_PER_DOWNLOAD} ({n_lon}×{n_lat})"
            )
            self.status_label.setText(
                f"Too many tiles ({n_tiles}): reduce area or use coarser resolution"
            )
            self.download_btn.setText("Download Grid")
            self.download_btn.setEnabled(False)
            QMessageBox.warning(
                self,
                "Too Many Tiles",
                f"This area would require {n_tiles} tiles ({n_lon}×{n_lat}), "
                f"which exceeds the limit of {MAX_TILES_PER_DOWNLOAD}.\n\n"
                "Reduce the area of interest or choose a coarser cell resolution, "
                "then try again.",
            )
            return

        if needs_tiling:
            self.log_message(
                f"Tiling required: lon+lat span {lon_span + lat_span:.2f}° exceeds "
                f"{span_sum_limit:g}° limit ({tile_size:g}° tiles at {mres_value:.0f} m resolution)"
            )
            self.log_message(
                f"  (lon: {lon_span:.2f}°, lat: {lat_span:.2f}°; threshold is sum > 2×tile size)"
            )
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
                    self._sync_download_button_for_tile_limit()
                    self.status_label.setText("")
                    return
                self.save_last_download_dir(dir_path)
                os.makedirs(dir_path, exist_ok=True)
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
                self._tile_retry_count = 0
                self.downloaded_tile_files = []  # Reset tile files list

                mres_log = float(self.mres_combo.currentText())
                init_gmrt_log_session(
                    dir_path,
                    (
                        f"Tiled download — {len(tiles)} tiles, layer={layer_type}, "
                        f"mresolution={mres_log:g} m, output format={format_type}"
                    ),
                )
                self.log_message(
                    f"Tile requests will be logged to {os.path.join(dir_path, GMRT_LOG_FILENAME)}"
                )
                
                self.download_btn.setEnabled(False)  # Disable button during download
                self._download_progress_mode = "tiles"
                self._open_download_progress(
                    "Downloading Tiles",
                    f"Downloading 1 of {len(tiles)} tiles...",
                )
                self.download_next_tile()
        else:
            # Single download (AOI fits one tile at this resolution)
            self.log_message(
                f"Single file download: lon+lat span {lon_span + lat_span:.2f}° is within "
                f"{span_sum_limit:g}° limit ({tile_size:g}° tile at {mres_value:.0f} m)"
            )
            self.log_message("Opening directory selection dialog...")
            dir_path = QFileDialog.getExistingDirectory(self, "Select Directory for Grid File", self.last_download_dir)
            if not dir_path:
                self.log_message("Directory selection cancelled")
                self.download_btn.setText("Download Grid")
                self._sync_download_button_for_tile_limit()
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

            self._download_progress_mode = "single"
            self._open_download_progress("Downloading Grid", "Connecting to GMRT server...")
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
        
        tile_num = self.current_tile_index + 1
        n_tiles = len(self.tiles_to_download)
        if self._tile_retry_count > 0:
            self.log_message(
                f"Retry {self._tile_retry_count}/{TILE_DOWNLOAD_MAX_RETRIES} for tile "
                f"{tile_num}/{n_tiles}: {tile_filename}"
            )
        else:
            self.log_message(f"Starting download of tile {tile_num}/{n_tiles}: {tile_filename}")
        
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

        n = len(self.tiles_to_download)
        i = self.current_tile_index
        self._update_download_progress(
            message=f"Connecting — tile {i + 1}/{n}: {tile_filename}",
            indeterminate=True,
        )
        started = self.download_single_grid(
            params, tile_path, self.on_tile_download_finished, self.format_type
        )
        if not started:
            self.log_message(
                f"ERROR: Could not start download for tile {tile_num}/{n_tiles}"
            )
            QTimer.singleShot(
                0,
                lambda: self.on_tile_download_finished(
                    False, "Failed to start download worker"
                ),
            )

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
        self._stop_tile_download_watchdog()
        self._disconnect_download_worker_signals()
        self.current_worker = None
        self.download_btn.setText("Download Grid")
        self._sync_download_button_for_tile_limit()
        if success:
            self._update_download_progress(
                message="Processing downloaded file...", indeterminate=True
            )
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
            self._update_download_progress(100, "Download complete.")
            self._close_download_progress()
            QMessageBox.information(self, "Success", f"Grid downloaded to:\n{result}")
        else:
            self.log_message(f"Download failed: {result}")
            self.status_label.setText("Download failed.")
            self._close_download_progress()
            QMessageBox.critical(self, "Error", f"Failed to download grid:\n{result}")

    def on_tile_download_finished(self, success, result):
        """Callback for tile download completion"""
        self._stop_tile_download_watchdog()
        self._disconnect_download_worker_signals()
        self.current_worker = None
        try:
            tile_num = self.current_tile_index + 1
            if success:
                self._tile_retry_count = 0
                self.success_count += 1
                self.downloaded_tile_files.append(result)  # Track downloaded file
                self.log_message(
                    f"Tile {tile_num} downloaded successfully: {os.path.basename(result)}"
                )
            else:
                self.log_message(f"Tile {tile_num} failed: {result}")
                if self._tile_retry_count < TILE_DOWNLOAD_MAX_RETRIES:
                    self._tile_retry_count += 1
                    delay_s = TILE_DOWNLOAD_RETRY_DELAY_MS // 1000
                    self.log_message(
                        f"Retrying tile {tile_num} in {delay_s}s "
                        f"(attempt {self._tile_retry_count + 1}/"
                        f"{TILE_DOWNLOAD_MAX_RETRIES + 1})..."
                    )
                    self._update_download_progress(
                        message=(
                            f"Tile {tile_num} failed; retry "
                            f"{self._tile_retry_count}/{TILE_DOWNLOAD_MAX_RETRIES} in {delay_s}s..."
                        )
                    )
                    QTimer.singleShot(TILE_DOWNLOAD_RETRY_DELAY_MS, self.download_next_tile)
                    return

                self.log_message(
                    f"Tile {tile_num} failed after {TILE_DOWNLOAD_MAX_RETRIES + 1} attempts; skipping"
                )
                self._tile_retry_count = 0

            # Move to next tile
            self.current_tile_index += 1
            n = len(self.tiles_to_download)
            done = self.current_tile_index
            if self.download_progress is not None:
                pct = min(99, int(done / n * 100)) if n else 0
                self._update_download_progress(
                    pct, f"Finished {done}/{n} tiles; preparing next..."
                )

            if self.current_tile_index < len(self.tiles_to_download):
                self.log_message(
                    f"Waiting {TILE_DOWNLOAD_INTER_TILE_DELAY_MS // 1000} seconds "
                    f"before downloading tile {self.current_tile_index + 1}"
                )
                QTimer.singleShot(TILE_DOWNLOAD_INTER_TILE_DELAY_MS, self.download_next_tile)
            else:
                # All tile attempts finished — mosaic if any succeeded
                expected, successful, failed = self._tile_download_counts()
                self.log_message(
                    f"Tile download pass finished: {successful}/{expected} succeeded"
                )
                self.log_message(f"Downloaded tile files: {self.downloaded_tile_files}")

                if self.downloaded_tile_files:
                    if failed > 0:
                        self._log_incomplete_tiles_error("Before mosaicking")
                        self.log_message(
                            f"Starting mosaic from {successful} downloaded tile(s) only"
                        )
                    else:
                        self.log_message(
                            "All tiles downloaded, automatically starting mosaicking process..."
                        )
                    QTimer.singleShot(3000, self.start_mosaicking)
                else:
                    self.log_message(
                        "ERROR: No tiles were downloaded successfully; skipping mosaic"
                    )
                    QTimer.singleShot(100, self.finish_tile_download)
                    
        except Exception as e:
            import traceback
            self.log_message(f"ERROR in on_tile_download_finished: {str(e)}")
            self.log_message(f"Traceback: {traceback.format_exc()}")
            QTimer.singleShot(100, self.finish_tile_download)
    
    def start_mosaicking(self):
        """Start the mosaicking process after a delay"""
        try:
            self.log_message("=== START_MOSAICKING CALLED ===")
            self.status_label.setText("Mosaicking tiles into single GeoTIFF...")
            self.status_label.repaint()
            self._download_progress_mode = "mosaic"
            self._update_download_progress(
                90, "Mosaicking downloaded tiles...", maximum=100
            )
            self.log_message("Starting mosaicking worker thread...")

            # Use rasterio for mosaicking
            self.log_message("Using rasterio for mosaicking")

            # Create and start the mosaic worker thread
            # Note: delete_tiles_checkbox is None since tiles are always deleted after mosaicking
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
            self.mosaic_worker.progress.connect(self.on_mosaic_progress)
            self.mosaic_worker.finished.connect(self.on_mosaic_finished)
            self.mosaic_worker.start()
            
        except Exception as e:
            import traceback
            self.log_message(f"ERROR in start_mosaicking: {str(e)}")
            self.log_message(f"Traceback: {traceback.format_exc()}")
            self.finish_tile_download()
    
    def on_mosaic_progress(self, message):
        """Handle progress updates from mosaic worker"""
        self.log_message(f"Mosaic: {message}")
        self.status_label.setText(f"Mosaicking: {message}")
        self.status_label.repaint()
        self._update_download_progress(92, f"Mosaicking: {message}")
    
    def on_mosaic_finished(self, success, result):
        """Handle completion of mosaic worker"""
        try:
            self.log_message(f"Mosaic worker finished - Success: {success}, Result: {result}")
            
            if success:
                self.log_message("✓ Mosaicking completed successfully")
                self.status_label.setText("Mosaicking completed successfully")
                
                # Get the mosaic path from the worker
                if hasattr(self.mosaic_worker, 'mosaic_path') and self.mosaic_worker.mosaic_path:
                    mosaic_path = self.mosaic_worker.mosaic_path
                    
                    # Always delete individual tile files after mosaicking
                    deleted_count = 0
                    for tile_file in self.downloaded_tile_files:
                        try:
                            if os.path.exists(tile_file):
                                os.remove(tile_file)
                                deleted_count += 1
                        except Exception as e:
                            pass
                    
                    # Get format type
                    format_type = self.format_type
                    
                    # Apply splitting to the mosaicked file if requested
                    if self.split_checkbox.isChecked():
                        self.log_message("Applying split to mosaicked file...")
                        self.split_grid_file(mosaic_path, format_type)
                    # Convert to NetCDF if format is NetCDF and split is not enabled
                    elif format_type == 'netcdf':
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
                    self.download_btn.setText("Download Grid")
                    self._sync_download_button_for_tile_limit()
                    self.log_message(f"Mosaicking completed successfully: {os.path.basename(mosaic_path)}")
                    self.status_label.setText(f"Mosaic complete: {os.path.basename(mosaic_path)} in {self.download_dir}")
                    self._update_download_progress(100, "Mosaic complete.")
                    self._close_download_progress()
                    expected, successful, failed = self._tile_download_counts()
                    if failed > 0:
                        self._log_incomplete_tiles_error("After mosaicking")
                        self.status_label.setText(
                            f"Mosaic incomplete: {successful}/{expected} tiles in {self.download_dir}"
                        )
                        QMessageBox.warning(
                            self,
                            "Incomplete Download",
                            f"Mosaic saved to:\n{mosaic_path}\n\n"
                            f"Only {successful} of {expected} tiles downloaded successfully "
                            f"({failed} failed). The mosaic may contain gaps.",
                        )
                    else:
                        QMessageBox.information(
                            self,
                            "Success",
                            f"Mosaicked {successful} tiles into:\n{mosaic_path}",
                        )
                else:
                    self.log_message("ERROR: No mosaic path found")
                    self.finish_tile_download()
            else:
                self.log_message(f"✗ Mosaicking failed: {result}")
                self.status_label.setText("Mosaicking failed")
                self.download_btn.setText("Download Grid")
                self._sync_download_button_for_tile_limit()
                self._close_download_progress()
                QMessageBox.critical(self, "Mosaicking Error", f"Mosaicking failed:\n{result}")
                self.finish_tile_download()
                
        except Exception as e:
            import traceback
            self.log_message(f"ERROR in on_mosaic_finished: {str(e)}")
            self.log_message(f"Traceback: {traceback.format_exc()}")
            self.finish_tile_download()
    
    

    def finish_tile_download(self):
        """Finish the tile download process without mosaicking"""
        self.download_btn.setText("Download Grid")
        self._sync_download_button_for_tile_limit()
        expected, successful, failed = self._tile_download_counts()
        self.log_message(f"Tile download finished: {successful}/{expected} successful")
        self._close_download_progress()

        if successful == 0:
            self.log_message("ERROR: No tiles were downloaded successfully")
            self.status_label.setText("Download failed: no tiles saved")
            QMessageBox.critical(
                self,
                "Download Failed",
                "No tiles were downloaded successfully.",
            )
        elif failed > 0:
            self._log_incomplete_tiles_error("Download finished")
            self.status_label.setText(
                f"Download incomplete: {successful}/{expected} tiles in {self.download_dir}"
            )
            QMessageBox.warning(
                self,
                "Incomplete Download",
                f"Only {successful} of {expected} tiles downloaded successfully "
                f"({failed} failed).\n\nFiles saved to:\n{self.download_dir}",
            )
        else:
            self.status_label.setText(
                f"Download complete: {successful}/{expected} tiles in {self.download_dir}"
            )
            QMessageBox.information(
                self,
                "Success",
                f"Downloaded {successful}/{expected} tiles to:\n{self.download_dir}",
            )

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
                        except Exception as e:
                            pass
                except Exception as e:
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
            self.download_btn.setText("Download Grid")
            self._sync_download_button_for_tile_limit()
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

        self._download_progress_mode = "single"
        self._open_download_progress("Downloading Grid", "Connecting to GMRT server...")
        self.download_btn.setEnabled(False)  # Disable button during download
        self.download_single_grid(params, file_name, self.on_single_download_finished, format_type)

    def on_rectangle_selected(self, bounds):
        # Save current AOI before switching to the new one
        self._push_aoi_to_history(self._get_current_aoi())
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

        self.map_widget.clear_selection()
        # Update map and pixel estimate after programmatic changes (signals were blocked)
        self.update_map_preview()  # Only one call now
        self.update_estimated_pixel_count()
        self.log_message(f"Rectangle selected: {bounds}")

    def zoom_to_default(self):
        """
        Zoom the map to the starting map defaults.
        """
        # Block signals to prevent multiple map updates
        # Save current AOI before switching to defaults
        self._push_aoi_to_history(self._get_current_aoi())
        self.west_spin.blockSignals(True)
        self.east_spin.blockSignals(True)
        self.south_spin.blockSignals(True)
        self.north_spin.blockSignals(True)
        self.west_spin.setValue(-180.0)
        self.east_spin.setValue(180.0)
        self.south_spin.setValue(-83.0)
        self.north_spin.setValue(85.0)
        self.west_spin.blockSignals(False)
        self.east_spin.blockSignals(False)
        self.south_spin.blockSignals(False)
        self.north_spin.blockSignals(False)
        # Update map and pixel estimate after programmatic changes (signals were blocked)
        self.update_map_preview()
        self.update_estimated_pixel_count()
        self.log_message("Zoomed to starting map defaults")

