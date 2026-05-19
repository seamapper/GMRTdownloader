# Copyright (c) 2026 Paul Johnson
# SPDX-License-Identifier: MIT

"""
Worker thread for loading map preview images from GMRT ImageServer.
"""

from PyQt6.QtCore import QThread, pyqtSignal
from PyQt6.QtGui import QPixmap, QImage
import requests

from config import GMRT_IMAGE_URL, MAP_PREVIEW_REQUEST_TIMEOUT


class MapWorker(QThread):
    """
    Worker thread for loading map preview images from GMRT ImageServer.
    Runs in a separate thread to prevent the GUI from freezing.
    """
    map_loaded = pyqtSignal(QPixmap)
    map_error = pyqtSignal(str)

    def __init__(self, west, east, south, north, width=800, mask=True):
        super().__init__()
        self.west = west
        self.east = east
        self.south = south
        self.north = north
        self.width = width
        self.mask = mask

    def run(self):
        try:
            params = {
                "minlongitude": self.west,
                "maxlongitude": self.east,
                "minlatitude": self.south,
                "maxlatitude": self.north,
                "width": self.width,
                "mask": "1" if self.mask else "0"
            }
            response = requests.get(
                GMRT_IMAGE_URL, params=params, timeout=MAP_PREVIEW_REQUEST_TIMEOUT
            )
            if response.status_code == 200:
                if not response.content:
                    self.map_error.emit("Failed to load map: empty response")
                    return
                image = QImage()
                if not image.loadFromData(response.content):
                    self.map_error.emit("Failed to load map: invalid image data")
                    return
                pixmap = QPixmap.fromImage(image)
                if pixmap.isNull():
                    self.map_error.emit("Failed to load map: could not decode image")
                    return
                self.map_loaded.emit(pixmap)
            else:
                self.map_error.emit(f"Failed to load map: HTTP {response.status_code}")
        except Exception as e:
            self.map_error.emit(f"Map loading error: {str(e)}")
