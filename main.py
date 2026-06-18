# Copyright (c) 2026 Paul Johnson
# SPDX-License-Identifier: MIT

"""
Entry point for GMRT Bathymetry Grid Downloader.
"""

import os
import sys
import signal

from PyQt6.QtWidgets import QApplication, QStyleFactory
from PyQt6.QtGui import QPalette, QColor
from PyQt6.QtCore import Qt, QCoreApplication
from ui.main_window import GMRTGrabber


def _configure_qt_plugins():
    """Point Qt at bundled plugins when running as a frozen Mac .app."""
    if getattr(sys, "frozen", False) and sys.platform == "darwin":
        base = getattr(sys, "_MEIPASS", os.path.dirname(os.path.abspath(__file__)))
        plugins = os.path.join(base, "PyQt6", "Qt6", "plugins")
        if os.path.isdir(plugins):
            QCoreApplication.setLibraryPaths([plugins])


def apply_light_mode(app):
    """Force light mode regardless of system theme."""
    if "Fusion" in QStyleFactory.keys():
        app.setStyle("Fusion")
    palette = QPalette()
    palette.setColor(QPalette.ColorRole.Window, QColor(240, 240, 240))
    palette.setColor(QPalette.ColorRole.WindowText, Qt.GlobalColor.black)
    palette.setColor(QPalette.ColorRole.Base, QColor(255, 255, 255))
    palette.setColor(QPalette.ColorRole.AlternateBase, QColor(233, 233, 233))
    palette.setColor(QPalette.ColorRole.ToolTipBase, QColor(255, 255, 220))
    palette.setColor(QPalette.ColorRole.ToolTipText, Qt.GlobalColor.black)
    palette.setColor(QPalette.ColorRole.Text, Qt.GlobalColor.black)
    palette.setColor(QPalette.ColorRole.PlaceholderText, QColor(120, 120, 120))
    palette.setColor(QPalette.ColorRole.Button, QColor(240, 240, 240))
    palette.setColor(QPalette.ColorRole.ButtonText, Qt.GlobalColor.black)
    palette.setColor(QPalette.ColorRole.BrightText, Qt.GlobalColor.red)
    palette.setColor(QPalette.ColorRole.Link, QColor(0, 102, 204))
    palette.setColor(QPalette.ColorRole.Highlight, QColor(42, 130, 218))
    palette.setColor(QPalette.ColorRole.HighlightedText, Qt.GlobalColor.white)
    palette.setColor(
        QPalette.ColorGroup.Disabled, QPalette.ColorRole.WindowText, QColor(140, 140, 140)
    )
    palette.setColor(
        QPalette.ColorGroup.Disabled, QPalette.ColorRole.Text, QColor(140, 140, 140)
    )
    palette.setColor(
        QPalette.ColorGroup.Disabled, QPalette.ColorRole.ButtonText, QColor(140, 140, 140)
    )
    app.setPalette(palette)


def signal_handler(signum, frame):
    """Handle signals to catch crashes"""
    print(f"Received signal {signum}")
    import traceback
    traceback.print_stack(frame)
    sys.exit(1)


def main():
    try:
        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)

        print("Starting GMRT Bathymetry Grid Downloader...")
        _configure_qt_plugins()
        app = QApplication(sys.argv)
        apply_light_mode(app)
        print("QApplication created successfully")

        win = GMRTGrabber()
        print("GMRTGrabber window created successfully")

        win.show()
        print("Window shown successfully")
        win.raise_()
        win.activateWindow()

        print("Starting application event loop...")
        sys.exit(app.exec())
    except Exception as e:
        print(f"Error starting application: {str(e)}")
        import traceback
        traceback.print_exc()
        input("Press Enter to exit...")


if __name__ == "__main__":
    main()
