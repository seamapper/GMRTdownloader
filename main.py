# Copyright (c) 2026 Paul Johnson
# SPDX-License-Identifier: MIT

"""
Entry point for GMRT Bathymetry Grid Downloader.
"""

import os
import sys
import signal

from PyQt6.QtWidgets import QApplication
from PyQt6.QtGui import QPalette, QColor
from PyQt6.QtCore import Qt, QCoreApplication
from ui.main_window import GMRTGrabber

# Fusion + dark palette on macOS: native QComboBox popups can be invisible or
# non-clickable. combobox-popup: 0 forces a Qt-drawn list; explicit colors keep
# the list readable.
_COMBOBOX_STYLE = """
QComboBox {
    combobox-popup: 0;
    background-color: #454545;
    color: #ffffff;
    border: 1px solid #666666;
    border-radius: 4px;
    padding: 4px 8px;
    min-height: 1.5em;
}
QComboBox:hover {
    border-color: #888888;
}
QComboBox::drop-down {
    subcontrol-origin: padding;
    subcontrol-position: top right;
    width: 22px;
    border-left: 1px solid #666666;
}
QComboBox::down-arrow {
    image: none;
    width: 0;
    height: 0;
    border-left: 5px solid transparent;
    border-right: 5px solid transparent;
    border-top: 6px solid #cccccc;
}
QComboBox QAbstractItemView {
    background-color: #353535;
    color: #ffffff;
    selection-background-color: #2a82da;
    selection-color: #ffffff;
    border: 1px solid #666666;
    outline: none;
}
"""


def _configure_qt_plugins():
    """Point Qt at bundled plugins when running as a frozen Mac .app."""
    if getattr(sys, "frozen", False) and sys.platform == "darwin":
        base = getattr(sys, "_MEIPASS", os.path.dirname(os.path.abspath(__file__)))
        plugins = os.path.join(base, "PyQt6", "Qt6", "plugins")
        if os.path.isdir(plugins):
            QCoreApplication.setLibraryPaths([plugins])


def apply_dark_mode(app):
    """Force dark mode regardless of system theme."""
    app.setStyle("Fusion")
    app.setStyleSheet(_COMBOBOX_STYLE)
    palette = QPalette()
    # Base colors
    palette.setColor(QPalette.ColorRole.Window, QColor(53, 53, 53))
    palette.setColor(QPalette.ColorRole.WindowText, Qt.GlobalColor.white)
    palette.setColor(QPalette.ColorRole.Base, QColor(35, 35, 35))
    palette.setColor(QPalette.ColorRole.AlternateBase, QColor(53, 53, 53))
    palette.setColor(QPalette.ColorRole.Text, Qt.GlobalColor.white)
    palette.setColor(QPalette.ColorRole.PlaceholderText, QColor(160, 160, 160))
    palette.setColor(QPalette.ColorRole.Button, QColor(53, 53, 53))
    palette.setColor(QPalette.ColorRole.ButtonText, Qt.GlobalColor.white)
    # Highlights
    palette.setColor(QPalette.ColorRole.Highlight, QColor(42, 130, 218))
    palette.setColor(QPalette.ColorRole.HighlightedText, Qt.GlobalColor.white)
    # Disabled
    palette.setColor(QPalette.ColorGroup.Disabled, QPalette.ColorRole.WindowText, QColor(127, 127, 127))
    palette.setColor(QPalette.ColorGroup.Disabled, QPalette.ColorRole.Text, QColor(127, 127, 127))
    palette.setColor(QPalette.ColorGroup.Disabled, QPalette.ColorRole.ButtonText, QColor(127, 127, 127))
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
        apply_dark_mode(app)
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
