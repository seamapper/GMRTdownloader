"""
Entry point for GMRT Bathymetry Grid Downloader.
"""

import sys
import signal

from PyQt6.QtWidgets import QApplication
from ui.main_window import GMRTGrabber


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
        app = QApplication(sys.argv)
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
