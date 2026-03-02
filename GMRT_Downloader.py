"""
GMRT Bathymetry Grid Downloader - Launcher

A PyQt6-based GUI application for downloading bathymetry data from the
Global Multi-Resolution Topography (GMRT) synthesis.

This file is the entry point for the application and for the PyInstaller build.
Implementation is split across: main.py, config.py, workers/, converters.py, ui/.

Credit: Ryan, W.B.F., S.M. Carbotte, J.O. Coplan, S. O'Hara, A. Melkonian, R. Arko,
R.A. Weissel, V. Ferrini, A. Goodwillie, F. Nitsche, J. Bonczkowski, and R. Zemsky (2009),
Global Multi-Resolution Topography synthesis, Geochem. Geophys. Geosyst., 10, Q03014,
doi: 10.1029/2008GC002332

Author: Paul Johnson
"""

from main import main

if __name__ == "__main__":
    main()
