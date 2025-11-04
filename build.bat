@echo off
REM Build script for GMRT Downloader Windows executable
REM This script automates the build process using PyInstaller

echo ========================================
echo GMRT Downloader - Windows Build Script
echo ========================================
echo.

REM Check if virtual environment exists and activate it
if exist venv\Scripts\activate.bat (
    echo Activating virtual environment...
    call venv\Scripts\activate.bat
) else (
    echo Warning: Virtual environment not found. Using system Python.
    echo.
)

REM Check if PyInstaller is installed
python -c "import PyInstaller" 2>nul
if errorlevel 1 (
    echo Error: PyInstaller is not installed!
    echo Please install it with: pip install pyinstaller
    pause
    exit /b 1
)

echo Cleaning previous builds...
if exist build (
    rmdir /s /q build
    echo Build directory cleaned.
)
if exist dist (
    rmdir /s /q dist
    echo Dist directory cleaned.
)
echo.

REM Check if spec file exists
if not exist GMRT_Downloader.spec (
    echo Error: GMRT_Downloader.spec file not found!
    echo Please create the spec file first.
    pause
    exit /b 1
)

echo Building executable...
echo.
pyinstaller GMRT_Downloader.spec

REM Check if build was successful
if exist dist\GMRT_Bathymetry_Downloader.exe (
    echo.
    echo ========================================
    echo Build successful!
    echo ========================================
    echo.
    echo Executable location: dist\GMRT_Bathymetry_Downloader.exe
    echo.
    echo To test the executable, run:
    echo   dist\GMRT_Bathymetry_Downloader.exe
    echo.
) else (
    echo.
    echo ========================================
    echo Build failed!
    echo ========================================
    echo.
    echo Please check the error messages above.
    echo.
    pause
    exit /b 1
)

pause

