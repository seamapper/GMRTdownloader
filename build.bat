@echo off
REM Build script for GMRT Downloader Windows executable
REM This script automates the build process using PyInstaller

echo ========================================
echo GMRT Downloader - Windows Build Script
echo ========================================
echo.

REM Check if virtual environment exists and set Python path
set PYTHON_CMD=python
if exist .venv\Scripts\python.exe (
    echo Using virtual environment .venv...
    set PYTHON_CMD=.venv\Scripts\python.exe
) else (
    if exist venv\Scripts\python.exe (
        echo Using virtual environment venv...
        set PYTHON_CMD=venv\Scripts\python.exe
    ) else (
        if exist ..\MultibeamToolsMolokai\.venv\Scripts\python.exe (
            echo Using virtual environment from MultibeamToolsMolokai...
            set PYTHON_CMD=..\MultibeamToolsMolokai\.venv\Scripts\python.exe
        ) else (
            echo Using system Python interpreter...
            echo Note: Make sure Python is configured correctly in your IDE.
        )
    )
)

REM Check if PyInstaller is installed
%PYTHON_CMD% -c "import PyInstaller" 2>nul
if errorlevel 1 (
    echo Error: PyInstaller is not installed!
    echo Please install it with: %PYTHON_CMD% -m pip install pyinstaller
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
%PYTHON_CMD% -m PyInstaller GMRT_Downloader.spec

REM Check if build was successful (check for versioned executable)
set BUILD_SUCCESS=0
for %%f in (dist\GMRT_Bathymetry_Downloader_v*.exe) do (
    echo.
    echo ========================================
    echo Build successful!
    echo ========================================
    echo.
    echo Executable location: dist\%%~nxf
    echo.
    echo To test the executable, run:
    echo   dist\%%~nxf
    echo.
    set BUILD_SUCCESS=1
    goto :build_check_done
)
:build_check_done
if %BUILD_SUCCESS%==0 (
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

