@echo off
:: =============================================================================
:: Industrial Vision System — Environment Setup Script
:: Run ONCE before first deployment
:: Requires Python 3.10+, CUDA 11.8+ drivers
:: =============================================================================

echo ============================================================
echo Industrial Vision System v2.0 — Setup
echo ============================================================

:: Check Python
python --version
if errorlevel 1 (
    echo ERROR: Python not found. Install Python 3.10+
    pause
    exit /b 1
)

:: Create virtual environment
echo Creating virtual environment...
python -m venv venv
call venv\Scripts\activate.bat

:: Upgrade pip
python -m pip install --upgrade pip

:: Install CUDA PyTorch (CUDA 11.8)
echo Installing PyTorch with CUDA 11.8...
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118

:: Install other dependencies
echo Installing other dependencies...
pip install numpy opencv-python PyYAML psutil ultralytics pynvml

:: Try USB relay library
pip install pyhid-usb-relay hid 2>nul

:: Create log directory
mkdir logs 2>nul
mkdir models 2>nul

echo.
echo ============================================================
echo Setup complete!
echo.
echo NEXT STEPS:
echo 1. Copy your YOLO model to: models\best.pt
echo 2. Edit config\config.yaml with your camera RTSP URLs
echo 3. Edit boundaries\camera_X_boundaries.json for each camera
echo 4. Run: python main.py
echo ============================================================
pause
