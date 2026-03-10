@echo off
:: =============================================================================
:: Industrial Vision System v2.1 — Windows Service Installer
:: Requirements: NSSM (Non-Sucking Service Manager)
::
:: NSSM INSTALL STEPS (do ONCE before running this script):
::   1. Download from: https://nssm.cc/download
::   2. Extract nssm.exe to C:\Windows\System32\  (or any PATH folder)
::   3. Verify: open cmd, type "nssm" — should show usage help
::
:: Run THIS script as Administrator.
:: =============================================================================

set SERVICE_NAME=IndustrialVisionSystem
set PYTHON_PATH=C:\Python310\python.exe
set APP_DIR=%~dp0..
set MAIN_SCRIPT=%APP_DIR%\main.py
set CONFIG_PATH=%APP_DIR%\config\config.yaml
set LOG_DIR=%APP_DIR%\logs

:: Auto-detect Python if not at default path
if not exist "%PYTHON_PATH%" (
    for /f "tokens=*" %%i in ('where python 2^>nul') do set PYTHON_PATH=%%i
    if "%PYTHON_PATH%"=="" (
        echo ERROR: Python not found. Set PYTHON_PATH manually in this script.
        pause & exit /b 1
    )
)

:: Verify NSSM is available
where nssm >nul 2>&1
if errorlevel 1 (
    echo ERROR: nssm.exe not found in PATH.
    echo.
    echo FIX:
    echo   1. Download from: https://nssm.cc/download
    echo   2. Copy nssm.exe to C:\Windows\System32\
    echo   3. Re-run this script as Administrator
    pause & exit /b 1
)

echo Python: %PYTHON_PATH%
echo App dir: %APP_DIR%
echo Service: %SERVICE_NAME%

:: Create log dir
mkdir "%LOG_DIR%" 2>nul

:: Remove old service
echo Removing existing service (if any)...
nssm stop %SERVICE_NAME% 2>nul
nssm remove %SERVICE_NAME% confirm 2>nul
timeout /t 2 /nobreak >nul

:: Install
echo Installing service...
nssm install %SERVICE_NAME% "%PYTHON_PATH%"
nssm set %SERVICE_NAME% AppParameters "%MAIN_SCRIPT% --config %CONFIG_PATH%"
nssm set %SERVICE_NAME% AppDirectory "%APP_DIR%"
nssm set %SERVICE_NAME% AppStdout "%LOG_DIR%\service_stdout.log"
nssm set %SERVICE_NAME% AppStderr "%LOG_DIR%\service_stderr.log"
nssm set %SERVICE_NAME% AppRotateFiles 1
nssm set %SERVICE_NAME% AppRotateBytes 10485760
nssm set %SERVICE_NAME% Start SERVICE_AUTO_START
nssm set %SERVICE_NAME% ObjectName LocalSystem
nssm set %SERVICE_NAME% AppThrottle 10000
nssm set %SERVICE_NAME% AppRestartDelay 15000
nssm set %SERVICE_NAME% DisplayName "Industrial Vision Detection System"
nssm set %SERVICE_NAME% Description "24/7 Multi-Camera YOLO Detection with Relay Control"

echo.
echo Starting service...
nssm start %SERVICE_NAME%

echo.
echo =============================================
echo Service installed and started: %SERVICE_NAME%
echo Logs: %LOG_DIR%
echo.
echo To manage:
echo   nssm start   %SERVICE_NAME%
echo   nssm stop    %SERVICE_NAME%
echo   nssm restart %SERVICE_NAME%
echo   nssm status  %SERVICE_NAME%
echo =============================================
pause
