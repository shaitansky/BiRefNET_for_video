@echo off
:: BiRefNet-CLI – Auto Setup (Admin)
title BiRefNet-CLI Setup

:: switch to script folder
cd /d "%~dp0"

:: check admin rights
net session >nul 2>&1
if %errorLevel% neq 0 (
    echo ERROR: Run as Administrator
    pause
    exit /b 1
)

:: Python check/install
python --version >nul 2>&1
if %errorLevel% neq 0 (
    echo INFO: Installing Python...
    powershell -NoP -Command "Invoke-WebRequest https://www.python.org/ftp/python/3.11.8/python-3.11.8-amd64.exe -OutFile %TEMP%\py.exe"
    %TEMP%\py.exe /quiet InstallAllUsers=1 PrependPath=1
    del %TEMP%\py.exe
)

:: FFmpeg check/install
ffmpeg -version >nul 2>&1
if %errorLevel% neq 0 (
    echo INFO: Installing FFmpeg...
    powershell -NoP -Command "Invoke-WebRequest https://www.gyan.dev/ffmpeg/builds/ffmpeg-release-essentials.zip -OutFile %TEMP%\ff.zip"
    powershell -NoP -Command "Expand-Archive -Force %TEMP%\ff.zip C:\"
    ren "C:\ffmpeg-*" "C:\FFmpeg" >nul 2>&1
    setx /M PATH "%PATH%;C:\FFmpeg\bin"
)

:: venv + deps
if not exist "venv\Scripts\activate.bat" (
    python -m venv venv
)
call venv\Scripts\activate.bat
python -m pip install --upgrade pip >nul

:: auto CPU vs GPU
nvidia-smi >nul 2>&1
if %errorLevel%==0 (
    pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
) else (
    pip install torch torchvision
)

pip install -r requirements.txt
echo.
echo ===== DONE =====
echo run RUN.bat
pause
