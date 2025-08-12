@echo off
:: BiRefNet-CLI – GPU/CPU auto-setup
cd /d "%~dp0"

:: ---------- Проверка/установка Python ----------
python -version >nul 2>&1
if errorlevel 1 (
    powershell -NoP -Command "Invoke-WebRequest https://www.python.org/ftp/python/3.11.8/python-3.11.8-amd64.exe -OutFile %TEMP%\py.exe"
    %TEMP%\py.exe /quiet InstallAllUsers=1 PrependPath=1
)

:: ---------- Проверка/установка FFmpeg ----------
ffmpeg -version >nul 2>&1
if errorlevel 1 (
    powershell -NoP -Command "Invoke-WebRequest https://www.gyan.dev/ffmpeg/builds/ffmpeg-release-essentials.zip -OutFile %TEMP%\ff.zip"
    powershell -NoP -Command "Expand-Archive -Force %TEMP%\ff.zip C:\"
    ren "C:\ffmpeg-*" "C:\FFmpeg" >nul 2>&1
    setx /M PATH "%PATH%;C:\FFmpeg\bin"
)

:: ---------- Виртуальное окружение ----------
if not exist "venv\Scripts\activate.bat" python -m venv venv
call venv\Scripts\activate.bat
python -m pip install --upgrade pip >nul

:: ---------- Авто-выбор PyTorch ----------
:: Проверяем CUDA-драйвер через nvidia-smi
nvidia-smi >nul 2>&1
if not errorlevel 1 (
    pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
) else (
    pip install torch torchvision
)

pip install -r requirements.txt
echo ===== DONE =====
pause
