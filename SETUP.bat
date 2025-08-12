@echo off
:: ============================================================
::  BiRefNet Video CLI – установка с авто-выбором CPU/GPU
::  Запускать: ПКМ → «Запуск от имени администратора»
:: ============================================================
chcp 65001 >nul
title BiRefNet-CLI – Auto Setup (CPU/GPU)

:: ---------- Переход в папку скрипта ----------
cd /d "%~dp0"

:: ---------- Проверка прав администратора ----------
net session >nul 2>&1
if %errorLevel% neq 0 (
    echo [ОШИБКА] Запустите скрипт через «ПКМ → Запуск от имени администратора»
    pause
    exit /b 1
)

:: ---------- Проверка / установка Python ----------
python --version >nul 2>&1
if %errorLevel% neq 0 (
    echo [ИНФО] Python не найден. Скачиваем и устанавливаем...
    powershell -NoP -Command "Invoke-WebRequest https://www.python.org/ftp/python/3.11.8/python-3.11.8-amd64.exe -OutFile %TEMP%\python_installer.exe"
    %TEMP%\python_installer.exe /quiet InstallAllUsers=1 PrependPath=1
    set "PATH=%PATH%;C:\Python311\;C:\Python311\Scripts\"
    del %TEMP%\python_installer.exe
) else (
    echo [OK] Python уже установлен.
)

:: ---------- Проверка / установка FFmpeg ----------
ffmpeg -version >nul 2>&1
if %errorLevel% neq 0 (
    echo [ИНФО] FFmpeg не найден. Скачиваем и устанавливаем...
    powershell -NoP -Command "Invoke-WebRequest https://www.gyan.dev/ffmpeg/builds/ffmpeg-release-essentials.zip -OutFile %TEMP%\ffmpeg.zip"
    powershell -NoP -Command "Expand-Archive -Force %TEMP%\ffmpeg.zip C:\"
    for /f "tokens=*" %%i in ('dir /b C:\ffmpeg-*') do move /y "C:\%%i" "C:\FFmpeg" >nul
    setx /M PATH "%PATH%;C:\FFmpeg\bin"
    set "PATH=%PATH%;C:\FFmpeg\bin"
    del %TEMP%\ffmpeg.zip
) else (
    echo [OK] FFmpeg уже установлен.
)

:: ---------- Создание виртуального окружения ----------
if not exist "venv\Scripts\activate.bat" (
    echo [ИНФО] Создаём виртуальное окружение venv...
    python -m venv venv
)

:: ---------- Активация venv ----------
call venv\Scripts\activate.bat
python -m pip install --upgrade pip >nul

:: ---------- Авто-выбор CPU vs CUDA ----------
nvidia-smi >nul 2>&1
if %errorLevel%==0 (
    echo [ИНФО] Обнаружена NVIDIA GPU. Устанавливаем CUDA-версию PyTorch...
    pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
) else (
    echo [ИНФО] GPU не обнаружена. Устанавливаем CPU-версию PyTorch...
    pip install torch torchvision
)

:: ---------- Остальные зависимости ----------
echo [ИНФО] Устанавливаем остальные пакеты...
pip install -r requirements.txt
if %errorLevel% neq 0 (
    echo [ОШИБКА] Не удалось установить зависимости
    pause
    exit /b 1
)

:: ---------- Готово ----------
echo.
echo ==========================================================
echo ВСЁ ГОТОВО!
echo Для запуска скрипта дважды кликните RUN.bat
echo или введите:
echo   venv\Scripts\activate
echo   python birefnet_cli.py
echo ==========================================================
pause
