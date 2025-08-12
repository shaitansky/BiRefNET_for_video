@echo off
:: ============================================================
::  BiRefNet Video CLI – полная автоматическая установка
::  Требует запуска «от имени администратора»
:: ============================================================
title BiRefNet-CLI – Full Auto-Setup (Admin)

:: ---------- Проверка, что мы действительно админ ----------
net session >nul 2>&1
if %errorLevel% neq 0 (
    echo [ERROR] Запустите скрипт ПКМ → «Запуск от имени администратора»
    pause
    exit /b 1
)

:: ---------- Проверка / установка Python ----------
python --version >nul 2>&1
if %errorLevel% neq 0 (
    echo [INFO] Python не найден. Скачиваем и устанавливаем...
    powershell -Command "Invoke-WebRequest https://www.python.org/ftp/python/3.11.8/python-3.11.8-amd64.exe -OutFile %TEMP%\python_installer.exe"
    %TEMP%\python_installer.exe /quiet InstallAllUsers=1 PrependPath=1
    set "PATH=%PATH%;C:\Python311\;C:\Python311\Scripts\"
    del %TEMP%\python_installer.exe
) else (
    echo [OK] Python уже установлен.
)

:: ---------- Проверка / установка FFmpeg ----------
ffmpeg -version >nul 2>&1
if %errorLevel% neq 0 (
    echo [INFO] FFmpeg не найден. Скачиваем и устанавливаем...
    powershell -Command "Invoke-WebRequest https://www.gyan.dev/ffmpeg/builds/ffmpeg-release-essentials.zip -OutFile %TEMP%\ffmpeg.zip"
    powershell -Command "Expand-Archive -Force %TEMP%\ffmpeg.zip C:\"
    ren "C:\ffmpeg-*" "C:\FFmpeg"
    setx /M PATH "%PATH%;C:\FFmpeg\bin"
    set "PATH=%PATH%;C:\FFmpeg\bin"
    del %TEMP%\ffmpeg.zip
) else (
    echo [OK] FFmpeg уже установлен.
)

:: ---------- Создание виртуального окружения ----------
if not exist "venv\Scripts\activate.bat" (
    echo [INFO] Создаём виртуальное окружение venv...
    python -m venv venv
)

:: ---------- Установка pip-зависимостей ----------
call venv\Scripts\activate.bat
python -m pip install --upgrade pip >nul
echo [INFO] Устанавливаем Python-пакеты...
pip install -r requirements.txt
if %errorLevel% neq 0 (
    echo [ERROR] Ошибка при установке пакетов
    pause
    exit /b 1
)

:: ---------- Готово ----------
echo.
echo ==========================================================
echo ВСЁ ГОТОВО!
echo Чтобы запустить скрипт, дважды кликните RUN.bat
echo или введите:
echo   venv\Scripts\activate
echo   python birefnet_cli.py
echo ==========================================================
pause
