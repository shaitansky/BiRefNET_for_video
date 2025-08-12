@echo off
chcp 1251 >nul
title BiRefNet Launcher

:: Переходим в папку, где лежит сам .bat
cd /d "%~dp0"

:menu
cls
echo ================================
echo   Choose script to run
echo ================================
echo 1 - birefnet_cli_LUMA.py
echo 2 - birefnet_cli_PRORES.py
echo 0 - Exit
echo ================================
set /p choice=Enter number and press Enter: 

if "%choice%"=="1" (
    python birefnet_cli_LUMA.py
) else if "%choice%"=="2" (
    python birefnet_cli_PRORES.py
) else if "%choice%"=="0" (
    exit
) else (
    echo Invalid choice. Try again.
    pause
    goto menu
)

pause
