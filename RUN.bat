@echo off
chcp 1251 >nul
title BiRefNet Launcher

:menu
cls
echo ================================
echo   Choose script to run
echo ================================
echo  1  -  birefnet_cli_LUMA.py
echo  2  -  birefnet_cli_PRORES.py
echo  0  -  Exit
echo ================================
set /p choice="Enter number and press Enter: "

if "%choice%"=="1" (
    echo Running birefnet_cli_LUMA.py...
    python birefnet_cli_LUMA.py
    pause
    goto menu
)

if "%choice%"=="2" (
    echo Running birefnet_cli_PRORES.py...
    python birefnet_cli_PRORES.py
    pause
    goto menu
)

if "%choice%"=="0" exit

echo Invalid choice, try again.
timeout /t 2 >nul
goto menu
