@echo off
chcp 1251 >nul
title BiRefNet Launcher

:: Переходим в папку, где лежит сам .bat
cd /d "%~dp0"

:: Проверка наличия Python
python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo Ошибка: Python не найден. Установите Python 3.7+ и добавьте в PATH.
    pause
    exit /b 1
)

:menu
cls
echo ================================
echo   Choose script to run
echo ================================
echo 1 - birefnet_cli_LUMA.py
echo 2 - birefnet_cli_PRORES.py
echo 3 - Setup (install dependencies)
echo 0 - Exit
echo ================================
set /p choice=Enter number and press Enter: 

if "%choice%"=="1" (
    echo Running birefnet_cli_LUMA.py...
    python birefnet_cli_LUMA.py
    if %errorlevel% neq 0 (
        echo Error running birefnet_cli_LUMA.py
        echo Try running setup first (option 3)
    )
) else if "%choice%"=="2" (
    echo Running birefnet_cli_PRORES.py...
    python birefnet_cli_PRORES.py
    if %errorlevel% neq 0 (
        echo Error running birefnet_cli_PRORES.py
        echo Try running setup first (option 3)
    )
) else if "%choice%"=="3" (
    call setup.bat
) else if "%choice%"=="0" (
    exit
) else (
    echo Invalid choice. Try again.
    pause
    goto menu
)

pause
goto menu
