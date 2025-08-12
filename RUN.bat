@echo off
chcp 65001 >nul
title BiRefNet Launcher

:menu
cls
echo ================================
echo   Выберите скрипт для запуска
echo ================================
echo  1  -  birefnet_cli_LUMA.py
echo  2  -  birefnet_cli_PRORES.py
echo  0  -  Выход
echo ================================
set /p choice="Введите цифру и нажмите Enter: "

if "%choice%"=="1" (
    echo Запуск birefnet_cli_LUMA.py...
    python birefnet_cli_LUMA.py
    pause
    goto menu
)

if "%choice%"=="2" (
    echo Запуск birefnet_cli_PRORES.py...
    python birefnet_cli_PRORES.py
    pause
    goto menu
)

if "%choice%"=="0" exit

echo Неверный выбор, попробуйте ещё раз.
timeout /t 2 >nul
goto menu
