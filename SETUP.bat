@echo off
:: ============================================================
::  BiRefNet Universal Setup
::  Устанавливает зависимости для обоих скриптов:
::  birefnet_cli_LUMA.py   и   birefnet_cli_PRORES.py
:: ============================================================

:: Переключаем кодовую страницу на русскую без проблем с UTF-8
chcp 1251 >nul

title BiRefNet Setup

:: Определяем путь к текущей папке
setlocal enabledelayedexpansion
set "DIR=%~dp0"

echo ==========================================
echo      BiRefNet Setup
echo ==========================================
echo Устанавливаются зависимости из requirements.txt...
echo.

:: Проверяем наличие Python
where python >nul 2>nul
if errorlevel 1 (
    echo [ОШИБКА] Python не найден в PATH.
    echo Установите Python 3.8+ и добавьте его в переменные среды.
    pause
    exit /b 1
)

:: Обновляем pip
echo [1/3] Обновление pip...
python -m pip install --upgrade pip

:: Устанавливаем зависимости
echo [2/3] Установка пакетов из requirements.txt...
python -m pip install -r "%DIR%requirements.txt"

:: Сообщение об успешном завершении
echo.
echo ==========================================
echo      Готово! Все зависимости установлены.
echo ==========================================
echo Теперь запустите RUN.bat для выбора скрипта.
echo.
pause
