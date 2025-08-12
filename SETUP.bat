@echo off
REM Установка Python-зависимостей для BiRefNet проекта
echo Установка зависимостей для BiRefNet...

REM Проверка наличия Python
python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo Ошибка: Python не найден. Установите Python 3.7+ и добавьте в PATH.
    pause
    exit /b 1
)

REM Проверка версии Python
python -c "import sys; exit(1) if sys.version_info < (3, 7) else exit(0)"
if %errorlevel% neq 0 (
    echo Ошибка: Требуется Python 3.7 или выше.
    pause
    exit /b 1
)

REM Проверка наличия pip
python -m pip --version >nul 2>&1
if %errorlevel% neq 0 (
    echo Ошибка: pip не найден. Убедитесь что pip установлен.
    pause
    exit /b 1
)

REM Удаление потенциально конфликтующих версий
echo Очистка возможных конфликтующих версий...
python -m pip uninstall torch torchvision torchaudio -y
python -m pip uninstall transformers accelerate -y

REM Установка совместимых версий
echo Установка совместимых версий библиотек...
python -m pip install --upgrade pip
python -m pip install torch==2.1.1+cu118 torchvision==0.16.1+cu118 --index-url https://download.pytorch.org/whl/cu118
python -m pip install transformers==4.35.0
python -m pip install accelerate==0.24.1
python -m pip install einops kornia timm
python -m pip install opencv-python==4.8.1.78 Pillow==10.0.1 numpy==1.24.4 tqdm==4.66.1
python -m pip install safetensors

REM Опциональные зависимости для GUI
echo Установка опциональных зависимостей для GUI...
python -m pip install tk

echo Все зависимости успешно установлены!
echo Теперь вы можете использовать birefnet_cli_PRORES.py и birefnet_cli_LUMA.py
pause
