BiRefNet Video CLI

Лёгкий способ убрать фон из видео без API и токенов

Этот инструмент позволяет легко удалить фон с видео без необходимости использования API или токенов. Он работает оффлайн и поддерживает улучшенный антиалиасинг для устранения «лесенок» на краях.

![Python](https://img.shields.io/badge/python-3.8+-blue.svg)
![OS](https://img.shields.io/badge/platform-Windows%7CLinux%7CmacOS-lightgrey.svg)
![License](https://img.shields.io/badge/license-MIT-green.svg)
![GPU](https://img.shields.io/badge/GPU-CUDA%20optional-blue.svg)

---

Что умеет скрипт

* Открывает видео через графический интерфейс или командную строку.
* Работает полностью локально после первого запуска.
* Сохраняет длительность, разрешение и частоту кадров исходного видео.
* Применяет улучшенный антиалиасинг и unsharp mask для сглаживания границ.
* Создаёт видео с прозрачным или белым фоном одной командой.
* Поддерживает запуск на CPU и GPU (флаг --no-gpu).
* Не требует регистрации или API-ключей.

---

Как начать

1. Клонируйте репозиторий:

git clone [https://github.com/shaitansky/BiRefNET_for_video.git]
cd BiRefNet-Video-CLI  
2. Создайте виртуальное окружение (рекомендуется):

python -m venv venv  
source venv/bin/activate  
# Для Windows: venv\Scripts\activate  
3. Установите зависимости:

pip install -r requirements.txt  
4. Запустите скрипт:
* Для запуска через GUI:

  python birefnet_cli.py  
* Или укажите путь к видео:

  python birefnet_cli.py -i path/to/video.mp4  

Результат будет сохранён рядом с исходным файлом: video_birefnet_processed_with_audio.mp4.

---

Примеры

Исходное видео:
![Demo before](https://user-images.githubusercontent.com/YOUR_ID/demo_before.gif)
Результат:
![Demo after](https://user-images.githubusercontent.com/YOUR_ID/demo_after.gif)

---

Системные требования

* Python: 3.8+
* PyTorch: 2.0+ (CUDA 11.7+ опционально)
* FFmpeg: для копирования аудиодорожки
* ОС: Windows 10/11, Ubuntu 20.04+, macOS 12+

Для установки FFmpeg на Ubuntu:

sudo apt update && sudo apt install ffmpeg  

Для Windows: скачайте официальный билд и добавьте ffmpeg.exe в PATH.

---

Аргументы командной строки

* -i, --input PATH — путь к видеофайлу (по умолчанию открывается GUI).
* --no-gpu — принудительно запустить на CPU.

---

Структура проекта

BiRefNet-Video-CLI/  
├── birefnet_cli.py       # основной скрипт  
├── requirements.txt      # список зависимостей  
├── LICENSE                # лицензия MIT  
├── README.md              # этот файл  
├── .gitignore  
└── assets/  
    ├── demo_before.gif     # примеры для README  
    └── demo_after.gif

---

Зависимости и авторы

* Библиотеки:
  * transformers (Hugging Face) — Apache-2.0
  * torch (PyTorch Team) — BSD
  * torchvision (PyTorch Team) — BSD-3-Clause
  * opencv-python (OpenCV) — Apache-2.0
  * Pillow (Alex Clark & Contributors) — PIL
  * numpy (NumPy Developers) — BSD-3-Clause
  * tqdm (tqdm Team) — MPL-2.0 & MIT
* Модель BiRefNet:
  Автор: ZhengPeng7
  Лицензия: Creative Commons Attribution-NonCommercial 4.0

---

Часто задаваемые вопросы

Q: Сколько VRAM нужно для работы на GPU?
≈ 2 ГБ для видео 1024×1024.

Q: Можно ли изменить выходной формат?
Скрипт сохраняет видео в формате MP4. Для других форматов замените строку fourcc в функции process_video(...).

Q: Как использовать собственный фон вместо белого?
Откройте birefnet_cli.py, найдите строку:

white_bg = np.full_like(frame, 255, dtype=np.float32)

И замените white_bg на изображение нужного размера.

---

Лицензия: MIT © 2025 SHaitansky. Подробности в файле LICENSE.

Благодарности:

* ZhengPeng7 за архитектуру BiRefNet.
* Hugging Face за удобную экосистему transformers.
* Сообщества PyTorch, OpenCV и FFmpeg за их инструменты.
