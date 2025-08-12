#!/usr/bin/env python3
"""
BiRefNet CLI (облегчённая версия с улучшенным антиалиасингом):
- GUI выбор файлов
- Поддержка русских символов
- Выходное видео всегда того же размера что и исходное
- Улучшенный антиалиасинг для устранения "лесенок"
- Открытый доступ без токенов или API
"""

import os
import sys
import cv2
import torch
import numpy as np
import subprocess
import argparse
from pathlib import Path
from PIL import Image, ImageEnhance
from tqdm import tqdm
from torchvision import transforms
from transformers import AutoModelForImageSegmentation

# GUI imports
try:
    import tkinter as tk
    from tkinter import filedialog, messagebox
    GUI_AVAILABLE = True
except ImportError:
    GUI_AVAILABLE = False
    print("Внимание: tkinter недоступен, GUI отключен")

# Константы
MODEL_ID = "ZhengPeng7/BiRefNet"
IMAGE_SIZE = 1024


class VideoProcessor:
    def __init__(self, use_gpu: bool = True):
        self.device = torch.device("cuda" if use_gpu and torch.cuda.is_available() else "cpu")

        # Папки для моделей
        self.model_dir = Path(__file__).parent / "models"
        self.birefnet_dir = self.model_dir / "BiRefNet"

        self.model_dir.mkdir(exist_ok=True)
        self.birefnet_dir.mkdir(exist_ok=True)

        print(f"Устройство: {self.device}")

        self._load_models()

    def _load_models(self):
        """Загрузка BiRefNet"""
        print("Загрузка BiRefNet...")

        torch.set_float32_matmul_precision('high')

        self.birefnet = AutoModelForImageSegmentation.from_pretrained(
            MODEL_ID,
            trust_remote_code=True,
            cache_dir=str(self.birefnet_dir)
        ).to(self.device).eval()

        if self.device.type == 'cuda':
            self.birefnet.half()

        self.transform = transforms.Compose([
            transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

    def select_video_file(self) -> Path:
        """Выбор видеофайла через GUI или консоль"""
        if GUI_AVAILABLE:
            root = tk.Tk()
            root.withdraw()
            root.attributes('-topmost', True)

            file_path = filedialog.askopenfilename(
                title="Выберите видеофайл",
                filetypes=[
                    ("Видео файлы", "*.mp4 *.avi *.mov *.mkv *.wmv *.flv *.webm"),
                    ("MP4 файлы", "*.mp4"),
                    ("AVI файлы", "*.avi"),
                    ("Все файлы", "*.*")
                ],
                initialdir=str(Path.home())
            )
            root.destroy()

            if not file_path:
                print("Файл не выбран")
                sys.exit(0)

            return Path(file_path)
        else:
            while True:
                try:
                    path_str = input("Введите путь к видеофайлу (или перетащите файл): ").strip()
                    path_str = path_str.strip('"').strip("'")

                    file_path = Path(path_str)
                    if file_path.exists() and file_path.is_file():
                        return file_path
                    else:
                        print(f"Файл не найден: {file_path}")

                except KeyboardInterrupt:
                    print("\nОтменено пользователем")
                    sys.exit(0)
                except Exception as e:
                    print(f"Ошибка: {e}")

    def sharp_resize_mask(self, mask_1024: np.ndarray, target_size: tuple) -> np.ndarray:
        """
        Масштабирование маски 1024×1024 до целевого разрешения с улучшенным антиалиасингом.
        :param mask_1024: float32 0-1 1024×1024
        :param target_size: (w, h) — целевое разрешение
        :return: float32 0-1 w×h
        """
        h_tgt, w_tgt = target_size[1], target_size[0]

        # 1. Мягкое повышение резкости (ослабленное, чтобы избежать усиления "лесенок")
        kernel = np.array([[0, -0.5, 0],
                           [-0.5, 3, -0.5],
                           [0, -0.5, 0]], dtype=np.float32)
        sharp = cv2.filter2D(mask_1024.astype(np.float32), -1, kernel)
        sharp = np.clip(sharp, 0, 1)

        # 2. Supersampling ×2 с Lanczos для лучшего сглаживания
        ss_h, ss_w = h_tgt * 2, w_tgt * 2
        supersampled = cv2.resize(sharp, (ss_w, ss_h), interpolation=cv2.INTER_LANCZOS4)

        # 3. Downsample до целевого размера
        down = cv2.resize(supersampled, (w_tgt, h_tgt), interpolation=cv2.INTER_AREA)
        return np.clip(down, 0, 1).astype(np.float32)

    def apply_fxaa_like(self, mask: np.ndarray) -> np.ndarray:
        """
        Улучшенный FXAA-подобный антиалиасинг для 1-канальной float32 маски.
        Работает с float32 для сохранения точности.
        """
        mask = mask.astype(np.float32)

        # 1. Edge detection
        grad_x = cv2.Sobel(mask, cv2.CV_32F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(mask, cv2.CV_32F, 0, 1, ksize=3)
        edge_magnitude = np.abs(grad_x) + np.abs(grad_y)
        edge_magnitude = edge_magnitude / (np.max(edge_magnitude) + 1e-6)
        edge = (edge_magnitude > 0.25).astype(np.float32)

        # 2. Morphological closing
        kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
        edge_closed = cv2.morphologyEx(edge, cv2.MORPH_CLOSE, kernel, iterations=1)

        # 3. Gaussian blur
        blurred = cv2.GaussianBlur(mask, (5, 5), 1.0)
        alpha = edge_closed
        smoothed = mask * (1 - alpha) + blurred * alpha

        # 4. Unsharp mask (перенесён сюда)
        smoothed_32 = smoothed.astype(np.float32)
        blur = cv2.GaussianBlur(smoothed_32, (0, 0), 1.5)
        unsharp = cv2.addWeighted(smoothed_32, 1.5, blur, -0.5, 0)

        return np.clip(unsharp, 0, 1).astype(np.float32)

    def create_mask_1024(self, frame: np.ndarray) -> np.ndarray:
        """Создание маски 1024×1024 с легким сглаживанием для предотвращения "лесенок"."""
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pil = Image.fromarray(rgb)
        pil = ImageEnhance.Contrast(pil).enhance(1.35)
        pil = ImageEnhance.Sharpness(pil).enhance(1.4)

        tensor = self.transform(pil).unsqueeze(0).to(self.device)

        if self.device.type == 'cuda':
            tensor = tensor.half()

        with torch.no_grad():
            pred = self.birefnet(tensor)[-1].sigmoid().cpu()[0].squeeze().numpy()

        # Приведение к float32
        pred = pred.astype(np.float32)

        # Легкое сглаживание
        pred = cv2.GaussianBlur(pred, (3, 3), 0.5)
        return np.clip(pred, 0, 1)

    def process_video(self, input_path: Path, output_suffix: str = "_LUMA"):
        """Полный цикл обработки с улучшенным антиалиасингом."""
        output_path = input_path.parent / f"{input_path.stem}{output_suffix}.mp4"

        print(f"Входной файл: {input_path}")
        print(f"Выходной файл: {output_path}")

        # 1. Открываем видео и читаем параметры
        cap = cv2.VideoCapture(str(input_path))
        if not cap.isOpened():
            raise ValueError(f"Не удается открыть видео: {input_path}")

        fps = int(cap.get(cv2.CAP_PROP_FPS))
        w_orig = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h_orig = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        print(f"Разрешение: {w_orig}×{h_orig}, FPS: {fps}, кадров: {total_frames}")

        # 2. Первый проход – создание масок 1024×1024
        print("Создание масок 1024×1024...")
        masks_1024 = []
        with torch.no_grad():
            for _ in tqdm(range(total_frames), desc="Маски"):
                ret, frame = cap.read()
                if not ret:
                    break
                mask = self.create_mask_1024(frame)
                masks_1024.append(mask)
        cap.release()

        # 3. Второй проход – апскейл + FXAA-like антиалиасинг
        print("Апскейл + FXAA-like...")
        full_masks = []
        for mask_1024 in tqdm(masks_1024, desc="Обработка"):
            sharp = self.sharp_resize_mask(mask_1024, (w_orig, h_orig))
            aa = self.apply_fxaa_like(sharp)
            full_masks.append(aa)

        # 4. Третий проход – композитинг
        cap = cv2.VideoCapture(str(input_path))
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        writer = cv2.VideoWriter(str(output_path), fourcc, fps, (w_orig, h_orig))

        print("Применение масок к кадрам...")
        for frame_idx in tqdm(range(len(full_masks)), desc="Композитинг"):
            ret, frame = cap.read()
            if not ret:
                break

            full_mask = full_masks[frame_idx]
            white_bg = np.full_like(frame, 255, dtype=np.float32)
            black_bg = np.zeros_like(frame, dtype=np.float32)
            result = (white_bg * full_mask[..., None] + black_bg * (1 - full_mask[..., None]))
            result = np.clip(result, 0, 255).astype(np.uint8)
            writer.write(result)

        cap.release()
        writer.release()

        # 5. Добавление аудио
        final_output = self._add_audio(input_path, output_path)
        print(f"✅ Обработка завершена: {final_output}")
        return final_output

    def _add_audio(self, input_path: Path, video_path: Path) -> Path:
        """Добавление аудиодорожки к обработанному видео"""
        audio_output = video_path.parent / f"{video_path.stem}_with_audio.mp4"

        try:
            cmd = [
                "ffmpeg", "-y",
                "-i", str(video_path),
                "-i", str(input_path),
                "-c:v", "copy",
                "-c:a", "aac",
                "-map", "0:v:0",
                "-map", "1:a:0?",
                str(audio_output)
            ]

            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                encoding='utf-8'
            )

            if result.returncode == 0:
                video_path.unlink()
                return audio_output
            else:
                print("Предупреждение: не удалось добавить аудио")
                return video_path

        except FileNotFoundError:
            print("Предупреждение: FFmpeg не найден, аудио не будет добавлено")
            return video_path
        except Exception as e:
            print(f"Ошибка при добавлении аудио: {e}")
            return video_path


def main():
    parser = argparse.ArgumentParser(description="BiRefNet Video Processor (облегчённая версия с улучшенным антиалиасингом)")
    parser.add_argument("--no-gpu", action="store_true", help="Отключить GPU")
    parser.add_argument("--input", "-i", type=str, help="Путь к входному видео")

    args = parser.parse_args()

    try:
        processor = VideoProcessor(use_gpu=not args.no_gpu)

        if args.input:
            input_path = Path(args.input)
            if not input_path.exists():
                print(f"Файл не найден: {input_path}")
                return
        else:
            input_path = processor.select_video_file()

        output_path = processor.process_video(input_path)

        if GUI_AVAILABLE:
            messagebox.showinfo("Готово", f"Обработка завершена!\nФайл сохранен: {output_path}")

    except KeyboardInterrupt:
        print("\nОбработка прервана пользователем")
    except Exception as e:
        error_msg = f"Ошибка: {e}"
        print(error_msg)
        if GUI_AVAILABLE:
            messagebox.showerror("Ошибка", error_msg)


if __name__ == "__main__":
    main()

