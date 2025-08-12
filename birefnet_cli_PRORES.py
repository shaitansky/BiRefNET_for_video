#!/usr/bin/env python3
"""
BiRefNet CLI (цветное видео + прозрачный фон, ProRes 4444):
- GUI выбор файлов
- Поддержка русских символов
- Выходное видео того же размера, что и исходное
- Альфа-канал (RGBA) + ProRes 4444 (.mov)
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
        h_tgt, w_tgt = target_size[1], target_size[0]
        kernel = np.array([[0, -0.5, 0],
                           [-0.5, 3, -0.5],
                           [0, -0.5, 0]], dtype=np.float32)
        sharp = cv2.filter2D(mask_1024.astype(np.float32), -1, kernel)
        sharp = np.clip(sharp, 0, 1)
        ss_h, ss_w = h_tgt * 2, w_tgt * 2
        supersampled = cv2.resize(sharp, (ss_w, ss_h), interpolation=cv2.INTER_LANCZOS4)
        down = cv2.resize(supersampled, (w_tgt, h_tgt), interpolation=cv2.INTER_AREA)
        return np.clip(down, 0, 1).astype(np.float32)

    def apply_fxaa_like(self, mask: np.ndarray) -> np.ndarray:
        mask = mask.astype(np.float32)
        grad_x = cv2.Sobel(mask, cv2.CV_32F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(mask, cv2.CV_32F, 0, 1, ksize=3)
        edge_magnitude = np.abs(grad_x) + np.abs(grad_y)
        edge_magnitude = edge_magnitude / (np.max(edge_magnitude) + 1e-6)
        edge = (edge_magnitude > 0.25).astype(np.float32)
        kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
        edge_closed = cv2.morphologyEx(edge, cv2.MORPH_CLOSE, kernel, iterations=1)
        blurred = cv2.GaussianBlur(mask, (5, 5), 1.0)
        alpha = edge_closed
        smoothed = mask * (1 - alpha) + blurred * alpha
        smoothed_32 = smoothed.astype(np.float32)
        blur = cv2.GaussianBlur(smoothed_32, (0, 0), 1.5)
        unsharp = cv2.addWeighted(smoothed_32, 1.5, blur, -0.5, 0)
        return np.clip(unsharp, 0, 1).astype(np.float32)

    def create_mask_1024(self, frame: np.ndarray) -> np.ndarray:
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pil = Image.fromarray(rgb)
        pil = ImageEnhance.Contrast(pil).enhance(1.35)
        pil = ImageEnhance.Sharpness(pil).enhance(1.4)

        tensor = self.transform(pil).unsqueeze(0).to(self.device)

        if self.device.type == 'cuda':
            tensor = tensor.half()

        with torch.no_grad():
            pred = self.birefnet(tensor)[-1].sigmoid().cpu()[0].squeeze().numpy()

        pred = pred.astype(np.float32)
        pred = cv2.GaussianBlur(pred, (3, 3), 0.5)
        return np.clip(pred, 0, 1)

    def process_video(self, input_path: Path, output_suffix: str = "_alpha"):
        output_path = input_path.parent / f"{input_path.stem}{output_suffix}.mov"
        print(f"Входной файл: {input_path}")
        print(f"Выходной файл: {output_path}")

        cap = cv2.VideoCapture(str(input_path))
        if not cap.isOpened():
            raise ValueError(f"Не удается открыть видео: {input_path}")

        fps = cap.get(cv2.CAP_PROP_FPS)
        w   = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h   = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        print(f"{w}×{h}, {fps} fps, {total_frames} кадров")

        # FFmpeg-пайп: ProRes 4444 (профиль 4) + 8-bit RGBA
        cmd = [
            "ffmpeg", "-y",
            "-f", "rawvideo",
            "-vcodec", "rawvideo",
            "-pix_fmt", "bgra",
            "-s", f"{w}x{h}",
            "-r", str(fps),
            "-i", "-",
            "-i", str(input_path),
            "-c:v", "prores_ks",
            "-profile:v", "4",       # 4444
            "-pix_fmt", "yuva444p10le",  # FFmpeg всё равно запишет 10-бит, но AE прочитает
            "-c:a", "copy",
            "-map", "0:v:0",
            "-map", "1:a:0?",
            "-shortest",
            str(output_path)
        ]
        proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stderr=subprocess.DEVNULL)

        # Маски 1024×1024
        masks_1024 = []
        with torch.no_grad():
            for _ in tqdm(range(total_frames), desc="Маски"):
                ret, frame = cap.read()
                if not ret:
                    break
                masks_1024.append(self.create_mask_1024(frame))
        cap.release()

        full_masks = [self.apply_fxaa_like(self.sharp_resize_mask(m, (w, h))) for m in
                    tqdm(masks_1024, desc="Обработка")]

        cap = cv2.VideoCapture(str(input_path))
        for mask in tqdm(full_masks, desc="Запись"):
            ret, frame = cap.read()
            if not ret:
                break
            alpha = (mask * 255).astype(np.uint8)     # 8-бит альфа
            bgra = cv2.cvtColor(frame, cv2.COLOR_BGR2BGRA)
            bgra[:, :, 3] = alpha                     # прозрачный фон
            proc.stdin.write(bgra.tobytes())

        cap.release()
        proc.stdin.close()
        proc.wait()

        print(f"✅ Готово (ProRes 4444 + alpha): {output_path}")
        return output_path

    # ---------- Вспомогательные методы ----------

    def _to_prores4444(self, temp_rgba: Path, output: Path) -> Path:
        """Перекодировка lossless RGBA → ProRes 4444"""
        cmd = [
            "ffmpeg", "-y",
            "-i", str(temp_rgba),
            "-c:v", "prores_ks",
            "-profile:v", "4",      # 4444
            "-pix_fmt", "yuva444p10le",
            "-vendor", "ap10",
            "-qscale:v", "20",       # качество 9-13 (меньше = лучше)
            str(output)
        ]
        subprocess.run(cmd, check=True)
        return output

    def _add_audio(self, input_path: Path, video_path: Path) -> Path:
        """Добавление аудио к ProRes-файлу"""
        audio_output = video_path.parent / f"{video_path.stem}_with_audio.mov"

        try:
            cmd = [
                "ffmpeg", "-y",
                "-i", str(video_path),
                "-i", str(input_path),
                "-c:v", "copy",
                "-c:a", "copy",
                "-map", "0:v:0",
                "-map", "1:a:0?",
                "-shortest",
                str(audio_output)
            ]
            subprocess.run(cmd, check=True)
            video_path.unlink()
            return audio_output
        except Exception as e:
            print(f"Предупреждение: не удалось добавить аудио ({e})")
            return video_path


def main():
    parser = argparse.ArgumentParser(description="BiRefNet Video Processor (ProRes 4444 + Alpha)")
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

        processor.process_video(input_path)

    except KeyboardInterrupt:
        print("\nОбработка прервана пользователем")
    except Exception as e:
        error_msg = f"Ошибка: {e}"
        print(error_msg)
        if GUI_AVAILABLE:
            messagebox.showerror("Ошибка", error_msg)


if __name__ == "__main__":
    main()

