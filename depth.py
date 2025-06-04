import os
import cv2
import torch
import numpy as np
import warnings
import logging
from PIL import Image
from transformers import pipeline
from tqdm import tqdm

def run_depth_estimation(frames_dir: str = "./frames", depth_maps_dir: str = "./depth_maps"):
    # Отключаем предупреждения и логи уровня WARNING и ниже из transformers и torch
    warnings.filterwarnings("ignore")
    logging.getLogger("transformers").setLevel(logging.ERROR)
    logging.getLogger("torch").setLevel(logging.ERROR)

    os.makedirs(depth_maps_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("Загружаем модель для оценки глубины...")
    depth_pipe = pipeline(task="depth-estimation", model="xingyang1/Distill-Any-Depth-Large-hf", device=0 if device.type == "cuda" else -1)
    print("Модель для оценки глубины загружена.")

    frame_files = sorted([f for f in os.listdir(frames_dir) if f.endswith((".png", ".jpg"))])
    if not frame_files:
        raise RuntimeError(f"Нет кадров в {frames_dir}")

    with tqdm(total=len(frame_files), desc="Processing frames") as pbar:
        for idx, filename in enumerate(frame_files, 1):
            frame_path = os.path.join(frames_dir, filename)
            frame = cv2.imread(frame_path)
            if frame is None:
                print(f"Ошибка чтения {frame_path}")
                continue

            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image = Image.fromarray(frame_rgb)

            result = depth_pipe(image)
            depth = np.array(result["depth"])

            if depth.dtype != np.float32:
                depth = depth.astype(np.float32)

            max_depth = np.nanmax(depth)
            inverted_depth = max_depth - depth
            inverted_depth = np.nan_to_num(inverted_depth, nan=0.0)

            depth_name = os.path.splitext(filename)[0] + ".npy"
            np.save(os.path.join(depth_maps_dir, depth_name), inverted_depth)

            pbar.update(1)

    print(f"Вычисление глубины завершено. Карты сохранены в {depth_maps_dir}")

if __name__ == "__main__":
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    run_depth_estimation("./frames", "./depth_maps")
