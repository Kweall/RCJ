import os
import cv2
import torch
import numpy as np
import warnings
import logging
from transformers import DetrImageProcessor, DetrForSegmentation
from tqdm import tqdm
from ultralytics import YOLO

def run_segmentation(frames_dir: str = "./frames", seg_maps_dir: str = "./segmented_maps"):
    warnings.filterwarnings("ignore")
    logging.getLogger("transformers").setLevel(logging.ERROR)
    logging.getLogger("torch").setLevel(logging.ERROR)

    os.makedirs(seg_maps_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Используем устройство: {device}")

    print("Загружаем модель и процессор...")
    model_name = "facebook/detr-resnet-50-panoptic"
    processor = DetrImageProcessor.from_pretrained(model_name)
    model = DetrForSegmentation.from_pretrained(model_name).to(device).eval()
    print("Модель и процессор загружены.")

    print("Загрузка модели YOLO для распознавания знаков...")
    yolo_model = YOLO("yolo_signs.pt")
    target_classes = {1, 2, 4, 5}

    frame_files = sorted([f for f in os.listdir(frames_dir) if f.endswith((".png", ".jpg"))])
    if not frame_files:
        raise RuntimeError(f"Нет кадров в {frames_dir}")

    with tqdm(total=len(frame_files), desc="Segmenting frames") as pbar:
        for filename in frame_files:
            frame_path = os.path.join(frames_dir, filename)
            frame = cv2.imread(frame_path)
            if frame is None:
                print(f"Ошибка чтения {frame_path}")
                continue

            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            inputs = processor(images=frame_rgb, return_tensors="pt").to(device)

            with torch.no_grad():
                outputs = model(**inputs)

            target_sizes = [(frame_rgb.shape[0], frame_rgb.shape[1])]
            panoptic_seg = processor.post_process_panoptic_segmentation(outputs, target_sizes=target_sizes)[0]
            segmentation = panoptic_seg['segmentation'].cpu().numpy()
            segments_info = panoptic_seg['segments_info']

            final_mask = np.zeros_like(segmentation, dtype=np.uint8)

            for segment in segments_info:
                if segment['label_id'] == 149:
                    final_mask[segmentation == segment['id']] = 1

            results = yolo_model(frame, device=device, verbose=False)[0]
            for box, cls in zip(results.boxes.xyxy.cpu().numpy(), results.boxes.cls.cpu().numpy()):
                if int(cls) in target_classes:
                    x1, y1, x2, y2 = map(int, box)
                    final_mask[y1:y2, x1:x2] = 13

            seg_name = os.path.splitext(filename)[0] + ".npy"
            np.save(os.path.join(seg_maps_dir, seg_name), final_mask)

            pbar.update(1)

    print(f"Сегментация завершена. Результаты сохранены в {seg_maps_dir}")

if __name__ == "__main__":
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    run_segmentation("./frames", "./segmented_maps")
