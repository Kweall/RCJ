import sys
from pathlib import Path
import cv2
import torch
import numpy as np
from tqdm import tqdm

def run_keypoints(frames_dir: str = "./frames", save_dir: str = "./keypoints"):
    current_dir = Path(__file__).parent.resolve()
    superpoint_path = current_dir / 'modules' / 'SuperPointPretrainedNetwork'
    sys.path.insert(0, str(superpoint_path))

    from demo_superpoint import SuperPointFrontend

    frames_dir = Path(frames_dir)
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    config = {
        'nms_dist': 4,
        'conf_thresh': 0.015,
        'nn_thresh': 0.7,
        'cuda': torch.cuda.is_available()
    }

    print("Загружаем модель SuperPoint...")
    superpoint = SuperPointFrontend(weights_path=str(superpoint_path / 'superpoint_v1.pth'), **config)
    print("Модель SuperPoint загружена.")

    frame_paths = sorted(frames_dir.glob('*.png'))
    if not frame_paths:
        raise RuntimeError(f"Нет кадров в {frames_dir}")

    expected_shape = (540, 960) 

    for frame_idx, frame_path in enumerate(tqdm(frame_paths, desc="Processing keypoints", unit="frame"), 1):
        image = cv2.imread(str(frame_path), cv2.IMREAD_GRAYSCALE)
        if image is None:
            print(f"Ошибка: Не удалось загрузить {frame_path} как изображение")
            continue

        # Проверяем форму изображения
        if image.shape[:2] != expected_shape:
            print(f"Ошибка: Изображение {frame_path.name} имеет размер {image.shape[:2]}, ожидается {expected_shape}")
            continue

        # Удаляем третье измерение, если оно есть
        if image.ndim == 3 and image.shape[2] == 1:
            image = image.squeeze(axis=2)

        if image.ndim != 2:
            print(f"Ошибка: Изображение {frame_path.name} не является grayscale (ndim={image.ndim})")
            continue

        # Подготовка изображения для SuperPoint
        inp = image.astype(np.float32) / 255.0
        try:
            pts, desc, _ = superpoint.run(inp)
        except Exception as e:
            print(f"Ошибка при обработке SuperPoint для {frame_path.name}: {e}")
            continue

        fname = save_dir / (frame_path.stem + '.npy')
        np.save(fname, {
            'keypoints': pts,
            'descriptors': desc.T
        })

    print(f"Обработка ключевых точек завершена. Результаты сохранены в ./{save_dir}")

if __name__ == "__main__":
    run_keypoints()