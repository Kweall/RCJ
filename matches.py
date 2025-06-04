import sys
from pathlib import Path
import numpy as np
import torch
from tqdm import tqdm
# Импорт из SuperGluePretrainedNetwork
from models.matching import Matching
from models.utils import frame2tensor

def run_matching(seg_maps_dir: str = "./segmented_maps", keypoints_dir: str = "./keypoints", matches_dir: str = "./matches"):
    current_dir = Path(__file__).parent.resolve()
    superglue_path = current_dir / 'modules' / 'SuperGluePretrainedNetwork'
    sys.path.insert(0, str(superglue_path))

    seg_maps_dir = Path(seg_maps_dir)
    keypoints_dir = Path(keypoints_dir)
    matches_dir = Path(matches_dir)
    matches_dir.mkdir(parents=True, exist_ok=True)

    seg_map_paths = sorted(seg_maps_dir.glob('*.npy'))
    if len(seg_map_paths) < 2:
        raise RuntimeError(f"Недостаточно сегментационных карт в {seg_maps_dir} для сопоставления")

    superglue_config = {
        'superpoint': {'nms_radius': 4, 'keypoint_threshold': 0.015, 'max_keypoints': -1},
        'superglue': {
            'weights': 'outdoor',
            'sinkhorn_iterations': 20,
            'match_threshold': 0.2,
        }
    }

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print("Инициализация модели SuperGlue...")
    matcher = Matching(superglue_config).eval().to(device)
    print("Модель SuperGlue загружена.")

    # Ожидаемое разрешение сегментационных карт
    expected_shape = (540, 960)

    for i in tqdm(range(len(seg_map_paths) - 1), desc="Сопоставление SuperGlue"):
        seg_map_path0 = seg_map_paths[i]
        seg_map_path1 = seg_map_paths[i + 1]

        img0 = np.load(seg_map_path0)
        img1 = np.load(seg_map_path1)

        if img0.shape != expected_shape or img1.shape != expected_shape:
            print(f"Ошибка: Сегментационная карта {seg_map_path0.name} или {seg_map_path1.name} имеет неверный размер: {img0.shape}, {img1.shape}")
            continue
        if img0.ndim != 2 or img1.ndim != 2:
            print(f"Ошибка: Сегментационная карта {seg_map_path0.name} или {seg_map_path1.name} не является двумерной: ndim={img0.ndim}, {img1.ndim}")
            continue

        img0 = img0.astype(np.uint8)
        img1 = img1.astype(np.uint8)

        kpt0_path = keypoints_dir / (seg_map_path0.stem + '.npy')
        kpt1_path = keypoints_dir / (seg_map_path1.stem + '.npy')

        if not kpt0_path.exists() or not kpt1_path.exists():
            print(f"Пропущена пара, ключевые точки не найдены: {kpt0_path}, {kpt1_path}")
            continue

        kpt0 = np.load(kpt0_path, allow_pickle=True).item()
        kpt1 = np.load(kpt1_path, allow_pickle=True).item()

        pts0 = kpt0['keypoints']
        if pts0.shape[0] == 3:
            pts0 = pts0[:2, :]
        kpts0 = torch.from_numpy(pts0.T)[None].float().to(device)

        pts1 = kpt1['keypoints']
        if pts1.shape[0] == 3:
            pts1 = pts1[:2, :]
        kpts1 = torch.from_numpy(pts1.T)[None].float().to(device)

        desc0 = torch.from_numpy(kpt0['descriptors'].T)[None].float().to(device)
        desc1 = torch.from_numpy(kpt1['descriptors'].T)[None].float().to(device)

        # Преобразуем сегментационные карты в тензоры
        image0 = frame2tensor(img0, device)
        image1 = frame2tensor(img1, device)

        data = {
            'image0': image0,
            'image1': image1,
            'keypoints0': kpts0,
            'keypoints1': kpts1,
            'descriptors0': desc0,
            'descriptors1': desc1,
            'scores0': torch.ones_like(kpts0[..., 0]),
            'scores1': torch.ones_like(kpts1[..., 0]),
            'image0_shape': torch.tensor(img0.shape[::-1], dtype=torch.float32)[None].to(device),
            'image1_shape': torch.tensor(img1.shape[::-1], dtype=torch.float32)[None].to(device),
        }

        with torch.no_grad():
            pred = matcher(data)
            matches = pred['matches0'][0].cpu().numpy()  # (N0,)
            valid = matches > -1
            matched_kpts0 = kpts0[0].cpu().numpy()[valid]
            matched_kpts1 = kpts1[0].cpu().numpy()[matches[valid]]

        save_path = matches_dir / f'{seg_map_path0.stem}_{seg_map_path1.stem}.npy'
        np.save(save_path, {
            'frame0': seg_map_path0.name,
            'frame1': seg_map_path1.name,
            'keypoints0': matched_kpts0,
            'keypoints1': matched_kpts1
        })

    print(f"Сопоставление ключевых точек завершено. Результаты сохранены в {matches_dir}")

if __name__ == "__main__":
    run_matching()