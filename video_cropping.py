import os
import cv2
from tqdm import tqdm

def run_video_cropping(input_video_path: str, frames_dir: str = "./frames", step: int = 20):
    os.makedirs(frames_dir, exist_ok=True)

    abs_video_path = os.path.abspath(input_video_path)
    print(f"Открытие видео: {abs_video_path}")

    cap = cv2.VideoCapture(abs_video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Не удалось открыть видео: {abs_video_path}")

    frame_idx = 1
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    skip = 0
    target_size = (960, 540)

    with tqdm(total=total_frames // step, desc="Processing frames") as pbar:
        while True:
            if skip == 0:
                ret, frame = cap.read()
                if not ret:
                    break
                resized_frame = cv2.resize(frame, target_size, interpolation=cv2.INTER_AREA)
                cv2.imwrite(os.path.join(frames_dir, f"{frame_idx:05d}.png"), resized_frame)
                frame_idx += 1
                skip += 1
                pbar.update(1)
            else:
                ret, frame = cap.read()
                if not ret:
                    break
                skip = (skip + 1) % step
                pbar.update(1)

    print(f"Сохранено {frame_idx - 1} кадров в директории {frames_dir}")

if __name__ == "__main__":
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    run_video_cropping("./input/1.mp4")
