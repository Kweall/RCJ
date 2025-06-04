import streamlit as st
import numpy as np
import open3d as o3d
import k3d
import os
import tempfile
import shutil
from main import run_video_cropping, run_segmentation, run_depth_estimation, run_keypoints, run_matching, build_point_cloud
st.set_page_config(
    page_title="3D Облако Точек",
    layout="wide",
)

st.title("Конвертация видео в 3D облако точек")

col1, col2 = st.columns([1, 2])

with col1:
    uploaded_file = st.file_uploader("Загрузите видео", type=["mp4", "avi", "mov"])
    step = st.number_input("Шаг обработки кадров", min_value=1, max_value=50, value=20, step=10)
    filter_to_road_only = st.checkbox("Оставить только проезжую часть")
    if uploaded_file is not None:
        shutil.rmtree("./frames", ignore_errors=True)
        shutil.rmtree("./segmented_maps", ignore_errors=True)
        shutil.rmtree("./depth_maps", ignore_errors=True)
        shutil.rmtree("./keypoints", ignore_errors=True)
        shutil.rmtree("./matches", ignore_errors=True)
        process_video = st.success(f"Видео {uploaded_file.name} загружено")

        if st.button("Преобразовать видео в 3D-сцену"):
            with tempfile.TemporaryDirectory() as tmpdir:
                video_path = os.path.join(tmpdir, uploaded_file.name)
                with open(video_path, "wb") as f:
                    f.write(uploaded_file.read())

                process_video.empty()
                placeholder = st.empty()
                placeholder.info("Обработка видео...")

                try:
                    run_video_cropping(input_video_path=video_path, step=step)
                    placeholder.empty()
                    placeholder.info("Вычисляется сегментация видео...")
                    run_segmentation()
                    placeholder.empty()
                    placeholder.info("Вычисляется глубина пикселей...")
                    run_depth_estimation()
                    placeholder.empty()
                    placeholder.info("Вычисляются ключевые точки между кадрами...")
                    run_keypoints()
                    placeholder.empty()
                    placeholder.info("Сопоставляются найденные ключевые точки...")
                    run_matching()
                    placeholder.empty()
                    placeholder.info("Генерируется 3D-сцена...")
                    build_point_cloud(valid=filter_to_road_only)
                    placeholder.empty()
                except Exception as e:
                    st.error(f"Ошибка обработки: {e}")
                    st.stop()

            st.success("Обработка завершена! Облако точек сгенерировано:")


with col2:
    ply_path = os.path.abspath("./output/cloud.ply")

    if os.path.exists(ply_path):
        try:
            pcd = o3d.io.read_point_cloud(ply_path)
            if not pcd.is_empty():
                points = np.asarray(pcd.points)
                colors = np.asarray(pcd.colors) if pcd.has_colors() else None

                if colors is not None and len(colors) == len(points):
                    colors_uint32 = (
                        (colors[:, 0] * 255).astype(np.uint32) * 256**2 +
                        (colors[:, 1] * 255).astype(np.uint32) * 256 +
                        (colors[:, 2] * 255).astype(np.uint32)
                    ).astype(np.uint32)
                else:
                    colors_uint32 = None

                plot = k3d.plot(grid_visible=False, camera_auto_fit=True)
                point_size = 0.02

                if colors_uint32 is not None:
                    scatter = k3d.points(points, point_size=point_size, shader="flat", colors=colors_uint32)
                else:
                    scatter = k3d.points(points, point_size=point_size, shader="flat", color=0xFF0000)

                plot += scatter
                plot.camera_auto_fit = True

                st.write("Вращайте и масштабируйте облако точек:")
                html = plot.get_snapshot()
                html = html.replace('width: 100%;', 'width: 1200px;')

                st.components.v1.html(html, height=600)
            else:
                st.info("Облако точек пока не сгенерировано.")
        except Exception as e:
            st.error(f"Ошибка при загрузке облака точек: {e}")
    else:
        st.info("Облако точек пока не сгенерировано.")
