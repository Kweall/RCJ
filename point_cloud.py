import os
import numpy as np
import cv2
import open3d as o3d
from glob import glob
from tqdm import tqdm

def build_point_cloud(
    frame_dir='./frames/',
    depth_dir='./depth_maps/',
    seg_dir='./segmented_maps/',
    matches_dir='./matches/',
    output_dir='./output/',
    fx=525.0, fy=525.0,
    cx=480, cy=270,
    voxel_size=0.05,
    valid=False,
    valid_seg_ids=[1, 10, 12, 13, 140, 149]
):
    os.makedirs(output_dir, exist_ok=True)
    K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]])

    frames = sorted(glob(os.path.join(frame_dir, '*.png')))[::-1]
    depths = sorted(glob(os.path.join(depth_dir, '*.npy')))[::-1]
    segs = sorted(glob(os.path.join(seg_dir, '*.npy')))[::-1]
    matches = sorted(glob(os.path.join(matches_dir, '*.npy')))[::-1]

    global_points = []
    global_colors = []
    occupied_voxels = set()
    global_pose = np.eye(4)

    def filter_new_points(pts, voxel_set, voxel_size):
        rounded = np.round(pts / voxel_size).astype(np.int32)
        mask = []
        for pt_voxel in map(tuple, rounded):
            if pt_voxel not in voxel_set:
                voxel_set.add(pt_voxel)
                mask.append(True)
            else:
                mask.append(False)
        return np.array(mask, dtype=bool)

    def constrain_pose(pose):
        pose[1, 3] = 0
        rvec, _ = cv2.Rodrigues(pose[:3, :3])
        rvec[0] = 0
        rvec[2] = 0
        R = cv2.Rodrigues(rvec)[0]
        pose[:3, :3] = R
        return pose

    for i in tqdm(range(len(frames) - 1), desc='Обратная реконструкция'):
        frame1 = cv2.imread(frames[i])
        frame1_rgb = cv2.cvtColor(frame1, cv2.COLOR_BGR2RGB)

        depth1 = np.load(depths[i])
        seg1 = np.load(segs[i])
        match_data = np.load(matches[i], allow_pickle=True).item()
        kpts0 = match_data['keypoints0']
        kpts1 = match_data['keypoints1']

        h, w = depth1.shape
        u = kpts0[:, 0].astype(int)
        v = kpts0[:, 1].astype(int)
        valid_mask = (u >= 0) & (u < w) & (v >= 0) & (v < h)

        kpts0 = kpts0[valid_mask]
        kpts1 = kpts1[valid_mask]
        u = u[valid_mask]
        v = v[valid_mask]

        z = depth1[v, u]
        valid_depth = z > 0.1

        if np.sum(valid_depth) >= 6:
            u = u[valid_depth]
            v = v[valid_depth]
            z = z[valid_depth]
            kpts0 = kpts0[valid_depth]
            kpts1 = kpts1[valid_depth]

            x = (u - cx) * z / fx
            y = (v - cy) * z / fy
            points_3d = np.stack((x, y, z), axis=1).astype(np.float32)
            points_2d = kpts1.astype(np.float32)

            success, rvec, tvec, inliers = cv2.solvePnPRansac(
                objectPoints=points_3d,
                imagePoints=points_2d,
                cameraMatrix=K,
                distCoeffs=None,
                flags=cv2.SOLVEPNP_ITERATIVE
            )

            if success:
                R, _ = cv2.Rodrigues(rvec)
                t = tvec.flatten()

                T = np.eye(4)
                T[:3, :3] = R
                T[:3, 3] = t
                T = constrain_pose(T)
                global_pose = global_pose @ T
            else:
                print(f"PnP failed at frame {i} — using previous pose.")
        else:
            print(f"Too few points for PnP at frame {i} — using previous pose.")

        MAX_DEPTH = np.max(depth1)
        MIN_DEPTH = 0.65 * MAX_DEPTH
        
        if i == 0:
            mask = depth1 > MIN_DEPTH
        else:
            mask = (depth1 > MIN_DEPTH) & (depth1 < MAX_DEPTH * 0.87)
            
        if valid:
            seg_mask = np.isin(seg1, valid_seg_ids)
            mask = mask & seg_mask

        mask = mask.astype(bool)

        u_map, v_map = np.meshgrid(np.arange(w), np.arange(h))
        z_map = depth1[mask]
        x_map = (u_map[mask] - cx) * z_map / fx
        y_map = (v_map[mask] - cy) * z_map / fy
        pts_local = np.stack((x_map, y_map, z_map), axis=1)

        pts_global = (global_pose[:3, :3] @ pts_local.T + global_pose[:3, 3:4]).T
        new_mask = filter_new_points(pts_global, occupied_voxels, voxel_size)
        pts_global = pts_global[new_mask]

        rgb_all = frame1_rgb[mask].astype(np.float32) / 255.0
        rgb = rgb_all[new_mask]

        global_points.extend(pts_global)
        global_colors.extend(rgb)

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(np.array(global_points))
    pcd.colors = o3d.utility.Vector3dVector(np.array(global_colors))

    output_path = os.path.join(output_dir, "cloud.ply")
    o3d.io.write_point_cloud(output_path, pcd)
    print(f"Облако точек сохранено в: {output_path}")

    return pcd, output_path
