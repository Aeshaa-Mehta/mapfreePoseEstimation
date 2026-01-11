import os
import cv2
import numpy as np
from scipy.spatial.transform import Rotation as SciR
import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import re

def natural_sort_key(s):
    return [int(text) if text.isdigit() else text.lower()
            for text in re.split('([0-9]+)', s)]

class WheelchairRunDataset(Dataset):
    def __init__(self, root_dir):
        self.root_dir = root_dir
        self.rgb_dir = os.path.join(root_dir, "rgb")
        self.depth_dir = os.path.join(root_dir, "aligned_depth")

        self.rgb_files = sorted(os.listdir(self.rgb_dir), key = natural_sort_key)
        self.depth_files = sorted(os.listdir(self.depth_dir), key = natural_sort_key)

        assert len(self.rgb_files) == len(self.depth_files)

        # --- intrinsics ---
        intr_path = os.path.join(root_dir, "intrinsics.txt")
        with open(intr_path) as f:
            for line in f:
                if line.startswith("#") or line.strip() == "":
                    continue
                fx, fy, cx, cy, w, h = map(float, line.split())
                break

        self.K = torch.tensor([
            [fx, 0,  cx],
            [0,  fy, cy],
            [0,  0,  1]
        ], dtype=torch.float32)

        # --- poses ---
        self.poses = self._load_poses(
            os.path.join(root_dir, "poses_camera_tum.txt")
        )

    def _load_poses(self, path):
        poses = []
        with open(path) as f:
            for line in f:
                vals = list(map(float, line.split()))
                _, tx, ty, tz, qx, qy, qz, qw = vals

                # Scipy uses [x, y, z, w] order by default, matching your README
                R = SciR.from_quat([qx, qy, qz, qw]).as_matrix()

                T = np.eye(4)
                T[:3, :3] = R
                T[:3, 3] = [tx, ty, tz]
                poses.append(torch.from_numpy(T).float())
        return poses

    # def _quat_to_rot(self, x, y, z, w):
    #     q = np.array([w, x, y, z])
    #     n = np.dot(q, q)
    #     if n < 1e-8:
    #         return np.eye(3)
    #     q *= np.sqrt(2.0 / n)
    #     q = np.outer(q, q)
    #     return np.array([
    #         [1-q[2,2]-q[3,3], q[1,2]-q[3,0], q[1,3]+q[2,0]],
    #         [q[1,2]+q[3,0], 1-q[1,1]-q[3,3], q[2,3]-q[1,0]],
    #         [q[1,3]-q[2,0], q[2,3]+q[1,0], 1-q[1,1]-q[2,2]]
    #     ])

    def __len__(self):
        return len(self.rgb_files)

    def __getitem__(self, idx):
        # --- RGB ---
        rgb = cv2.imread(os.path.join(self.rgb_dir, self.rgb_files[idx]))
        rgb = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)
        rgb = torch.from_numpy(rgb).permute(2, 0, 1).float() / 255.0

        # --- Depth ---
        depth = cv2.imread(
            os.path.join(self.depth_dir, self.depth_files[idx]),
            cv2.IMREAD_UNCHANGED
        ).astype(np.float32) / 1000.0
        depth = torch.from_numpy(depth)

        return {
            "rgb": rgb,
            "depth": depth,
            "K": self.K.clone(),
            "pose": self.poses[idx],
            "idx": idx
        }

# data_path = "../../../../../scratch/toponavgroup/indoor-topo-loc/datasets/rrc-lab-data/wheelchair-runs-20241220/run-1-wheelchair-mapping"

