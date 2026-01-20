import os
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
from scipy.spatial.transform import Rotation as SciR
import re

def natural_sort_key(s):
    return [int(text) if text.isdigit() else text.lower()
            for text in re.split('([0-9]+)', s)]

class TorWICDataset(Dataset):
    def __init__(self, root_dir, resize=(512, 288)):
        """
        Args:
            root_dir: Path to the specific run (e.g., dataset_root/Jun15/Run_1)
            resize: (width, height) to resize images and depth
        """
        self.root_dir = root_dir
        self.resize = resize  # (W, H)
        
        # TorWIC specific subfolder names
        self.rgb_dir = os.path.join(root_dir, "image_right")
        self.depth_dir = os.path.join(root_dir, "depth_right")

        self.rgb_files = sorted(os.listdir(self.rgb_dir), key=natural_sort_key)
        self.depth_files = sorted(os.listdir(self.depth_dir), key=natural_sort_key)

        # --- Intrinsics ---
        # TorWIC intrinsics_right.txt format: fx fy cx cy w h
        intr_path = os.path.join(root_dir, "intrinsics_right.txt")
        with open(intr_path) as f:
            for line in f:
                if line.startswith("#") or line.strip() == "":
                    continue
                fx, fy, cx, cy, w_orig, h_orig = map(float, line.split())
                break

        # Calculate scaling factors for K based on resize
        scale_x = self.resize[0] / w_orig
        scale_y = self.resize[1] / h_orig

        self.K = torch.tensor([
            [fx * scale_x, 0,            cx * scale_x],
            [0,            fy * scale_y, cy * scale_y],
            [0,            0,            1           ]
        ], dtype=torch.float32)

        # --- Poses ---
        # TorWIC pose file: poses_camera_right_tum.txt
        pose_path = os.path.join(root_dir, "poses_camera_right_tum.txt")
        self.poses = self._load_poses(pose_path)
        
        # Verify alignment
        assert len(self.rgb_files) == len(self.poses), \
            f"Mismatch: {len(self.rgb_files)} images vs {len(self.poses)} poses"

    def _load_poses(self, path):
        poses = []
        if not os.path.exists(path):
            return []
        with open(path) as f:
            for line in f:
                if line.startswith("#"): continue
                vals = list(map(float, line.split()))
                # TUM Format: timestamp tx ty tz qx qy qz qw
                _, tx, ty, tz, qx, qy, qz, qw = vals

                R = SciR.from_quat([qx, qy, qz, qw]).as_matrix()
                T = np.eye(4)
                T[:3, :3] = R
                T[:3, 3] = [tx, ty, tz]
                poses.append(torch.from_numpy(T).float())
        return poses

    def __len__(self):
        return len(self.rgb_files)

    def __getitem__(self, idx):
        # --- RGB ---
        rgb_path = os.path.join(self.rgb_dir, self.rgb_files[idx])
        rgb = cv2.imread(rgb_path)
        rgb = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)
        
        # Resize RGB
        if self.resize:
            rgb = cv2.resize(rgb, self.resize, interpolation=cv2.INTER_LINEAR)
            
        rgb = torch.from_numpy(rgb).permute(2, 0, 1).float() / 255.0

        # --- Depth ---
        depth_path = os.path.join(self.depth_dir, self.depth_files[idx])
        depth = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED).astype(np.float32)
        
        # TorWIC depth is usually in mm, convert to meters
        depth = depth / 1000.0
        
        # Resize Depth (Must use NEAREST to avoid invalid interpolated depth values)
        if self.resize:
            depth = cv2.resize(depth, self.resize, interpolation=cv2.INTER_NEAREST)
            
        depth = torch.from_numpy(depth)

        return {
            "rgb": rgb,
            "depth": depth,
            "K": self.K.clone(),
            "pose": self.poses[idx],
            "idx": idx,
            "rgb_path": rgb_path # Useful for debug/matching
        }

# Example Usage:dataset = TorWICDataset("/path/to/TorWIC-SLAM/Jun15/Aisle_CCW_Run_1", resize=(512, 288))
# loader = DataLoader(dataset, batch_size=1, shuffle=False)
# 