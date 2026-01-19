import torch
import cv2
import json
import numpy as np
from pathlib import Path
import os
import time
from torch.utils.data import DataLoader
import rerun as rr
from moge.model.v2 import MoGeModel
from romav2 import RoMaV2 
from data.wheelchair_runs import WheelchairRunDataset
from PIL import Image
from metrics import r_error, t_error
from scipy.spatial.transform import Rotation as SciR


# --- CONFIG FOR POSE ORACLE ---
MAX_ROTATION_ERROR = 45.0    # Degrees
MAX_TRANSLATION_ERROR = 3.0  # Meters

# def log_results(filename, query_idx, ref_idx, inliers, error):
#     file_exists = os.path.isfile(filename)
#     headers = ['query_idx', 'reference_idx', 'inliers', 'error']
#     with open(filename, 'a', newline='') as f:
#         writer = csv.DictWriter(f, fieldnames=headers)
#         if not file_exists:
#             writer.writeheader()
#         writer.writerow({
#             'query_idx': query_idx,
#             'reference_idx': ref_idx,
#             'inliers': inliers,
#             'error': f"{error:.6f}" if error is not None else -1.0,
#         })

class Relocalizer:
    def __init__(self, moge_path, mapping_data_path):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # 1. Load Mapping Dataset and Cache Poses
        print("Loading Mapping Dataset for Pose Oracle...")
        self.mapping_dataset = WheelchairRunDataset(mapping_data_path)
        
        # Pre-cache all reference poses and paths for fast lookup
        self.ref_poses = []
        self.ref_paths = []
        for i in range(len(self.mapping_dataset)):
            sample = self.mapping_dataset[i]
            self.ref_poses.append(sample["pose"].cpu().numpy())
            # Assuming the dataset has a way to get the file path
            img_path = os.path.join(self.mapping_dataset.rgb_dir, self.mapping_dataset.rgb_files[i])
            self.ref_paths.append(img_path)

        # 2. Load RoMa v2 (Matching)
        print("Loading RoMa v2...")
        self.roma = RoMaV2().to(self.device).eval()
        
        # 3. Load MoGe-2 (Geometry)
        print("Loading MoGe-2...")
        self.moge = MoGeModel.from_pretrained(moge_path).to(self.device).eval()

    def find_best_ref_oracle(self, query_gt_pose):
        """Pose Oracle Retrieval Strategy"""
        min_score = float('inf')
        best_ref_idx = -1

        for i, ref_pose in enumerate(self.ref_poses):
            # Calculate rotation error (deg) and translation error (m)
            # Using imported utils from mapfree.pose.utils
            r_err = r_error(query_gt_pose[:3, :3], ref_pose[:3, :3])
            t_err = t_error(query_gt_pose[:3, 3], ref_pose[:3, 3])

            # Skip if errors exceed thresholds
            if r_err > MAX_ROTATION_ERROR or t_err > MAX_TRANSLATION_ERROR:
                continue
            
            # Using Rotation Error as the sorting score (as per your snippet)
            score = r_err

            if score < min_score:
                min_score = score
                best_ref_idx = i
        
        return best_ref_idx

    def relocalize(self, query_img_path, query_K, query_gt_pose):
        # -- 1. Retrieval (Pose Oracle) --
        ref_idx = self.find_best_ref_oracle(query_gt_pose)
        
        if ref_idx == -1:
            print(f"   Oracle: No reference frame found within thresholds.")
            return None, 0, -1, None

        ref_path = self.ref_paths[ref_idx]
        ref_sample = self.mapping_dataset[ref_idx]
        T_ref_world = self.ref_poses[ref_idx]
        K_ref = ref_sample["K"].cpu().numpy()

        # Get query image dimensions
        query_cv2 = cv2.imread(query_img_path)
        h_q, w_q = query_cv2.shape[:2]

        # -- 2. RoMa Matching --
        preds = self.roma.match(query_img_path, ref_path)
        matches, _, _, _ = self.roma.sample(preds, 2000) 

        kpts_q, kpts_ref = self.roma.to_pixel_coordinates(matches, h_q, w_q, h_q, w_q)
        kpts_q, kpts_ref = kpts_q.cpu().numpy(), kpts_ref.cpu().numpy()
        torch.cuda.empty_cache()

        # -- 3. MoGe-2 Depth --
        ref_cv2 = cv2.imread(ref_path)
        ref_tensor = torch.from_numpy(cv2.cvtColor(ref_cv2, cv2.COLOR_BGR2RGB)).permute(2, 0, 1).float().unsqueeze(0) / 255.0
        moge_out = self.moge.infer(ref_tensor.to(self.device))
        depth_ref = moge_out['depth'][0].cpu().numpy() 
        torch.cuda.empty_cache()

        # -- 4. Back-projection --
        u_ref = np.clip(kpts_ref[:, 0].astype(int), 0, w_q - 1)
        v_ref = np.clip(kpts_ref[:, 1].astype(int), 0, h_q - 1)
        z_ref = depth_ref[v_ref, u_ref] 

        valid = z_ref > 0
        kpts_q, kpts_ref, z_ref = kpts_q[valid], kpts_ref[valid], z_ref[valid]

        x_c = (kpts_ref[:, 0] - K_ref[0, 2]) * z_ref / K_ref[0, 0]
        y_c = (kpts_ref[:, 1] - K_ref[1, 2]) * z_ref / K_ref[1, 1]
        pts_ref_cam = np.stack([x_c, y_c, z_ref, np.ones_like(z_ref)], axis=-1)

        # Transform to World Space
        pts_world = (T_ref_world @ pts_ref_cam.T).T[:, :3]

        # -- 5. Pose Estimation (PnP) --
        success, rvec, tvec, inliers = cv2.solvePnPRansac(
            pts_world.astype(np.float32), 
            kpts_q.astype(np.float32), 
            query_K.astype(np.float32), 
            distCoeffs=None, iterationsCount=1500, reprojectionError=1.5, flags=cv2.SOLVEPNP_SQPNP
        )

        #pose refinement
        if success and len(inliers) >= 6:
            if success and inliers is not None and len(inliers) >= 6:
                idx = inliers.flatten()
                inlier_pts_world = pts_world[idx].astype(np.float32)
                inlier_kpts_q = kpts_q[idx].astype(np.float32)

                success, rvec, tvec = cv2.solvePnP(
                inlier_pts_world,
                inlier_kpts_q,
                query_K.astype(np.float32),
                distCoeffs=None,
                rvec=rvec,
                tvec=tvec,
                useExtrinsicGuess=True,
                flags=cv2.SOLVEPNP_ITERATIVE
            )

        if success:
            if np.linalg.norm(tvec) > 1000:
                success = False

        inlier_count = len(inliers) if inliers is not None else 0

        if success: #and inlier_count > max_inlier_count:
            R_qw, _ = cv2.Rodrigues(rvec)
            T_qw = np.eye(4)
            T_qw[:3, :3] = R_qw
            T_qw[:3, 3] = tvec.flatten()
            best_overall_pose = np.linalg.inv(T_qw)
        else:
            best_overall_pose = None

        return best_overall_pose, inlier_count, ref_idx, ref_path
           
if __name__ == "__main__":
    EXPERIMENT_NAME = "run-2-oracle-roma"
    MOGE_PATH = "../../../../../scratch/dynrecon/checkpoints/moge-vits.pt"
    QUERY_DATA_PATH = "../../../../../scratch/toponavgroup/indoor-topo-loc/datasets/rrc-lab-data/wheelchair-runs-20241220/run-2-wheelchair-query"
    MAPPING_DATA_PATH = "../../../../../scratch/toponavgroup/indoor-topo-loc/datasets/rrc-lab-data/wheelchair-runs-20241220/run-1-wheelchair-mapping"
    pred_tum_path = f"../../../../../scratch/dynrecon/exps/pred_trajectory_tum/{EXPERIMENT_NAME}.txt"
    retrieved_tum_path = f"../../../../../scratch/dynrecon/exps/retrieved_trajectory_tum/{EXPERIMENT_NAME}.txt"
    # 1. Init
    reloc = Relocalizer(MOGE_PATH, MAPPING_DATA_PATH)

    # 2. Setup Dataset
    query_dataset = WheelchairRunDataset(QUERY_DATA_PATH)
    query_loader = DataLoader(query_dataset, batch_size=1, shuffle=False)
    
    # 3. Initialize Rerun
    rr.init("Wheelchair_Reloc_Oracle", spawn=False)
    rr.connect_grpc()  
    rr.log("world", rr.ViewCoordinates.RIGHT_HAND_Z_UP, static=True)

    gt_path_history = []
    pred_path_history = []
    with open(pred_tum_path, "w") as pred_f, open(retrieved_tum_path, "w") as retrieved_f:
        with torch.no_grad():
            for i, batch in enumerate(query_loader):
                rr.set_time("frame_idx", sequence=i)
                
                query_K = batch["K"].numpy()[0]
                gt_pose = batch["pose"].numpy()[0]
                query_img_path = os.path.join(query_dataset.rgb_dir, query_dataset.rgb_files[i])
                
                # relocalize
                pred_pose, inlier_count, ref_idx, ref_path = reloc.relocalize(query_img_path, query_K, gt_pose)

                #exp pose logging
                if ref_idx != -1:
                    T_ref_world = reloc.ref_poses[ref_idx]
                    t_ref = T_ref_world[:3, 3]
                    q_ref = SciR.from_matrix(T_ref_world[:3, :3]).as_quat() # [qx, qy, qz, qw]

                    retrieved_line = (
                        f"{float(i):.9f} {t_ref[0]:.9f} {t_ref[1]:.9f} {t_ref[2]:.9f} "
                        f"{q_ref[0]:.9f} {q_ref[1]:.9f} {q_ref[2]:.9f} {q_ref[3]:.9f}\n"
                    )
                    retrieved_f.write(retrieved_line)

                    if pred_pose is not None:
                        t_pred = pred_pose[:3, 3]
                        q_pred = SciR.from_matrix(pred_pose[:3, :3]).as_quat()

                        pred_line = (
                            f"{float(i):.9f} {t_pred[0]:.9f} {t_pred[1]:.9f} {t_pred[2]:.9f} "
                            f"{q_pred[0]:.9f} {q_pred[1]:.9f} {q_pred[2]:.9f} {q_pred[3]:.9f}\n"
                        )
                        pred_f.write(pred_line)

                # Log Ground Truth
                gt_path_history.append(gt_pose[:3, 3])
                rr.log("world/gt_camera", rr.Transform3D(translation=gt_pose[:3, 3], mat3x3=gt_pose[:3, :3]))
                rr.log("world/gt_path", rr.LineStrips3D([gt_path_history], colors=[0, 255, 0], radii=0.02))

                query_img_np = (batch["rgb"][0].permute(1, 2, 0).numpy() * 255).astype(np.uint8)
                rr.log("debug/images/query", rr.Image(query_img_np))

                error = None
                if pred_pose is not None:
                    # Log Predicted
                    pred_path_history.append(pred_pose[:3, 3])
                    rr.log("world/pred_camera", rr.Transform3D(translation=pred_pose[:3, 3], mat3x3=pred_pose[:3, :3]))
                    rr.log("world/pred_path", rr.LineStrips3D([pred_path_history], colors=[255, 0, 0], radii=0.03))                
                    
                    error = np.linalg.norm(pred_pose[:3, 3] - gt_pose[:3, 3])
                    print(f"Frame {i}: Error {error:.4f}m | Inliers: {inlier_count} | Ref: {ref_idx}")

                    ref_img_rgb = cv2.cvtColor(cv2.imread(ref_path), cv2.COLOR_BGR2RGB)
                    rr.log("debug/images/retrieved", rr.Image(ref_img_rgb))
                    rr.log("debug/inliers", rr.Scalars(inlier_count))
                else:
                    print(f"Frame {i}: Oracle matching failed (No ref found or PnP failed)")
                    rr.log("world/pred_camera", rr.Clear(recursive=True))
                # log_results(LOG_FILE, i, ref_idx, inlier_count, error)

try:
    while True:
            time.sleep(1)
except KeyboardInterrupt:
        print("Disconnecting...")
        rr.disconnect() 