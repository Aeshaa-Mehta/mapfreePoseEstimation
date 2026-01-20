import torch
import cv2
import json
import numpy as np
from pathlib import Path
from transformers import AutoImageProcessor, AutoModel
import os
import time
from torch.utils.data import DataLoader
import rerun as rr
from moge.model.v2 import MoGeModel
from romav2 import RoMaV2 
from data.torwik_runs import TorWICDataset
import faiss
import torchvision.transforms as T
from PIL import Image
import csv
from metrics import pose_error, r_error, t_error
from scipy.spatial.transform import Rotation as SciR


def log_results(filename, query_idx, ref_idx, inliers, error):
    file_exists = os.path.isfile(filename)
    headers = ['query_idx', 'reference_idx', 'inliers', 'error']
    
    with open(filename, 'a', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=headers)
        
        # Write header only if creating a new file
        if not file_exists:
            writer.writeheader()
            
        writer.writerow({
            'query_idx': query_idx,
            'reference_idx': ref_idx,
            'inliers': inliers,
            'error': f"{error:.6f}",
        })

class Relocalizer:
    def __init__(self, moge_path, mapping_data_path):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # 1. Load Database
        print("Loading Database...")
        self.index = faiss.read_index("./data/data.bin")
        with open("./data/file_database.json", "r") as f:
            self.metadata = json.load(f)

        print("Loading Mapping Dataset for GT lookup...")
        self.mapping_dataset = TorWICDataset(mapping_data_path, resize=(512, 288))

        self.ref_poses = []
        for i in range(len(self.mapping_dataset)):
            self.ref_poses.append(self.mapping_dataset[i]["pose"].cpu().numpy())

        # 2. Load Retrieval Model (DINOv2)
        print("Loading DINOv2-vits14 ...")
        self.retrieval_model = torch.hub.load("facebookresearch/dinov2", "dinov2_vits14").to(self.device).eval()

        # Standard DINOv2 Transformation
        self.transform = T.Compose([
            T.Resize(224, interpolation=T.InterpolationMode.BICUBIC),
            T.CenterCrop(224),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        # 3. Load RoMa v2 (Matching)
        print("Loading RoMa v2...")
        self.roma = RoMaV2().to(DEVICE).eval()
        # if roma_weights:
        #     self.roma.load_state_dict(torch.load(roma_weights))
        
        # 4. Load MoGe-2 (Geometry)
        print("Loading MoGe-2...")
        self.moge = MoGeModel.from_pretrained(moge_path).to(self.device).eval()

    @torch.no_grad()
    def get_query_embedding(self, image_path):
        img = Image.open(image_path).convert('RGB')
        img_t = self.transform(img).unsqueeze(0).to(self.device)
        
        # Extract features
        embedding = self.retrieval_model(img_t)
        # L2 Normalize if you used IndexFlatIP, or keep raw if you used IndexFlatL2
        # (Standardizing to L2 normalization is usually safer)
        embedding = torch.nn.functional.normalize(embedding, p=2, dim=-1)
        
        return embedding.cpu().numpy().astype('float32')

    def relocalize(self, query_img_path, query_K):
        # -- 1. Retrieval --
        query_cv2 = cv2.imread(query_img_path)
        h_q, w_q = query_cv2.shape[:2]
        query_rgb = cv2.cvtColor(query_cv2, cv2.COLOR_BGR2RGB)
        query_tensor = torch.from_numpy(query_rgb).permute(2, 0, 1).float().unsqueeze(0) / 255.0
        
        q_emb = self.get_query_embedding(query_img_path)
        # Search for top 1
        # For IndexFlatIP, higher is better (1.0000 is perfect match)
        similarities, indices = self. index.search(q_emb.astype('float32'), 1)
        
        best_overall_pose = None
        max_inlier_count = 100
        
        # print(f"-> Testing Top 5 Candidates...")
        # sims = torch.matmul(q_desc, self.ref_descriptors.T)
        # best_row = torch.argmax(sims).item()
        
        ref_idx = indices[0][0]
        best_sim = similarities[0][0]
    
        # print(f"Query: {query_img_path}")
        # print(f"Match: {self.metadata[ref_idx]}") # Look up filename in the saved JSON
        # print(f"Similarity Score: {best_sim:.4f}")

        ref_path = self.metadata[ref_idx]
        ref_sample = self.mapping_dataset[ref_idx]
        T_ref_world = ref_sample["pose"].cpu().numpy()
        K_ref = (ref_sample["K"]).cpu().numpy()
        # print(f"-> Query matched to Reference: {Path(ref_path).name} (Sim: {sims[0, ref_idx]:.3f})")
        ref_pos_in_mapping_world = T_ref_world[:3, 3]

        # 2. romav2 matching
        # print("Matching pixels with RoMa v2...")
        preds = self.roma.match(query_img_path, ref_path)
        matches, _, _, _ = self.roma.sample(preds, 2000) # Sample 2000 matches

        # Convert to pixel coordinates
        kpts_q, kpts_ref = self.roma.to_pixel_coordinates(matches, h_q, w_q, h_q, w_q)
        kpts_q, kpts_ref = kpts_q.cpu().numpy(), kpts_ref.cpu().numpy()
        torch.cuda.empty_cache()

        # 3. moge2 depth
        # print("Inferring Ref depth with MoGe-2...")
        ref_cv2 = cv2.imread(ref_path)
        ref_tensor = torch.from_numpy(cv2.cvtColor(ref_cv2, cv2.COLOR_BGR2RGB)).permute(2, 0, 1).float().unsqueeze(0) / 255.0
        moge_out = self.moge.infer(ref_tensor.to(self.device))
        depth_ref = moge_out['depth'][0].cpu().numpy() # (H, W)
        torch.cuda.empty_cache()

        # 4.lift Ref Pixels to 3D World Points
        u_ref = np.clip(kpts_ref[:, 0].astype(int), 0, w_q - 1)
        v_ref = np.clip(kpts_ref[:, 1].astype(int), 0, h_q - 1)
        z_ref = depth_ref[v_ref, u_ref] 
        # print(u_ref[:10])
        # print(v_ref[:10])
        # print(z_ref[:10])   

        # Filter out invalid depth points
        valid = z_ref > 0
        kpts_q, kpts_ref, z_ref = kpts_q[valid], kpts_ref[valid], z_ref[valid]

        # Back-project ref pixels to Ref Camera Space
        x_c = (kpts_ref[:, 0] - K_ref[0, 2]) * z_ref / K_ref[0, 0]
        y_c = (kpts_ref[:, 1] - K_ref[1, 2]) * z_ref / K_ref[1, 1]
        pts_ref_cam = np.stack([x_c, y_c, z_ref, np.ones_like(z_ref)], axis=-1)

        # to 3d
        pts_world = (T_ref_world @ pts_ref_cam.T).T[:, :3]

        # -- 5. Pose Estimation (PnP) --
        # print(f"-> Solving PnP with {len(pts_world)} matches...")
        success, rvec, tvec, inliers = cv2.solvePnPRansac(
            pts_world.astype(np.float32), 
            kpts_q.astype(np.float32), 
            query_K.astype(np.float32), 
            distCoeffs=None, iterationsCount=1000, reprojectionError=1.5, flags=cv2.SOLVEPNP_SQPNP
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

        if success and inlier_count > max_inlier_count:
            R_qw, _ = cv2.Rodrigues(rvec)
            T_qw = np.eye(4)
            T_qw[:3, :3] = R_qw
            T_qw[:3, 3] = tvec.flatten()
            best_overall_pose = np.linalg.inv(T_qw)
        else:
            best_overall_pose = None

        return best_overall_pose, inlier_count, ref_idx, ref_path
           
if __name__ == "__main__":

    REF_DAY = "Jun15"
    REF_RUN = "Aisle_CCW_Run_1"
    QUERY_DAY = "Jun23"
    QUERY_RUN = "Aisle_CW_Run_2"
    EXPERIMENT_NAME = f"-{REF_RUN.replace('_', '').lower()}-query-{QUERY_DAY.lower()}-{QUERY_RUN.replace('_', '').lower()}-dinov2-roma"
    MOGE_PATH = "../../../../../scratch/dynrecon/checkpoints/moge-vits.pt"
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    DATA_ROOT = "../../../../../scratch/rohit.jayanti/indoor-topo-loc/datasets/TorWIC-SLAM"
    QUERY_DATA_PATH = os.path.join(DATA_ROOT, f"{QUERY_DAY}/{QUERY_RUN}")
    mapping_data_path = os.path.join(DATA_ROOT, f"{REF_DAY}/{REF_RUN}")
    pred_tum_path = f"../../../../../scratch/dynrecon/exps/pred_trajectory_tum/{EXPERIMENT_NAME}.txt"
    retrieved_tum_path = f"../../../../../scratch/dynrecon/exps/retrieved_trajectory_tum/{EXPERIMENT_NAME}.txt"
    log_file = f"../../../../../scratch/dynrecon/results/{EXPERIMENT_NAME}_results.csv"

    # 1. Init
    reloc = Relocalizer(MOGE_PATH, mapping_data_path)

    # 2. Load a Query Image from the dataset
    query_dataset = TorWICDataset(QUERY_DATA_PATH, resize=(512, 288))
    query_loader = DataLoader(query_dataset, batch_size=1, shuffle=False)
    
    #initialize Rerun
    rr.init("Wheelchair_Mapping", spawn=False)
    rr.connect_grpc()  

    # Set the world coordinate system to Z-up 
    rr.log("world", rr.ViewCoordinates.RIGHT_HAND_Z_UP, static=True)
    # Lists to store the full history for drawing paths
    gt_path_history = []
    pred_path_history = []

    with open(pred_tum_path, "w") as pred_f, open(retrieved_tum_path, "w") as retrieved_f:
        with torch.no_grad():
            for i,batch in enumerate(query_loader):

                # Set the timeline
                rr.set_time("frame_idx", sequence = i)
                query_img = batch["rgb"]  # [1, 3, H, W]
                query_K = batch["K"].numpy()[0]
                gt_pose = batch["pose"].numpy()[0]
                gt_path_history.append(gt_pose[:3, 3])
                query_img_path = os.path.join(query_dataset.rgb_dir, query_dataset.rgb_files[i])
                pred_pose, inlier_count, ref_idx, ref_path = reloc.relocalize(query_img_path, query_K)

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

                #log gt (green)
                rr.log(
                    "world/gt_camera",
                    rr.Transform3D(
                        translation=gt_pose[:3, 3],
                        mat3x3=gt_pose[:3, :3],
                        relation=rr.TransformRelation.ChildFromParent
                    )
                )
                rr.log("world/gt_path", rr.LineStrips3D([gt_path_history], colors=[0, 255, 0], radii=0.02))

                # Log the image to the GT camera frustum
                query_img_np = (batch["rgb"][0].permute(1, 2, 0).numpy() * 255).astype(np.uint8)
                # rr.log("world/gt_camera", rr.Image(query_img_np))



                if pred_pose is not None:
                    # predicted (Red)
                    rr.log(
                        "world/pred_camera",
                        rr.Transform3D(
                            translation=pred_pose[:3, 3],
                            mat3x3=pred_pose[:3, :3],
                            relation=rr.TransformRelation.ChildFromParent
                        )
                    )
                    pred_path_history.append(pred_pose[:3, 3])
                    rr.log("world/pred_path", rr.LineStrips3D([pred_path_history], colors=[255, 0, 0], radii=0.03))                
                    error = np.linalg.norm(pred_pose[:3, 3] - gt_pose[:3, 3])

                    print(f"Frame {i}: Error {error:.4f}m | Inliers: {inlier_count}")

                    rr.log("debug/images/query", rr.Image(query_img_np))
                    ref_img_cv2 = cv2.imread(ref_path)
                    ref_img_rgb = cv2.cvtColor(ref_img_cv2, cv2.COLOR_BGR2RGB)
                    rr.log("debug/images/retrieved", rr.Image(ref_img_rgb))
                    # rr.log("debug/ref_idx", rr.Scalars(ref_idx))
                    rr.log("debug/inliers", rr.Scalars(inlier_count))
                    # rr.log("debug/translation_error", rr.Scalars(error))

                else:
                    print(f"Frame {i}: Localization FAILED")
                    # Clear the pred_camera from the view if it failed
                    rr.log("world/pred_camera", rr.Clear(recursive=True))
                    # break  # stop on first failure


                    # print("\n--- RESULTS ---")
                    # print("Predicted Position:", pred_pose[:3, 3])
                    # print("GT Position:       ", gt_pose[:3, 3])
                    # error = np.linalg.norm(pred_pose[:3, 3] - gt_pose[:3, 3])
                    # print(f"Translation Error: {error:.4f} meters")
                    # print(f"Number of Inliers: {inlier_count}\n")

                # Log results to CSV
                log_results(
                        log_file, 
                        query_idx=i,
                        ref_idx=ref_idx, 
                        inliers=inlier_count, 
                        error=error if pred_pose is not None else -1.0, 
        )

    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("Disconnecting...")
        rr.disconnect()  # Disconnect gracefully on script termination
            