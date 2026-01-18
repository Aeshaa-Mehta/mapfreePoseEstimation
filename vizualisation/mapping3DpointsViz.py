import torch
import numpy as np
import cv2
import rerun as rr
from tqdm import tqdm
from torch.utils.data import DataLoader
from pathlib import Path
from data.wheelchair_runs import WheelchairRunDataset
from moge.model.v2 import MoGeModel
import time

def run_rerun_mapping(data_path, model_path, stride=2, pixel_step=8):
    # --- 1. Initialize Rerun ---
    # If running on a server, this starts a web server on port 9876
    rr.init("Wheelchair_Mapping", spawn=False)
    rr.connect_grpc() 

    device = torch.device("cuda")
    
    # --- 2. Load Model & Data ---
    model = MoGeModel.from_pretrained(model_path).to(device).eval()
    dataset = WheelchairRunDataset(data_path)
    loader = DataLoader(dataset, batch_size=1, shuffle=False)

    # Log the World Coordinate System (Z-up)
    rr.log("world", rr.ViewCoordinates.RIGHT_HAND_Z_UP, static=True)

    print(f"Streaming mapping to Rerun. Open the link provided above.")

    with torch.no_grad():
        for i, batch in enumerate(tqdm(loader)):
            if i % stride != 0:
                continue
            
            # Set the timeline
            rr.set_time("frame_idx", sequence = i)

            # Data to device
            image_tensor = batch["rgb"].to(device)
            T_wc = batch["pose"][0].numpy()
            K = batch["K"][0].numpy()
            
            # Inference
            output = model.infer(image_tensor)
            depth = output['depth'][0]
            mask = output['mask'][0] > 0.5
            
            # --- Log Camera Pose ---
            # Extract translation and rotation
            translation = T_wc[:3, 3]
            # Rerun expects rotation as a Mat3x3 or Quaternion
            rotation_matrix = T_wc[:3, :3]
            
            rr.log(
                "world/camera",
                rr.Transform3D(
                    translation=translation,
                    mat3x3=rotation_matrix,
                    from_parent=False # T_wc is world-from-camera
                )
            )

            # --- Log Intrinsics & Image ---
            # This allows Rerun to "project" the image from the camera's frustum
            rr.log(
                "world/camera/image",
                rr.Pinhole(
                    resolution=[image_tensor.shape[3], image_tensor.shape[2]],
                    image_from_camera=K,
                )
            )
            
            rgb_np = (image_tensor[0].permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
            rr.log("world/camera/image", rr.Image(rgb_np))

            # --- Log 3D Points ---
            h, w = depth.shape
            ii, jj = torch.meshgrid(torch.arange(h, device=device), 
                                    torch.arange(w, device=device), indexing='ij')
            
            # Subsample for smooth visualization
            sub_mask = (ii % pixel_step == 0) & (jj % pixel_step == 0) & mask
            
            z = depth[sub_mask]
            u, v = jj[sub_mask].float(), ii[sub_mask].float()
            
            # Back-project to Camera Frame
            x_c = (u - K[0, 2]) * z / K[0, 0]
            y_c = (v - K[1, 2]) * z / K[1, 1]
            z_c = z
            pts_cam = torch.stack([x_c, y_c, z_c], dim=-1)
            
            # Transform to World Frame
            # Note: We can log points in 'world' or 'world/camera'
            # Logging in 'world' makes them stay permanent
            pts_cam_homo = torch.cat([pts_cam, torch.ones_like(z).unsqueeze(-1)], dim=-1)
            T_wc_torch = torch.from_numpy(T_wc).to(device).float()
            pts_world = (T_wc_torch @ pts_cam_homo.T).T[:, :3]
            
            colors = rgb_np[sub_mask.cpu().numpy()]
            
            # Log the points as part of a persistent map
            # We use a unique name for each frame if we want them all to stay,
            # or use the same name if we just want to see the current frustum.
            rr.log(
                f"world/points/frame_{i}", 
                rr.Points3D(pts_world.cpu().numpy(), colors=colors),
                static=False # Set to True if you want the map to "build up"
            )

            print("Keep the script running to maintain the connection.")
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("Disconnecting...")
        rr.disconnect()  # Disconnect gracefully on script termination

if __name__ == "__main__":
    DATA_PATH = "../../../../../scratch/toponavgroup/indoor-topo-loc/datasets/rrc-lab-data/wheelchair-runs-20241220/run-1-wheelchair-mapping"
    MODEL_PATH = "../../../../../scratch/dynrecon/checkpoints/moge-vits.pt"
    run_rerun_mapping(DATA_PATH, MODEL_PATH)



# import torch
# descriptors = torch.load("../../../../../scratch/dynrecon/wheelchair_reloc_db_dinov2/descriptors.pt")
# # Check the standard deviation of the descriptors
# print("Std Dev across frames:", descriptors.std(dim=0).mean().item())
# # Check similarity between the first two frames
# sim = torch.nn.functional.cosine_similarity(descriptors[0:1], descriptors[1:2])
# print("Similarity between neighbors:", sim.item())