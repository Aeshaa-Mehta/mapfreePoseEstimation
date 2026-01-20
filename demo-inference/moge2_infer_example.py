import cv2
import torch
# from moge.model.v1 import MoGeModel
from torch.utils.data import DataLoader
from moge.model.v2 import MoGeModel # Let's try MoGe-2
from data.wheelchair_runs import WheelchairRunDataset

CHECKPOINT_DIR = "../../../../../scratch/dynrecon/checkpoints/"
DATA_DIR = "../../../../../scratch/toponavgroup/indoor-topo-loc/datasets/rrc-lab-data/wheelchair-runs-20241220/run-1-wheelchair-mapping"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load the model from huggingface hub (or load from local).

model_path = f"{CHECKPOINT_DIR}/moge-vits.pt"  # MoGe-2 with normal prediction
model = MoGeModel.from_pretrained(model_path).to(device)
model.eval()
dataset = WheelchairRunDataset(DATA_DIR)
loader = DataLoader(dataset, batch_size=1, shuffle=False)

def backprojectTo3d(output, T_wc, K):
    # 1. Get Depth (Z-buffer) from MoGe
    # (Note: MoGe-2 depth is often just the 3rd channel of 'points')
    depth = output['depth'].squeeze(0) # (H, W)
    h, w = depth.shape
    print("Depth shape:", depth.shape)
    print("K shape:", K.shape)
    print("T_wc shape:", T_wc.shape)
    
    # 2. Create a grid of pixel coordinates
    i, j = torch.meshgrid(torch.arange(h), torch.arange(w), indexing='ij')
    pixels = torch.stack([j, i, torch.ones_like(i)], dim=-1).float().to(depth.device) # (H, W, 3)
    pixels = pixels.reshape(-1, 3) # (N, 3) [u, v, 1]
    print("Pixels shape:", pixels.shape)
    print(pixels.device)
    
    # 3. Apply Inverse Intrinsics: P_cam = Z * inv(K) @ [u, v, 1]^T
    K_inv = torch.inverse(K).to(depth.device)
    print(K_inv.device)
    # We multiply K_inv by the pixels, then scale by depth
    pts_cam = (K_inv @ pixels.T).T * depth.reshape(-1, 1) # (N, 3)
    print("Points in camera frame shape:", pts_cam.shape)
    print(pts_cam.device)
    
    # 4. Transform to World
    ones = torch.ones((pts_cam.shape[0], 1), device=pts_cam.device)
    pts_cam_homo = torch.cat([pts_cam, ones], dim=1)
    T_wc = T_wc.to(pts_cam_homo.device)
    pts_world = (T_wc @ pts_cam_homo.T).T
    
    return pts_world[:, :3] # (N, 3)

# Read the input image and convert to tensor (3, H, W) with RGB values normalized to [0, 1]
input_image = dataset[0]["rgb"].numpy() * 255  # 3,H,W                     
print("Input image shape:", input_image.shape)
input_image = torch.tensor(input_image / 255, dtype=torch.float32, device=device).unsqueeze(0)  # 1, 3, H, W  
print("Input tensor shape:", input_image.shape)  

# Infer 
output = model.infer(input_image)
# print(output["depth"])  # (1, H, W)
"""
`output` has keys "points", "depth", "mask", "normal" (optional) and "intrinsics",
The maps are in the same size as the input image. 
{
    "points": (H, W, 3),    # point map in OpenCV camera coordinate system (x right, y down, z forward). For MoGe-2, the point map is in metric scale.
    "depth": (H, W),        # depth map
    "normal": (H, W, 3)     # normal map in OpenCV camera coordinate system. (available for MoGe-2-normal)
    "mask": (H, W),         # a binary mask for valid pixels. 
    "intrinsics": (3, 3),   # normalized camera intrinsics
}
"""

points_3d = backprojectTo3d(output, dataset[0]["pose"], dataset[0]["K"])
print("3D points shape:", points_3d.shape)  # (N, 3)
print("Some 3D points:", points_3d[:30, :])  # Print first 10 points