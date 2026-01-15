import rerun as rr
import numpy as np
from romav2 import RoMaV2
import torch
import cv2
from PIL import Image
import os
import argparse
import time

# Constants
QUERY_PATH = "../../../../../scratch/toponavgroup/indoor-topo-loc/datasets/rrc-lab-data/wheelchair-runs-20241220/run-2-wheelchair-query"
REF_PATH = "../../../../../scratch/toponavgroup/indoor-topo-loc/datasets/rrc-lab-data/wheelchair-runs-20241220/run-1-wheelchair-mapping"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def vis_matches(q_idx, ref_idx, num_vis=50):
    print("Loading model...")
    roma = RoMaV2().to(device).eval()

    # 1. Load Images
    q_path = f"{QUERY_PATH}/rgb/{q_idx}.png"
    r_path = f"{REF_PATH}/rgb/{ref_idx}.png"

    img1 = cv2.imread(q_path)
    img2 = cv2.imread(r_path)

    h1, w1 = img1.shape[:2]
    h2, w2 = img2.shape[:2]

    # 2. Inference
    print("Running Roma matching...")
    with torch.inference_mode():
        # RomaV2 match usually takes paths or PIL
        preds = roma.match(q_path, r_path)
        matches, _, _, _ = roma.sample(preds, 2000)
        
        # To Pixel Coords
        kpts_q, kpts_ref = roma.to_pixel_coordinates(matches, h1, w1, h2, w2)
        kpts_q = kpts_q.cpu().numpy()
        kpts_ref = kpts_ref.cpu().numpy()

    # 3. Create Canvas (Side-by-Side)
    height = max(h1, h2)
    width = w1 + w2
    canvas = np.zeros((height, width, 3), dtype=np.uint8)
    canvas[:h1, :w1, :] = img1
    canvas[:h2, w1:w1+w2, :] = img2

    # 4. Draw Lines with OpenCV
    for i in range(min(num_vis, len(kpts_q))):
        pt1 = (int(kpts_q[i][0]), int(kpts_q[i][1]))
        pt2 = (int(kpts_ref[i][0] + w1), int(kpts_ref[i][1]))
        
        cv2.line(canvas, pt1, pt2, (255, 255, 0), 1, cv2.LINE_AA) # Cyan-ish line
        cv2.circle(canvas, pt1, 2, (0, 0, 255), -1) # Red dot query
        cv2.circle(canvas, pt2, 2, (0, 255, 0), -1) # Green dot ref

    # 5. Log to Rerun
    canvas_rgb = cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB)
    rr.log("debug/matches", rr.Image(canvas_rgb))
    
    print(f"Done")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("q", type=int)
    parser.add_argument("ref", type=int)
    parser.add_argument("n", type=int, nargs='?', default=50)
    args = parser.parse_args()

    rr.init("RomaV2_Debug", spawn=True)
    rr.connect_grpc()
    vis_matches(args.q, args.ref, args.n)

    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        rr.disconnect()