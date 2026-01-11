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
from data.wheelchair_runs import WheelchairRunDataset
import torchvision.transforms as T
from PIL import Image


mapping_data_path = "../../../../../scratch/toponavgroup/indoor-topo-loc/datasets/rrc-lab-data/wheelchair-runs-20241220/run-1-wheelchair-mapping"

mapping_dataset = WheelchairRunDataset(mapping_data_path)
mapping_loader = DataLoader(mapping_dataset, batch_size=1, shuffle=False)

rr.init("mapping_dataset", spawn=False)
rr.connect_grpc() 

with torch.no_grad():
        for i,batch in enumerate(mapping_loader):

            mapping_img_np = (batch["rgb"][0].permute(1, 2, 0).numpy() * 255).astype(np.uint8)
            rr.log("world/mapping", rr.Image(mapping_img_np))

try:
        while True:
            time.sleep(1)
except KeyboardInterrupt:
        print("Disconnecting...")
        rr.disconnect() 