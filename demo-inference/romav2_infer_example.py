import torch
from romav2 import RoMaV2
import torchvision.transforms as T
from PIL import Image

CHECKPOINT_DIR = "../../../../../scratch/dynrecon/checkpoints/romav2.pt"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DATA_DIR = "../../../../../scratch/toponavgroup/indoor-topo-loc/datasets/rrc-lab-data/wheelchair-runs-20241220"
to_tensor = T.ToTensor()
# load pretrained model
model = RoMaV2().to(DEVICE)
# model.weights=torch.load(CHECKPOINT_DIR,map_location=DEVICE)
# model = model.to(DEVICE)
model.eval()

# Match densely for any image-like pair of inputs
img_A = Image.open(f"{DATA_DIR}/run-1-wheelchair-mapping/rgb/0.png")
img_B = Image.open(f"{DATA_DIR}/run-2-wheelchair-query/rgb/0.png")
img_A = to_tensor(img_A).unsqueeze(0).to(DEVICE)  # 1, 3, H, W
img_B = to_tensor(img_B).unsqueeze(0).to(DEVICE)  # 1, 3, H, W
print("Image A shape:", img_A.shape)

H_A, W_A = img_A.shape[2], img_A.shape[3]
H_B, W_B = img_B.shape[2], img_B.shape[3]
# preds = model.match(f"{DATA_DIR}/run-1-wheelchair-mapping/rgb/0.png", f"{DATA_DIR}/run-2-wheelchair-query/rgb/0.png")
preds = model.match(img_A, img_B)
print("Preds keys:", preds.keys())

# you can also run the forward method directly as 
# preds = model(img_A, img_B)

# Sample 5000 matches for estimation
matches, overlaps, precision_AB, precision_BA = model.sample(preds, 5000)

# Convert to pixel coordinates (RoMaV2 produces matches in [-1,1]x[-1,1])
kptsA, kptsB = model.to_pixel_coordinates(matches, H_A, W_A, H_B, W_B)
# print("Keypoints A shape:", kptsA.shape)
# print("Keypoints B shape:", kptsB.shape)

# Find a fundamental matrix (or anything else of interest)
# F, mask = cv2.findFundamentalMat(
#     kptsA.cpu().numpy(), kptsB.cpu().numpy(), ransacReprojThreshold=0.2, method=cv2.USAC_MAGSAC, confidence=0.999999, maxIters=10000
# )