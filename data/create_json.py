# import faiss
# import numpy as np
# import torch
# import torchvision.transforms as T
# from PIL import Image
# import cv2
# import json
# from tqdm import tqdm
# from matplotlib import pyplot as plt
# import os
# # import supervision as sv


# ROOT_DIR = "../../../../../scratch/toponavgroup/indoor-topo-loc/datasets/rrc-lab-data/wheelchair-runs-20241220/run-1-wheelchair-mapping/rgb"
# files = os.listdir(ROOT_DIR)
# files = [os.path.join(ROOT_DIR, f) for f in files if f.lower().endswith(".png")]

# dinov2_vits14 = torch.hub.load("facebookresearch/dinov2", "dinov2_vits14")

# device = torch.device('cuda' if torch.cuda.is_available() else "cpu")

# dinov2_vits14.to(device)

# transform_image = T.Compose([
#     T.Resize(224, interpolation=T.InterpolationMode.BICUBIC),
#     T.CenterCrop(224),
#     T.ToTensor(),
#     T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
# ])

# def load_image(img: str) -> torch.Tensor:
#     """
#     Load an image and return a tensor that can be used as an input to DINOv2.
#     """
#     img = Image.open(img)

#     transformed_img = transform_image(img)[:3].unsqueeze(0)

#     return transformed_img

# def create_index(files: list) -> faiss.IndexFlatL2:
#     """
#     Create an index that contains all of the images in the specified list of files.
#     """
#     index = faiss.IndexFlatL2(384)

#     all_embeddings = {}
    
#     with torch.no_grad():
#       for i, file in enumerate(tqdm(files)):
#         embeddings = dinov2_vits14(load_image(file).to(device))

#         embedding = embeddings[0].cpu().numpy()

#         all_embeddings[file] = np.array(embedding).reshape(1, -1).tolist()

#         index.add(np.array(embedding).reshape(1, -1))

#     with open("all_embeddings.json", "w") as f:
#         f.write(json.dumps(all_embeddings))

#     faiss.write_index(index, "data.bin")

#     return index, all_embeddings


# if __name__ == "__main__":
#     index, embeddings = create_index(files)

# import torch
# import faiss
# import numpy as np
# import os
# from PIL import Image
# import torchvision.transforms as T

# # --- 1. SETUP (Must match the creation script) ---
# device = torch.device('cuda' if torch.cuda.is_available() else "cpu")
# model = torch.hub.load("facebookresearch/dinov2", "dinov2_vits14").to(device).eval()

# transform_image = T.Compose([
#     T.Resize(224, interpolation=T.InterpolationMode.BICUBIC),
#     T.CenterCrop(224),
#     T.ToTensor(),
#     T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
# ])

# # Re-create the file list exactly as you did in the creation script
# ROOT_DIR = "../../../../../scratch/toponavgroup/indoor-topo-loc/datasets/rrc-lab-data/wheelchair-runs-20241220/run-1-wheelchair-mapping/rgb"
# # Use sorted() to ensure the indices match the index.add order
# files = sorted([os.path.join(ROOT_DIR, f) for f in os.listdir(ROOT_DIR) if f.lower().endswith(".png")])

# # Load the saved FAISS index
# index = faiss.read_index("data.bin")

# def search_similar_images(query_img_path, k=5):
#     """
#     Finds the top k most similar images to the query image.
#     """
#     # 1. Load and Transform Query Image
#     img = Image.open(query_img_path).convert('RGB')
#     img_t = transform_image(img).unsqueeze(0).to(device)

#     # 2. Extract Query Embedding
#     with torch.no_grad():
#         query_embedding = model(img_t).cpu().numpy()

#     # 3. Search Index
#     # D: Distances (L2 distance, so lower is better)
#     # I: Indices of the matching images in the 'files' list
#     distances, indices = index.search(query_embedding.astype('float32'), k)

#     results = []
#     for i in range(k):
#         match_idx = indices[0][i]
#         match_dist = distances[0][i]
#         results.append({
#             "path": files[match_idx],
#             "index": match_idx,
#             "distance": match_dist
#         })
    
#     return results

# # --- 2. EXECUTION ---
# if __name__ == "__main__":
#     # Example Query
#     QUERY_PATH = "../../../../../scratch/toponavgroup/indoor-topo-loc/datasets/rrc-lab-data/wheelchair-runs-20241220/run-1-wheelchair-mapping/rgb/2867.png"

#     matches = search_similar_images(QUERY_PATH, k=1)
    
#     best_match = matches[0]
#     print(f"Best Match Found!")
#     print(f"Reference Image: {best_match['path']}")
#     print(f"L2 Distance: {best_match['distance']:.4f}")

import os
import json
import torch
import faiss
import numpy as np
import torchvision.transforms as T
from PIL import Image
from tqdm import tqdm
import re

def natural_sort_key(s):
    return [int(text) if text.isdigit() else text.lower()
            for text in re.split('([0-9]+)', s)]

# --- 1. SETUP ---
ROOT_DIR = "../../../../../scratch/toponavgroup/indoor-topo-loc/datasets/rrc-lab-data/wheelchair-runs-20241220/run-1-wheelchair-mapping/rgb"

# CRITICAL FIX: Always use sorted() so the order is identical every time
files = sorted([os.path.join(ROOT_DIR, f) for f in os.listdir(ROOT_DIR) if f.lower().endswith(".png")], key = natural_sort_key)

device = torch.device('cuda' if torch.cuda.is_available() else "cpu")
model = torch.hub.load("facebookresearch/dinov2", "dinov2_vits14").to(device).eval()

# Use standard ImageNet normalization
transform_image = T.Compose([
    T.Resize(224, interpolation=T.InterpolationMode.BICUBIC),
    T.CenterCrop(224),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

def create_index():
    # IndexFlatIP (Inner Product) + Normalization = Cosine Similarity (better than L2)
    index = faiss.IndexFlatIP(384) 
    file_database = [] # We will save this list to a JSON

    with torch.no_grad():
        for i, file_path in enumerate(tqdm(files)):
            img = Image.open(file_path).convert('RGB')
            img_t = transform_image(img).unsqueeze(0).to(device)
            
            # Extract and Normalize
            emb = model(img_t)
            emb = torch.nn.functional.normalize(emb, p=2, dim=-1).cpu().numpy()

            index.add(emb.astype('float32'))
            file_database.append(file_path)

    # SAVE BOTH TOGETHER
    faiss.write_index(index, "data.bin")
    with open("file_database.json", "w") as f:
        json.dump(file_database, f)
    
    print("Database created and saved.")

if __name__ == "__main__":
    create_index()


# import torch
# import faiss
# import json
# import numpy as np
# from PIL import Image
# import torchvision.transforms as T

# # --- 1. LOAD DATABASE ---
# index = faiss.read_index("data.bin")
# with open("file_database.json", "r") as f:
#     file_database = json.load(f)

# device = torch.device('cuda' if torch.cuda.is_available() else "cpu")
# model = torch.hub.load("facebookresearch/dinov2", "dinov2_vits14").to(device).eval()

# transform_image = T.Compose([
#     T.Resize(224, interpolation=T.InterpolationMode.BICUBIC),
#     T.CenterCrop(224),
#     T.ToTensor(),
#     T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
# ])

# def search(query_path):
#     img = Image.open(query_path).convert('RGB')
#     img_t = transform_image(img).unsqueeze(0).to(device)

#     with torch.no_grad():
#         emb = model(img_t)
#         emb = torch.nn.functional.normalize(emb, p=2, dim=-1).cpu().numpy()

#     # Search for top 1
#     # For IndexFlatIP, higher is better (1.0000 is perfect match)
#     similarities, indices = index.search(emb.astype('float32'), 1)
    
#     best_idx = indices[0][0]
#     best_sim = similarities[0][0]
    
#     print(f"Query: {query_path}")
#     print(f"Match: {file_database[best_idx]}") # Look up filename in the saved JSON
#     print(f"Similarity Score: {best_sim:.4f}")

# if __name__ == "__main__":
#     # TEST: Use an image that IS in the database
#     # It should return the exact same path and a similarity of 1.0000
#     TEST_PATH = file_database[200] 
#     search(TEST_PATH)