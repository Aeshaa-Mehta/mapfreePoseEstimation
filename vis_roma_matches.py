import rerun as rr
import numpy as np
from romav2 import RomaV2

QUERY_PATH = "../../../../../scratch/toponavgroup/indoor-topo-loc/datasets/rrc-lab-data/wheelchair-runs-20241220/run-2-wheelchair-query"
REF_PATH = "../../../../../scratch/toponavgroup/indoor-topo-loc/datasets/rrc-lab-data/wheelchair-runs-20241220/run-1-wheelchair-mapping"


def vis_matchess(q_idx, ref_idx):
    rr.init("roma_matches_example", spawn=False)

    # Load the RomaV2 model
    roma = RomaV2()

    # Load query and reference images along with their keypoints and descriptors
    q_img, q_kps, q_descs = roma.load_image_and_features(q_idx)
    ref_img, ref_kps, ref_descs = roma.load_image_and_features(ref_idx)

    # Match descriptors between query and reference images
    matches = roma.match_descriptors(q_descs, ref_descs)

    # Visualize the query image with keypoints
    rr.log_image("query_image", q_img)
    rr.log_points("query_keypoints", q_kps, colors=[255, 0, 0], radii=2)

    # Visualize the reference image with keypoints
    rr.log_image("reference_image", ref_img)
    rr.log_points("reference_keypoints", ref_kps, colors=[0, 255, 0], radii=2)

    # Visualize matches
    matched_q_kps = q_kps[matches[:, 0]]
    matched_ref_kps = ref_kps[matches[:, 1]]

    for (q_pt, ref_pt) in zip(matched_q_kps, matched_ref_kps):
        rr.log_line("matches", [q_pt, ref_pt], colors=[0, 0, 255])