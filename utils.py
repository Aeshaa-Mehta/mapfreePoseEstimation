import numpy as np

def r_error(R1, R2):
    """Compute rotation error in degrees between two rotation matrices."""
    R_diff = R1.T @ R2
    trace = np.clip((np.trace(R_diff) - 1) / 2, -1.0, 1.0)
    angle_rad = np.arccos(trace)
    angle_deg = np.degrees(angle_rad)
    return angle_deg

def t_error(t1, t2):
    """Compute translation error (Euclidean distance) between two translation vectors."""
    return np.linalg.norm(t1 - t2)  

def pose_error(T1, T2):
    """Compute rotation and translation error between two poses."""
    R1, t1 = T1[:3, :3], T1[:3, 3]
    R2, t2 = T2[:3, :3], T2[:3, 3]
    
    rot_err = r_error(R1, R2)
    trans_err = t_error(t1, t2)             
    return rot_err, trans_err