import numpy as np

def normalize(v: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(v)
    if n == 0:
        raise ValueError("Zero-length vector")
    return v / n

def build_nadir_pose(x: float, y: float, z: float) -> np.ndarray:
    position = np.array([x, y, z], dtype=np.float32)

    # Scene appears Y-up; to look "down" onto the ground plane (X-Z), use -Y.
    forward = np.array([0.0, -1.0, 0.0], dtype=np.float32)
    world_up = np.array([0.0, 0.0, 1.0], dtype=np.float32)

    right = normalize(np.cross(forward, world_up))
    up = normalize(np.cross(right, forward))

    c2w = np.eye(4, dtype=np.float32)
    c2w[:3, 0] = right
    c2w[:3, 1] = up
    c2w[:3, 2] = -forward
    c2w[:3, 3] = position

    return c2w
