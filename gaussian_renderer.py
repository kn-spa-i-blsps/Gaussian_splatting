import numpy as np


def load_splat(path: str):
    raw = np.fromfile(path, dtype=np.uint8)
    if raw.size % 32 != 0:
        raise ValueError(f"Invalid splat file size: {raw.size} bytes")
    count = raw.size // 32
    raw = raw.reshape(count, 32)

    float_part = raw[:, :24].reshape(-1).view(np.float32).reshape(count, 6)
    positions = float_part[:, :3].astype(np.float32)
    scales = float_part[:, 3:6].astype(np.float32)

    colors = raw[:, 24:27].astype(np.float32) / 255.0
    opacities = raw[:, 27].astype(np.float32) / 255.0

    rotations = raw[:, 28:32].astype(np.float32)
    rotations = rotations / 255.0 * 2.0 - 1.0
    norms = np.linalg.norm(rotations, axis=1)
    mask = norms > 0
    rotations[mask] /= norms[mask][:, None]
    rotations[~mask] = np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32)

    return positions, scales, colors, opacities, rotations


def quat_to_matrix(q: np.ndarray) -> np.ndarray:
    x, y, z, w = q
    xx = x * x
    yy = y * y
    zz = z * z
    xy = x * y
    xz = x * z
    yz = y * z
    wx = w * x
    wy = w * y
    wz = w * z
    return np.array(
        [
            [1.0 - 2.0 * (yy + zz), 2.0 * (xy - wz), 2.0 * (xz + wy)],
            [2.0 * (xy + wz), 1.0 - 2.0 * (xx + zz), 2.0 * (yz - wx)],
            [2.0 * (xz - wy), 2.0 * (yz + wx), 1.0 - 2.0 * (xx + yy)],
        ],
        dtype=np.float32,
    )


def invert_c2w(c2w: np.ndarray) -> np.ndarray:
    r = c2w[:3, :3]
    t = c2w[:3, 3]
    w2c = np.eye(4, dtype=np.float32)
    w2c[:3, :3] = r.T
    w2c[:3, 3] = -r.T @ t
    return w2c


def render_gaussian(
    c2w: np.ndarray,
    splat_path: str = "splats/BS.splat",
    width: int = 512,
    height: int = 512,
    fov_deg: float = 70.0,
    near: float = 0.01,
    max_splats: int | None = None,
) -> np.ndarray:
    positions, scales, colors, opacities, rotations = load_splat(splat_path)
    if max_splats is not None:
        positions = positions[:max_splats]
        scales = scales[:max_splats]
        colors = colors[:max_splats]
        opacities = opacities[:max_splats]
        rotations = rotations[:max_splats]

    w2c = invert_c2w(c2w)
    r = w2c[:3, :3]
    t = w2c[:3, 3]
    pos_cam = positions @ r.T + t

    # Camera looks along -Z by default; if that yields nothing, fall back to +Z.
    z_front = -pos_cam[:, 2]
    z_alt = pos_cam[:, 2]
    valid_front = z_front > near
    valid_alt = z_alt > near
    if valid_front.sum() == 0 and valid_alt.sum() > 0:
        z = z_alt
    else:
        z = z_front
    valid = z > near
    pos_cam = pos_cam[valid]
    scales = scales[valid]
    colors = colors[valid]
    opacities = opacities[valid]
    rotations = rotations[valid]
    z = z[valid]

    fov_rad = np.deg2rad(fov_deg)
    fx = (width * 0.5) / np.tan(fov_rad * 0.5)
    fy = (height * 0.5) / np.tan(fov_rad * 0.5)
    cx = (width - 1) * 0.5
    cy = (height - 1) * 0.5

    x_proj = fx * (pos_cam[:, 0] / z) + cx
    y_proj = fy * (pos_cam[:, 1] / z) + cy

    order = np.argsort(z)

    img = np.zeros((height, width, 3), dtype=np.float32)
    alpha_acc = np.zeros((height, width), dtype=np.float32)

    total = order.size
    if total == 0:
        return (img * 255.0).astype(np.uint8)
    print(f"Rendering {total} splats on CPU... (this can be slow)")

    for i, idx in enumerate(order):
        if i % 100000 == 0 and i > 0:
            print(f"  processed {i}/{total}")

        mean_x = x_proj[idx]
        mean_y = y_proj[idx]

        if (
            mean_x < -10.0
            or mean_x > width + 10.0
            or mean_y < -10.0
            or mean_y > height + 10.0
        ):
            continue

        z_inv = 1.0 / z[idx]
        j00 = fx * z_inv
        j11 = fy * z_inv
        j02 = -fx * pos_cam[idx, 0] * z_inv * z_inv
        j12 = -fy * pos_cam[idx, 1] * z_inv * z_inv
        j = np.array([[j00, 0.0, j02], [0.0, j11, j12]], dtype=np.float32)

        rot = quat_to_matrix(rotations[idx])
        s2 = scales[idx] * scales[idx]
        c3 = rot @ np.diag(s2) @ rot.T

        cov2 = j @ c3 @ j.T
        cov2[0, 0] += 1e-6
        cov2[1, 1] += 1e-6

        det = cov2[0, 0] * cov2[1, 1] - cov2[0, 1] * cov2[1, 0]
        if det <= 0.0:
            continue

        trace = cov2[0, 0] + cov2[1, 1]
        disc = trace * trace * 0.25 - det
        if disc < 0.0:
            continue
        max_eig = trace * 0.5 + np.sqrt(disc)
        if max_eig <= 0.0:
            continue

        radius = 3.0 * np.sqrt(max_eig)
        if radius < 0.5:
            continue

        xmin = int(max(0, np.floor(mean_x - radius)))
        xmax = int(min(width - 1, np.ceil(mean_x + radius)))
        ymin = int(max(0, np.floor(mean_y - radius)))
        ymax = int(min(height - 1, np.ceil(mean_y + radius)))
        if xmin > xmax or ymin > ymax:
            continue

        inv_det = 1.0 / det
        inv00 = cov2[1, 1] * inv_det
        inv01 = -cov2[0, 1] * inv_det
        inv11 = cov2[0, 0] * inv_det

        xs = np.arange(xmin, xmax + 1, dtype=np.float32)
        ys = np.arange(ymin, ymax + 1, dtype=np.float32)
        xx, yy = np.meshgrid(xs, ys)
        dx = xx - mean_x
        dy = yy - mean_y
        mahal = inv00 * dx * dx + 2.0 * inv01 * dx * dy + inv11 * dy * dy
        weight = np.exp(-0.5 * mahal)

        alpha = opacities[idx] * weight
        if np.max(alpha) <= 0.0:
            continue

        sl_alpha = alpha_acc[ymin : ymax + 1, xmin : xmax + 1]
        one_minus = 1.0 - sl_alpha
        contrib = (one_minus * alpha)[..., None] * colors[idx]
        img[ymin : ymax + 1, xmin : xmax + 1] += contrib
        alpha_acc[ymin : ymax + 1, xmin : xmax + 1] = sl_alpha + one_minus * alpha

    img = np.clip(img, 0.0, 1.0)
    return (img * 255.0).astype(np.uint8)
