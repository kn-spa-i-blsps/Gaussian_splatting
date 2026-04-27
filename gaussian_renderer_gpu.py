"""
GPU-accelerated Gaussian splatting renderer.

Architecture:
  - GaussianRenderer loads splats once from disk and keeps them on device.
  - render() vectorises ALL pre-processing (transform → project → covariance →
    eigenvalue → bbox) using PyTorch (CUDA if available, else CPU tensors).
  - Sequential alpha-compositing is JIT-compiled with Numba when installed,
    otherwise falls back to a plain NumPy loop.

Speedup over gaussian_renderer.py:
  - Splats loaded once instead of per-render.
  - Covariance, Jacobian, and eigenvalue math vectorised in batch.
  - Numba JIT compiles the inner rasterisation loop to native code (~50-100x
    faster than the Python loop for the compositing step).
"""

from __future__ import annotations

import numpy as np

try:
    import torch
    _TORCH_AVAILABLE = True
except ImportError:
    _TORCH_AVAILABLE = False

try:
    from numba import njit as _njit
    import numba as _numba

    @_njit(cache=True)
    def _composite_numba(
        xp: np.ndarray,
        yp: np.ndarray,
        radii: np.ndarray,
        inv00: np.ndarray,
        inv01: np.ndarray,
        inv11: np.ndarray,
        opacities: np.ndarray,
        colors: np.ndarray,
        width: int,
        height: int,
    ) -> np.ndarray:
        img = np.zeros((height, width, 3), dtype=np.float32)
        alpha_acc = np.zeros((height, width), dtype=np.float32)

        for i in range(xp.shape[0]):
            mx = xp[i]
            my = yp[i]
            r = radii[i]

            xmin = max(0, int(np.floor(mx - r)))
            xmax = min(width - 1, int(np.ceil(mx + r)))
            ymin = max(0, int(np.floor(my - r)))
            ymax = min(height - 1, int(np.ceil(my + r)))

            if xmin > xmax or ymin > ymax:
                continue

            for py in range(ymin, ymax + 1):
                for px in range(xmin, xmax + 1):
                    dx = float(px) - mx
                    dy = float(py) - my
                    mahal = (
                        inv00[i] * dx * dx
                        + 2.0 * inv01[i] * dx * dy
                        + inv11[i] * dy * dy
                    )
                    alpha = opacities[i] * np.exp(-0.5 * mahal)
                    old_a = alpha_acc[py, px]
                    one_minus = 1.0 - old_a
                    contrib = one_minus * alpha
                    img[py, px, 0] += contrib * colors[i, 0]
                    img[py, px, 1] += contrib * colors[i, 1]
                    img[py, px, 2] += contrib * colors[i, 2]
                    alpha_acc[py, px] = old_a + contrib

        return img

    _NUMBA_AVAILABLE = True
except ImportError:
    _NUMBA_AVAILABLE = False


def _composite_numpy(
    xp: np.ndarray,
    yp: np.ndarray,
    radii: np.ndarray,
    inv00: np.ndarray,
    inv01: np.ndarray,
    inv11: np.ndarray,
    opacities: np.ndarray,
    colors: np.ndarray,
    width: int,
    height: int,
) -> np.ndarray:
    img = np.zeros((height, width, 3), dtype=np.float32)
    alpha_acc = np.zeros((height, width), dtype=np.float32)

    for i in range(xp.shape[0]):
        mx, my, r = float(xp[i]), float(yp[i]), float(radii[i])

        xmin = max(0, int(np.floor(mx - r)))
        xmax = min(width - 1, int(np.ceil(mx + r)))
        ymin = max(0, int(np.floor(my - r)))
        ymax = min(height - 1, int(np.ceil(my + r)))

        if xmin > xmax or ymin > ymax:
            continue

        xs = np.arange(xmin, xmax + 1, dtype=np.float32) - mx
        ys = np.arange(ymin, ymax + 1, dtype=np.float32) - my
        dx, dy = np.meshgrid(xs, ys)
        mahal = inv00[i] * dx * dx + 2.0 * inv01[i] * dx * dy + inv11[i] * dy * dy
        alpha = opacities[i] * np.exp(-0.5 * mahal)

        sl = alpha_acc[ymin : ymax + 1, xmin : xmax + 1]
        one_minus = 1.0 - sl
        img[ymin : ymax + 1, xmin : xmax + 1] += (one_minus * alpha)[..., None] * colors[i]
        alpha_acc[ymin : ymax + 1, xmin : xmax + 1] = sl + one_minus * alpha

    return img


class GaussianRenderer:
    """
    Loads a .splat file once and renders arbitrary camera views.

    Parameters
    ----------
    splat_path : str
        Path to the binary .splat file.
    device : str
        'cuda', 'cpu', or 'auto' (picks CUDA when available).
    max_splats : int | None
        Cap the number of splats loaded (useful for debugging).
    """

    def __init__(
        self,
        splat_path: str,
        device: str = "auto",
        max_splats: int | None = None,
    ) -> None:
        if not _TORCH_AVAILABLE:
            raise RuntimeError(
                "PyTorch is required for GaussianRenderer. "
                "Install it with: pip install torch"
            )

        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device

        if _NUMBA_AVAILABLE:
            backend = "Numba JIT"
        else:
            backend = "NumPy loop"
        print(
            f"[GaussianRenderer] device={device}  compositing={backend}"
        )

        self._load(splat_path, max_splats)

    # ------------------------------------------------------------------
    # Data loading
    # ------------------------------------------------------------------

    def _load(self, path: str, max_splats: int | None) -> None:
        if path.endswith(".compressed.ply"):
            from gaussian_renderer import load_compressed_ply
            pos, scales, colors, opacities, rots = load_compressed_ply(path)
        else:
            from gaussian_renderer import load_splat
            pos, scales, colors, opacities, rots = load_splat(path)
        if max_splats is not None:
            pos = pos[:max_splats]
            scales = scales[:max_splats]
            colors = colors[:max_splats]
            opacities = opacities[:max_splats]
            rots = rots[:max_splats]

        dev = self.device
        self._pos = torch.from_numpy(pos).to(dev)
        self._scales = torch.from_numpy(scales).to(dev)
        self._colors = torch.from_numpy(colors).to(dev)
        self._opacities = torch.from_numpy(opacities).to(dev)
        self._rots = torch.from_numpy(rots).to(dev)
        print(f"[GaussianRenderer] loaded {self._pos.shape[0]:,} splats from {path}")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def render(
        self,
        c2w: np.ndarray,
        width: int = 512,
        height: int = 512,
        fov_deg: float = 70.0,
        near: float = 0.01,
        splat_radius: float = 3.0,
        ewa_min: float = 0.0,
        max_anisotropy: float = 10.0,
        supersample: int = 1,
    ) -> np.ndarray:
        """
        Render the scene from the given camera-to-world pose.

        Parameters
        ----------
        c2w : np.ndarray  shape (4, 4)
        width, height : int
            Output image dimensions in pixels.
        fov_deg : float
            Horizontal field-of-view in degrees.
        near : float
            Near-plane clipping distance.
        splat_radius : float
            Standard deviations around each Gaussian to rasterise.
            3.0 covers 99.7 % of each splat's signal.
        ewa_min : float
            EWA minimum screen-space variance (0.0 = off).
        max_anisotropy : float
            Cap on major/minor eigenvalue ratio — prevents streak artifacts.
        supersample : int
            Render at (width*supersample) × (height*supersample) then
            Lanczos-downsample to (width, height).  2 is a good default;
            higher values cost O(supersample²) compositing time.

        Returns
        -------
        np.ndarray  dtype=uint8  shape (height, width, 3)
        """
        if supersample > 1:
            from PIL import Image as _Image
            raw = self._render(
                c2w,
                width * supersample,
                height * supersample,
                fov_deg, near, splat_radius, ewa_min, max_anisotropy,
            )
            return np.array(
                _Image.fromarray(raw).resize((width, height), _Image.LANCZOS)
            )
        return self._render(c2w, width, height, fov_deg, near, splat_radius, ewa_min, max_anisotropy)

    # ------------------------------------------------------------------
    # Internal rendering pipeline
    # ------------------------------------------------------------------

    def _render(
        self,
        c2w: np.ndarray,
        width: int,
        height: int,
        fov_deg: float,
        near: float,
        splat_radius: float,
        ewa_min: float,
        max_anisotropy: float,
    ) -> np.ndarray:
        from gaussian_renderer import invert_c2w

        dev = self.device

        # ---- world → camera transform ----
        w2c = invert_c2w(c2w)
        R = torch.from_numpy(w2c[:3, :3]).to(dev)
        T = torch.from_numpy(w2c[:3, 3]).to(dev)

        # ---- transform positions to camera space ----
        pos_cam = self._pos @ R.T + T  # (N, 3)

        # ---- depth selection (handle -Z and +Z conventions) ----
        z_neg = -pos_cam[:, 2]
        z_pos = pos_cam[:, 2]
        if (z_neg > near).sum() >= (z_pos > near).sum():
            z = z_neg
        else:
            z = z_pos

        valid = z > near
        pc = pos_cam[valid]
        sc = self._scales[valid]
        co = self._colors[valid]
        op = self._opacities[valid]
        ro = self._rots[valid]
        z = z[valid]

        n = pc.shape[0]
        if n == 0:
            return np.zeros((height, width, 3), dtype=np.uint8)

        # ---- camera intrinsics ----
        fov_rad = float(np.deg2rad(fov_deg))
        fx = float((width * 0.5) / np.tan(fov_rad * 0.5))
        fy = float((height * 0.5) / np.tan(fov_rad * 0.5))
        cx = float((width - 1) * 0.5)
        cy = float((height - 1) * 0.5)

        # ---- project to 2-D ----
        z_inv = 1.0 / z
        xp = fx * (pc[:, 0] * z_inv) + cx
        yp = fy * (pc[:, 1] * z_inv) + cy

        # ---- batch quaternion → rotation matrices (N, 3, 3) ----
        rx, ry, rz, rw = ro[:, 0], ro[:, 1], ro[:, 2], ro[:, 3]
        xx, yy, zz = rx * rx, ry * ry, rz * rz
        xy, xz, yz = rx * ry, rx * rz, ry * rz
        wx, wy, wz = rw * rx, rw * ry, rw * rz

        Rm = torch.stack(
            [
                torch.stack([1 - 2 * (yy + zz), 2 * (xy - wz), 2 * (xz + wy)], dim=1),
                torch.stack([2 * (xy + wz), 1 - 2 * (xx + zz), 2 * (yz - wx)], dim=1),
                torch.stack([2 * (xz - wy), 2 * (yz + wx), 1 - 2 * (xx + yy)], dim=1),
            ],
            dim=1,
        )  # (N, 3, 3)

        # ---- 3-D covariance:  R @ diag(s²) @ Rᵀ ----
        s2 = sc * sc  # (N, 3)
        RS = Rm * s2.unsqueeze(1)  # (N, 3, 3) — each column scaled
        cov3 = RS @ Rm.transpose(1, 2)  # (N, 3, 3)

        # ---- Jacobian of perspective projection (N, 2, 3) ----
        J = torch.zeros(n, 2, 3, device=dev, dtype=torch.float32)
        J[:, 0, 0] = fx * z_inv
        J[:, 1, 1] = fy * z_inv
        J[:, 0, 2] = -fx * pc[:, 0] * z_inv * z_inv
        J[:, 1, 2] = -fy * pc[:, 1] * z_inv * z_inv

        # ---- 2-D covariance:  J @ cov3 @ Jᵀ ----
        cov2 = J @ cov3 @ J.transpose(1, 2)  # (N, 2, 2)

        # EWA anti-aliasing: optional minimum screen-space variance.
        cov2[:, 0, 0] += ewa_min
        cov2[:, 1, 1] += ewa_min

        # ---- eigenvalues ----
        det = cov2[:, 0, 0] * cov2[:, 1, 1] - cov2[:, 0, 1] ** 2
        trace = cov2[:, 0, 0] + cov2[:, 1, 1]
        disc = torch.clamp(trace * trace * 0.25 - det, min=0.0)
        sqrt_disc = torch.sqrt(disc)
        max_eig = trace * 0.5 + sqrt_disc
        min_eig = torch.clamp(trace * 0.5 - sqrt_disc, min=0.0)

        # ---- anisotropy clamp: prevent spike/streak artifacts ----
        # Oblique splats projected nadir become needle-thin ellipses
        # (max_eig >> min_eig).  Boosting the diagonal by `boost` raises
        # both eigenvalues equally, capping the ratio at max_anisotropy.
        if max_anisotropy > 0:
            min_minor = max_eig / max_anisotropy
            boost = torch.clamp(min_minor - min_eig, min=0.0)
            cov2[:, 0, 0] += boost
            cov2[:, 1, 1] += boost
            # Recompute after clamping
            det = cov2[:, 0, 0] * cov2[:, 1, 1] - cov2[:, 0, 1] ** 2
            trace = cov2[:, 0, 0] + cov2[:, 1, 1]
            disc = torch.clamp(trace * trace * 0.25 - det, min=0.0)
            max_eig = trace * 0.5 + torch.sqrt(disc)

        radii = splat_radius * torch.sqrt(torch.clamp(max_eig, min=0.0))

        # ---- filter out degenerate / off-screen splats ----
        on_screen = (
            (det > 0)
            & (max_eig > 0)
            & (radii >= 0.5)
            & (xp > -10)
            & (xp < width + 10)
            & (yp > -10)
            & (yp < height + 10)
        )

        # ---- inverse 2-D covariance elements ----
        inv_det = 1.0 / det.clamp(min=1e-12)
        inv00 = cov2[:, 1, 1] * inv_det
        inv01 = -cov2[:, 0, 1] * inv_det
        inv11 = cov2[:, 0, 0] * inv_det

        # ---- depth-sort (front → back for front-to-back compositing) ----
        depth_order = torch.argsort(z[on_screen])
        vis_idx = on_screen.nonzero(as_tuple=True)[0][depth_order]

        # ---- move rasterisation data to CPU as contiguous float32 arrays ----
        def _cpu(t: torch.Tensor) -> np.ndarray:
            return t[vis_idx].contiguous().cpu().numpy().astype(np.float32)

        xp_cpu   = _cpu(xp)
        yp_cpu   = _cpu(yp)
        rad_cpu  = _cpu(radii)
        i00_cpu  = _cpu(inv00)
        i01_cpu  = _cpu(inv01)
        i11_cpu  = _cpu(inv11)
        op_cpu   = _cpu(op)
        co_cpu   = _cpu(co)

        total = xp_cpu.shape[0]
        print(f"[GaussianRenderer] compositing {total:,} visible splats ...")

        # ---- sequential alpha compositing ----
        if _NUMBA_AVAILABLE:
            img = _composite_numba(
                xp_cpu, yp_cpu, rad_cpu,
                i00_cpu, i01_cpu, i11_cpu,
                op_cpu, co_cpu, width, height,
            )
        else:
            img = _composite_numpy(
                xp_cpu, yp_cpu, rad_cpu,
                i00_cpu, i01_cpu, i11_cpu,
                op_cpu, co_cpu, width, height,
            )

        return np.clip(img * 255.0, 0, 255).astype(np.uint8)
