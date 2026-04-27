"""
GSPhotoSource — replaces the drone camera for simulation runs.

Tracks a simulated position in the Gaussian-splat scene and converts
between real-world metres (used by the VLM and the grid overlay) and
scene units (used by the renderer) via a single scale factor:

    metres_per_unit  —  how many real metres one scene unit represents.

All public state (start positions, apply_move arguments) is in metres.
Scene-unit positions are kept internally and only used for rendering.
"""

from __future__ import annotations

import io
import json
import os
from datetime import datetime

import numpy as np
from PIL import Image

from camera_pose import build_nadir_pose
from gaussian_renderer_gpu import GaussianRenderer


class GSPhotoSource:
    """
    Render-based photo source that uses a pre-loaded Gaussian splat scene.

    Parameters
    ----------
    splat_path : str
        Path to the .splat file.
    upload_dir, telemetry_dir : str
        Output directories (mirror Config.upload_dir / Config.telemetry_dir).
    start_x, start_y, start_z : float
        Initial camera position in **metres**.
        Y is altitude (camera height above scene ground plane).
    metres_per_unit : float
        Real-world metres that correspond to one scene unit.
        Calibrate by measuring a known distance in the splat against
        its real-world length.  Default 1.0 (scene units == metres).
    width, height : int
        Render resolution in pixels.
    fov_deg : float
        Horizontal field-of-view for the renderer.
    jpeg_quality : int
        JPEG compression quality (1-95).
    device : str
        'auto' | 'cuda' | 'cpu'
    max_splats : int | None
        Cap splat count (useful for fast dev renders).
    """

    def __init__(
        self,
        splat_path: str,
        upload_dir: str,
        telemetry_dir: str,
        start_x: float = 0.0,
        start_y: float = 50.0,
        start_z: float = 0.0,
        metres_per_unit: float = 1.0,
        width: int = 1024,
        height: int = 1024,
        fov_deg: float = 70.0,
        jpeg_quality: int = 95,
        splat_radius: float = 3.0,
        ewa_min: float = 0.0,
        max_anisotropy: float = 10.0,
        supersample: int = 2,
        device: str = "auto",
        max_splats: int | None = None,
    ) -> None:
        self.upload_dir = upload_dir
        self.telemetry_dir = telemetry_dir
        self.width = width
        self.height = height
        self.fov_deg = fov_deg
        self.jpeg_quality = jpeg_quality
        self.splat_radius = splat_radius
        self.ewa_min = ewa_min
        self.max_anisotropy = max_anisotropy
        self.supersample = supersample
        self.metres_per_unit = metres_per_unit

        # Internal position stored in metres.
        self._x_m = float(start_x)
        self._y_m = float(start_y)   # altitude
        self._z_m = float(start_z)

        self._renderer = GaussianRenderer(
            splat_path=splat_path,
            device=device,
            max_splats=max_splats,
        )
        print(
            f"[GSPhotoSource] scale={metres_per_unit} m/unit  "
            f"start=({start_x}, {start_y}, {start_z}) m"
        )

    # ------------------------------------------------------------------
    # Position management  (all values in metres)
    # ------------------------------------------------------------------

    def apply_move(self, east_m: float, north_m: float, up_m: float) -> None:
        """
        Apply a VLM-returned (east, north, up) offset in metres.

        VLM convention (parsers.py):  east=x, north=y, up=z  (negative = descend)
        GS world axes:                East=X,  Up=Y,  North=Z
        """
        self._x_m += east_m
        self._z_m += north_m
        self._y_m += up_m
        print(
            f"[GSPhotoSource] position → "
            f"east={self._x_m:.1f}m  alt={self._y_m:.1f}m  north={self._z_m:.1f}m"
        )

    @property
    def position_m(self) -> tuple[float, float, float]:
        """Current position as (east_m, alt_m, north_m)."""
        return self._x_m, self._y_m, self._z_m

    # ------------------------------------------------------------------
    # Internal: metres → scene units
    # ------------------------------------------------------------------

    @property
    def _x_u(self) -> float:
        return self._x_m / self.metres_per_unit

    @property
    def _y_u(self) -> float:
        return self._y_m / self.metres_per_unit

    @property
    def _z_u(self) -> float:
        return self._z_m / self.metres_per_unit

    # ------------------------------------------------------------------
    # Capture
    # ------------------------------------------------------------------

    def capture_and_save(self) -> tuple[str, str]:
        """
        Render the current view, write JPEG + telemetry JSON to disk.

        Returns (photo_path, telemetry_path).
        """
        ts = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        photo_path = os.path.join(self.upload_dir, f"gs_{ts}.jpg")
        telemetry_path = os.path.join(self.telemetry_dir, f"gs_{ts}.json")

        # Render using scene-unit coordinates.
        c2w = build_nadir_pose(self._x_u, self._y_u, self._z_u)
        rgb: np.ndarray = self._renderer.render(
            c2w,
            width=self.width,
            height=self.height,
            fov_deg=self.fov_deg,
            splat_radius=self.splat_radius,
            ewa_min=self.ewa_min,
            max_anisotropy=self.max_anisotropy,
            supersample=self.supersample,
        )

        img = Image.fromarray(rgb)
        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=self.jpeg_quality)
        with open(photo_path, "wb") as fh:
            fh.write(buf.getvalue())

        # Telemetry in metres — parse_telemetry() reads ["data"]["position"]["alt"].
        telemetry = {
            "data": {
                "position": {
                    "alt": self._y_m,
                    "east": self._x_m,
                    "north": self._z_m,
                },
                "camera_fov_deg": self.fov_deg,
            }
        }
        with open(telemetry_path, "w", encoding="utf-8") as fh:
            json.dump(telemetry, fh)

        print(f"[GSPhotoSource] rendered at ({self._x_u:.3f}, {self._y_u:.3f}, {self._z_u:.3f}) units  →  {photo_path}")
        return photo_path, telemetry_path
