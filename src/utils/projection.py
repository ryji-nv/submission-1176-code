"""
World position to image pixel projection for drawing circles on camera images.

Matches USD/Isaac Sim camera: right = cross(fwd, world_up), up = cross(right, fwd);
camera looks along -fwd. Uses aspect-ratio-correct focal and optional vertical flip
for renderers that output with (0,0) at bottom-left.
"""

from __future__ import annotations

import math
from typing import Tuple


def _vec3_sub(
    a: Tuple[float, float, float], b: Tuple[float, float, float]
) -> Tuple[float, float, float]:
    return (a[0] - b[0], a[1] - b[1], a[2] - b[2])


def _vec3_dot(a: Tuple[float, float, float], b: Tuple[float, float, float]) -> float:
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]


def _normalize(v: Tuple[float, float, float]) -> Tuple[float, float, float]:
    L = math.sqrt(_vec3_dot(v, v))
    if L < 1e-9:
        return v
    return (v[0] / L, v[1] / L, v[2] / L)


def _cross(
    a: Tuple[float, float, float], b: Tuple[float, float, float]
) -> Tuple[float, float, float]:
    return (
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    )


def project_world_to_fraction(
    world_pos: Tuple[float, float, float],
    camera_position: Tuple[float, float, float],
    camera_look_at: Tuple[float, float, float],
    *,
    horizontal_fov_deg: float = 82.0,
    aspect_ratio: float = 16 / 9,
    world_up: Tuple[float, float, float] = (0, 0, 1),
) -> Tuple[float, float] | None:
    """
    Project a 3D world point to (u, v) in [0, 1] image-fraction space (unclamped).
    Returns None if the point is behind the camera.

    u=0 is left, u=1 is right; v=0 is top, v=1 is bottom.
    aspect_ratio = image_width / image_height (default 16/9).
    """
    fwd = _normalize(_vec3_sub(camera_look_at, camera_position))
    right = _cross(fwd, world_up)
    rlen = math.sqrt(_vec3_dot(right, right))
    if rlen < 1e-9:
        return None
    right = (right[0] / rlen, right[1] / rlen, right[2] / rlen)
    up = _cross(right, fwd)

    to_pt = _vec3_sub(world_pos, camera_position)
    depth = _vec3_dot(to_pt, fwd)
    if depth <= 0.01:
        return None
    x_cam = _vec3_dot(to_pt, right)
    y_cam = _vec3_dot(to_pt, up)

    tan_h = math.tan(math.radians(horizontal_fov_deg / 2.0))
    tan_v = tan_h / aspect_ratio  # tan(v_fov/2) = tan(h_fov/2) * (h/w)

    u = 0.5 + (x_cam / depth) / (2.0 * tan_h)
    v = 0.5 - (y_cam / depth) / (2.0 * tan_v)
    return (u, v)


def project_world_to_pixel(
    world_pos: Tuple[float, float, float],
    camera_position: Tuple[float, float, float],
    camera_look_at: Tuple[float, float, float],
    image_width: int,
    image_height: int,
    *,
    horizontal_fov_deg: float = 82.0,
    world_up: Tuple[float, float, float] = (0, 0, 1),
    flip_y: bool = False,
    clamp: bool = True,
) -> Tuple[int, int] | None:
    """
    Project a 3D world point into 2D image pixel (px, py).
    Returns (px, py) or None if point is behind the camera.

    Camera axes match USD _set_camera_lookat: right = cross(fwd, up), up = cross(right, fwd).
    Image: row 0 = top (PIL/numpy/Replicator default). flip_y: if True, flip py for
    renderers that output row 0 at bottom.
    """
    fwd = _normalize(_vec3_sub(camera_look_at, camera_position))
    right = _cross(fwd, world_up)
    rlen = math.sqrt(_vec3_dot(right, right))
    if rlen < 1e-9:
        return None
    right = (right[0] / rlen, right[1] / rlen, right[2] / rlen)
    up = _cross(right, fwd)

    to_pt = _vec3_sub(world_pos, camera_position)
    depth = _vec3_dot(to_pt, fwd)
    if depth <= 0.01:  # behind camera
        return None
    x_cam = _vec3_dot(to_pt, right)
    y_cam = _vec3_dot(to_pt, up)

    half_w = image_width / 2.0
    half_h = image_height / 2.0
    h_fov_rad = math.radians(horizontal_fov_deg / 2.0)
    focal_x = half_w / math.tan(h_fov_rad)
    # Vertical FOV from aspect ratio so circles aren't vertically stretched
    v_fov_rad = 2.0 * math.atan(math.tan(h_fov_rad) * image_height / image_width)
    focal_y = half_h / math.tan(v_fov_rad / 2.0)

    u = half_w + (x_cam / depth) * focal_x
    v = half_h - (y_cam / depth) * focal_y

    px = int(round(u))
    py = int(round(v))
    if flip_y:
        py = image_height - 1 - py
    if clamp:
        px = max(0, min(image_width - 1, px))
        py = max(0, min(image_height - 1, py))
    return (px, py)
