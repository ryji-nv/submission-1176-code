"""
Camera pose generators for SAGE / SceneSmith rooms (no USD/Isaac dependency).

Corner cameras: 4 corners × 2 heights, looking at room centroid.
Edge cameras:   4 wall midpoints (south/east/north/west), looking at room centroid.
Stepped cameras: forward/left/right translations from a base camera.
"""

from __future__ import annotations

import math


def get_corner_camera_poses(
    room_bounds: tuple[float, float, float, float, float],
    *,
    pad: float = 0.12,
    look_at_height_frac: float = 0.35,
) -> list[dict]:
    """
    Return list of camera pose dicts for SAGE-style corner cameras.

    room_bounds: (rx, ry, rw, rl, rh).
    Each entry: {"name": str, "position": (x,y,z), "look_at": (x,y,z)}.
    """
    rx, ry, rw, rl, rh = room_bounds
    room_cx = rx + rw / 2.0
    room_cy = ry + rl / 2.0
    look_z = rh * look_at_height_frac
    look_at = (room_cx, room_cy, look_z)

    corner_xy = [
        (rx + pad, ry + pad),
        (rx + rw - pad, ry + pad),
        (rx + pad, ry + rl - pad),
        (rx + rw - pad, ry + rl - pad),
    ]
    cam_z_top = rh - 0.15
    cam_z_mid = rh / 2.0

    out = []
    for i, (cx, cy) in enumerate(corner_xy):
        for z, name_suffix in (
            (cam_z_top, str(i)),
            (cam_z_mid, f"mid_{i}"),
        ):
            name = f"sage_corner_camera_{name_suffix}"
            out.append(
                {
                    "name": name,
                    "position": (cx, cy, z),
                    "look_at": look_at,
                }
            )
    return out


_EDGE_NAMES = ("south", "east", "north", "west")


def get_edge_camera_poses(
    room_bounds: tuple[float, float, float, float, float],
    *,
    pad: float = 0.12,
    look_at_height_frac: float = 0.35,
    prefix: str = "sage",
) -> list[dict]:
    """
    Return camera pose dicts at the midpoint of each wall edge.

    room_bounds: (rx, ry, rw, rl, rh).
    Order: south (min-y), east (max-x), north (max-y), west (min-x).
    Each entry has "edge_direction" identifying the wall.
    """
    rx, ry, rw, rl, rh = room_bounds
    cx = rx + rw / 2.0
    cy = ry + rl / 2.0
    look_at = (cx, cy, rh * look_at_height_frac)
    cam_z = rh - 0.15

    edge_xy = [
        (cx, ry + pad),  # south
        (rx + rw - pad, cy),  # east
        (cx, ry + rl - pad),  # north
        (rx + pad, cy),  # west
    ]

    return [
        {
            "name": f"{prefix}_edge_camera_{name}",
            "position": (ex, ey, cam_z),
            "look_at": look_at,
            "edge_direction": name,
        }
        for name, (ex, ey) in zip(_EDGE_NAMES, edge_xy)
    ]


def get_stepped_camera_poses(
    room_bounds: tuple[float, float, float, float, float],
    base_camera: dict,
    *,
    step_size: float = 0.5,
    pad: float = 0.12,
    directions: tuple[str, ...] = ("forward", "left", "right"),
) -> list[dict]:
    """
    Return stepped camera poses for base_camera: pure translation along
    forward/left/right, clamped to room bounds.

    room_bounds: (rx, ry, rw, rl, rh).
    base_camera: {"name": str, "position": (x,y,z), "look_at": (x,y,z), ...}.
    Returns dicts with extra keys "step_direction" and "step_from".
    """
    rx, ry, rw, rl, _ = room_bounds
    x_min, x_max = rx + pad, rx + rw - pad
    y_min, y_max = ry + pad, ry + rl - pad

    pos = base_camera["position"]
    look_at = base_camera["look_at"]

    dx, dy = look_at[0] - pos[0], look_at[1] - pos[1]
    dlen = math.sqrt(dx * dx + dy * dy)
    if dlen < 1e-9:
        return []
    fx, fy = dx / dlen, dy / dlen  # forward (XY plane)

    dir_vectors = {
        "forward": (fx, fy),
        "right": (fy, -fx),
        "left": (-fy, fx),
    }

    result = []
    for d in directions:
        vx, vy = dir_vectors[d]
        new_x = max(x_min, min(x_max, pos[0] + step_size * vx))
        new_y = max(y_min, min(y_max, pos[1] + step_size * vy))
        delta_x = new_x - pos[0]
        delta_y = new_y - pos[1]
        result.append(
            {
                "name": f"{base_camera['name']}_step_{d}",
                "position": (new_x, new_y, pos[2]),
                "look_at": (look_at[0] + delta_x, look_at[1] + delta_y, look_at[2]),
                "step_direction": d,
                "step_from": base_camera["name"],
            }
        )
    return result
