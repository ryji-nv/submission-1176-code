"""
Surface detection and placement feasibility utilities.

Uses normals maps, instance segmentation, and 3D bounding-box geometry to
identify horizontal surfaces and check whether an object footprint fits.
"""

from __future__ import annotations

import json
import os
from typing import Tuple

import numpy as np

from src.utils.projection import project_world_to_fraction

_SURFACE_TYPE_KEYWORDS = frozenset(
    [
        "table",
        "desk",
        "shelf",
        "counter",
        "bench",
        "nightstand",
        "dresser",
        "stand",
        "cabinet",
        "sideboard",
        "console",
    ]
)


def _type_is_surface(obj_type: str) -> bool:
    t = obj_type.lower().replace("-", "_")
    return any(kw in t for kw in _SURFACE_TYPE_KEYWORDS)


def identify_surface_objects(objects: list[dict]) -> list[dict]:
    """Return objects that qualify as placement surfaces (keyword or flat geometry)."""
    surfaces = []
    for obj in objects:
        w = obj.get("width", 0)
        l = obj.get("length", 0)
        h = obj.get("height", 0)
        if w <= 0 or l <= 0 or h <= 0:
            continue
        obj_type = str(obj.get("type", obj.get("id", "")))
        if _type_is_surface(obj_type):
            surfaces.append(obj)
            continue
        if w * l > 4 * h * max(w, l) and h >= 0.1:
            surfaces.append(obj)
    return surfaces


def surface_top_z(obj: dict) -> float:
    return obj.get("z", 0.0) + obj.get("height", 0.0)


def objects_on_surface(
    surface: dict, all_objects: list[dict], z_tolerance: float = 0.15
) -> list[dict]:
    """Return objects whose base sits on *surface* within z_tolerance."""
    stop_z = surface_top_z(surface)
    sx, sy = surface["x"], surface["y"]
    sw, sl = surface.get("width", 0) / 2, surface.get("length", 0) / 2
    result = []
    for obj in all_objects:
        if obj.get("id") == surface.get("id"):
            continue
        obj_z = obj.get("z", 0.0)
        if abs(obj_z - stop_z) > z_tolerance:
            continue
        ox, oy = obj["x"], obj["y"]
        ow, ol = obj.get("width", 0) / 2, obj.get("length", 0) / 2
        if (abs(ox - sx) < sw + ow) and (abs(oy - sy) < sl + ol):
            result.append(obj)
    return result


# ---------------------------------------------------------------------------
# Auxiliary data loading (normals + instance segmentation)
# ---------------------------------------------------------------------------


def load_aux_data(scene_dir: str, cam_name: str) -> dict | None:
    """
    Load normals and instance segmentation for *cam_name*.

    Returns dict with normals (H,W,3), seg (H,W), id_to_obj mapping,
    or None if data is missing.
    """
    normals_path = os.path.join(scene_dir, "normals", f"{cam_name}.npy")
    seg_path = os.path.join(scene_dir, "instance_seg", f"{cam_name}.npy")
    info_path = os.path.join(scene_dir, "instance_seg", f"{cam_name}_info.json")

    if not (
        os.path.isfile(normals_path)
        and os.path.isfile(seg_path)
        and os.path.isfile(info_path)
    ):
        return None

    normals = np.load(normals_path)
    seg = np.load(seg_path)
    with open(info_path) as f:
        info = json.load(f)
    return {
        "normals": normals,
        "seg": seg,
        "id_to_obj": info.get("idToObjectId", {}),
        "id_to_labels": info.get("idToLabels", {}),
    }


def _seg_ids_for_object(aux: dict, object_id: str) -> list[int]:
    ids = []
    for sid_str, oid in aux["id_to_obj"].items():
        if oid == object_id:
            try:
                ids.append(int(sid_str))
            except ValueError:
                pass
    return ids


# ---------------------------------------------------------------------------
# Empty-surface pixel detection
# ---------------------------------------------------------------------------


def find_empty_surface_pixels(
    aux: dict,
    surface_obj_id: str,
    *,
    normal_z_threshold: float = 0.9,
) -> np.ndarray:
    """Boolean mask (H,W) of upward-facing pixels belonging to the surface object."""
    seg_ids = _seg_ids_for_object(aux, surface_obj_id)
    if not seg_ids:
        return np.zeros(aux["seg"].shape[:2], dtype=bool)

    surface_mask = np.isin(aux["seg"], seg_ids)
    upward = aux["normals"][:, :, 2] > normal_z_threshold
    return surface_mask & upward


# ---------------------------------------------------------------------------
# World-to-pixel footprint conversion
# ---------------------------------------------------------------------------


def footprint_to_pixel_kernel(
    footprint: tuple[float, float],
    surface_center: tuple[float, float, float],
    camera_position: Tuple[float, float, float],
    camera_look_at: Tuple[float, float, float],
    image_width: int,
    image_height: int,
    *,
    horizontal_fov_deg: float = 82.0,
) -> tuple[int, int] | None:
    """Convert world-space footprint (w,l) to pixel-space kernel (h_px, w_px)."""
    aspect = image_width / image_height
    cx, cy, cz = surface_center
    uv0 = project_world_to_fraction(
        (cx, cy, cz),
        camera_position,
        camera_look_at,
        horizontal_fov_deg=horizontal_fov_deg,
        aspect_ratio=aspect,
    )
    if uv0 is None:
        return None

    hw, hl = footprint[0] / 2, footprint[1] / 2
    uv_right = project_world_to_fraction(
        (cx + hw, cy, cz),
        camera_position,
        camera_look_at,
        horizontal_fov_deg=horizontal_fov_deg,
        aspect_ratio=aspect,
    )
    uv_fwd = project_world_to_fraction(
        (cx, cy + hl, cz),
        camera_position,
        camera_look_at,
        horizontal_fov_deg=horizontal_fov_deg,
        aspect_ratio=aspect,
    )
    if uv_right is None or uv_fwd is None:
        return None

    w_px = max(1, int(round(abs(uv_right[0] - uv0[0]) * image_width * 2)))
    h_px = max(1, int(round(abs(uv_fwd[1] - uv0[1]) * image_height * 2)))
    return (h_px, w_px)


# ---------------------------------------------------------------------------
# Erosion-based valid placement detection
# ---------------------------------------------------------------------------


def find_valid_placement_centers(
    free_mask: np.ndarray,
    kernel_h: int,
    kernel_w: int,
) -> np.ndarray:
    """
    Boolean mask of pixels where an object with footprint (kernel_h, kernel_w)
    can be centered. Uses integral-image approach: O(H*W) regardless of kernel.
    """
    if kernel_h <= 1 and kernel_w <= 1:
        return free_mask.copy()

    H, W = free_mask.shape
    if kernel_h > H or kernel_w > W:
        return np.zeros((H, W), dtype=bool)

    pad_top = kernel_h // 2
    pad_left = kernel_w // 2

    padded = np.pad(
        free_mask.astype(np.int32),
        ((pad_top, kernel_h - 1 - pad_top), (pad_left, kernel_w - 1 - pad_left)),
        mode="constant",
        constant_values=0,
    )

    PH, PW = padded.shape
    integral = np.zeros((PH + 1, PW + 1), dtype=np.int64)
    integral[1:, 1:] = padded.cumsum(axis=0).cumsum(axis=1)

    total = kernel_h * kernel_w
    r = np.arange(H)
    c = np.arange(W)
    rr = r[:, None]
    cc = c[None, :]
    window_sums = (
        integral[rr + kernel_h, cc + kernel_w]
        - integral[rr, cc + kernel_w]
        - integral[rr + kernel_h, cc]
        + integral[rr, cc]
    )

    return window_sums >= total


# ---------------------------------------------------------------------------
# Placement feasibility (geometry-only check)
# ---------------------------------------------------------------------------


def check_placement_feasibility(
    surface: dict,
    target_xy: tuple[float, float],
    target_dims: tuple[float, float],
    occupied_objects: list[dict],
) -> bool:
    """Check whether footprint target_dims fits at target_xy on surface without collisions."""
    sx, sy = surface["x"], surface["y"]
    sw, sl = surface.get("width", 0) / 2, surface.get("length", 0) / 2
    tw, tl = target_dims[0] / 2, target_dims[1] / 2
    tx, ty = target_xy

    if (tx - tw < sx - sw) or (tx + tw > sx + sw):
        return False
    if (ty - tl < sy - sl) or (ty + tl > sy + sl):
        return False

    for obj in occupied_objects:
        ox, oy = obj["x"], obj["y"]
        ow, ol = obj.get("width", 0) / 2, obj.get("length", 0) / 2
        if (abs(tx - ox) < tw + ow) and (abs(ty - oy) < tl + ol):
            return False

    return True
