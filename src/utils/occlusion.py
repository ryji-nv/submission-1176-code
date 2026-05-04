"""
Object filtering: frustum test + multi-ray occlusion check.

All computations use object centroids: (x, y, z + height/2).
"""

from __future__ import annotations

import math
from typing import Any

from src.utils.projection import project_world_to_fraction

# Non-physical / non-solid object types excluded from all questions.
_NON_SOLID_TYPES = {"lighting"}


def object_centroid(obj: dict[str, Any]) -> tuple[float, float, float]:
    """Return the object's centroid: floor origin + half height."""
    return (obj["x"], obj["y"], obj["z"] + obj.get("height", 0.0) / 2.0)


def enrich_objects(
    objects: list[dict],
    camera_position: tuple[float, float, float],
) -> list[dict]:
    """Add ``centroid`` and ``distance`` fields expected by curators."""
    out = []
    for o in objects:
        c = object_centroid(o)
        d = math.sqrt(sum((a - b) ** 2 for a, b in zip(camera_position, c)))
        out.append({**o, "centroid": c, "distance": d})
    return out


def _is_solid(obj: dict[str, Any], min_thickness: float) -> bool:
    """Return False for non-physical types or objects too thin in any dimension."""
    if obj.get("type", "").lower() in _NON_SOLID_TYPES:
        return False
    w = obj.get("width") or 0.0
    l = obj.get("length") or 0.0
    h = obj.get("height") or 0.0
    if not (w > 0 and l > 0 and h > 0):
        return False
    if min(w, l, h) < min_thickness:
        return False
    return True


# ---------------------------------------------------------------------------
# Ray-AABB intersection (slab method)
# ---------------------------------------------------------------------------


def _ray_aabb_intersect(
    origin: tuple[float, float, float],
    direction: tuple[float, float, float],
    aabb_min: tuple[float, float, float],
    aabb_max: tuple[float, float, float],
) -> float | None:
    """Return t of first positive hit, or None. Uses the slab method."""
    tmin = -math.inf
    tmax = math.inf
    for i in range(3):
        d = direction[i]
        if abs(d) < 1e-12:
            if origin[i] < aabb_min[i] or origin[i] > aabb_max[i]:
                return None
        else:
            t1 = (aabb_min[i] - origin[i]) / d
            t2 = (aabb_max[i] - origin[i]) / d
            tmin = max(tmin, min(t1, t2))
            tmax = min(tmax, max(t1, t2))
    if tmax < max(tmin, 0.0):
        return None
    return tmin if tmin >= 0.0 else tmax


def _bbox_sample_points(
    centroid: tuple[float, float, float],
    w: float,
    l: float,
    h: float,
) -> list[tuple[float, float, float]]:
    """Return centroid + all 8 bounding-box corners."""
    cx, cy, cz = centroid
    hw, hl, hh = w / 2, l / 2, h / 2
    points = [centroid]
    for sx in (-1, 1):
        for sy in (-1, 1):
            for sz in (-1, 1):
                points.append((cx + sx * hw, cy + sy * hl, cz + sz * hh))
    return points


def _visible_fraction(
    camera_pos: tuple[float, float, float],
    candidate: dict,
    all_objects: list[dict],
    shrink: float,
) -> tuple[float, bool]:
    """
    Returns (fraction, centroid_visible).

    fraction:         share of sample points (centroid + 8 corners) unoccluded.
    centroid_visible:  True iff the centroid ray itself is unoccluded.
    """
    cx, cy, cz = candidate["centroid"]
    w = candidate.get("width") or 0.0
    l = candidate.get("length") or 0.0
    h = candidate.get("height") or 0.0
    sample_points = _bbox_sample_points((cx, cy, cz), w, l, h)

    occluders = []
    for obj in all_objects:
        if obj.get("id") == candidate.get("id"):
            continue
        oc = object_centroid(obj)
        hw = (obj.get("width") or 0.0) / 2 * shrink
        hl = (obj.get("length") or 0.0) / 2 * shrink
        hh = (obj.get("height") or 0.0) / 2 * shrink
        if hw < 1e-6 and hl < 1e-6 and hh < 1e-6:
            continue
        occluders.append(
            (
                (oc[0] - hw, oc[1] - hl, oc[2] - hh),
                (oc[0] + hw, oc[1] + hl, oc[2] + hh),
            )
        )

    unoccluded = 0
    centroid_visible = False
    for idx, pt in enumerate(sample_points):
        dx = pt[0] - camera_pos[0]
        dy = pt[1] - camera_pos[1]
        dz = pt[2] - camera_pos[2]
        t_target = math.sqrt(dx * dx + dy * dy + dz * dz)
        if t_target < 1e-9:
            unoccluded += 1
            if idx == 0:
                centroid_visible = True
            continue
        ray_dir = (dx / t_target, dy / t_target, dz / t_target)
        blocked = False
        for aabb_min, aabb_max in occluders:
            t_hit = _ray_aabb_intersect(camera_pos, ray_dir, aabb_min, aabb_max)
            if t_hit is not None and t_hit < t_target - 1e-3:
                blocked = True
                break
        if not blocked:
            unoccluded += 1
            if idx == 0:
                centroid_visible = True

    return unoccluded / len(sample_points), centroid_visible


# ---------------------------------------------------------------------------
# Main filter
# ---------------------------------------------------------------------------


def filter_visible_objects(
    camera_pose: dict,
    all_objects: list[dict],
    *,
    min_dist: float = 0.3,
    max_dist: float = 15.0,
    min_thickness: float = 0.08,
    min_visible_fraction: float = 0.5,
    shrink: float = 0.85,
    require_centroid_visible: bool = True,
) -> list[dict]:
    """
    Return objects that pass all visibility filters:

    1. Solid-object filter  — exclude non-physical types and objects thinner than
                              min_thickness in any bounding-box dimension.
    2. Distance filter      — centroid must be between min_dist and max_dist metres.
    3. Frustum filter       — centroid must project within the image frame [0, 1]².
    4. Occlusion filter     — at least min_visible_fraction of the object's bounding-box
                              sample points (centroid + 8 corners) must be unobstructed.
    5. Centroid-ray filter  — the centroid itself must be unoccluded (when
                              require_centroid_visible is True).  This prevents selecting
                              objects whose annotation point is hidden behind another
                              object even though enough bbox corners are visible.

    Each returned dict has "distance" and "centroid" keys added.
    """
    pos = tuple(camera_pose["position"])
    look_at = tuple(camera_pose["look_at"])
    fov = camera_pose.get("horizontal_fov_deg", 82.0)

    # Pass 1: solid + distance + frustum
    candidates = []
    for obj in all_objects:
        if not _is_solid(obj, min_thickness):
            continue
        centroid = object_centroid(obj)
        dx = centroid[0] - pos[0]
        dy = centroid[1] - pos[1]
        dz = centroid[2] - pos[2]
        dist = math.sqrt(dx * dx + dy * dy + dz * dz)
        if dist < min_dist or dist > max_dist:
            continue
        uv = project_world_to_fraction(centroid, pos, look_at, horizontal_fov_deg=fov)
        if uv is None:
            continue
        u, v = uv
        if 0.0 <= u <= 1.0 and 0.0 <= v <= 1.0:
            candidates.append({**obj, "distance": dist, "centroid": centroid})

    # Pass 2: occlusion — multi-ray against all original objects
    result = []
    for cand in candidates:
        frac, centroid_vis = _visible_fraction(pos, cand, all_objects, shrink)
        if frac < min_visible_fraction:
            continue
        if require_centroid_visible and not centroid_vis:
            continue
        result.append(cand)
    return result
