"""
Physics-understanding questions from scene + camera.

See ALL_QUESTION_TYPES for the list of supported question types.
See spatial_reasoning_capabilities.md for descriptions and benchmark coverage.
"""

from __future__ import annotations

import math
import random
import re
from typing import Any

from src.utils.constants import (
    ALL_QUESTION_TYPES,
    DOT_ANNOTATED_TYPES,
    MAX_TRIALS as _MAX_TRIALS,
    MIN_DISTANCE_GAP,
    NAME_ONLY_TYPES,
)
from src.utils.occlusion import filter_visible_objects
from src.utils.projection import project_world_to_fraction


_NUMERIC_SUFFIX_RE = re.compile(r"_?\d+$")

_csr_good_types_cache: dict[int, set[str]] = {}


def _get_csr_good_types(all_objects: list[dict]) -> set[str]:
    """Get VLM-filtered object types for compound_spatial_referring (cached per scene)."""
    cache_key = id(all_objects)
    if cache_key in _csr_good_types_cache:
        return _csr_good_types_cache[cache_key]
    obj_types = sorted({_normalize_type(o.get("type", "")) for o in all_objects
                        if _normalize_type(o.get("type", ""))})
    good: set[str] = set(obj_types)
    if obj_types:
        import os
        api_key = os.environ.get("VLM_API_KEY", "")
        if api_key:
            try:
                import requests
                prompt = (
                    "From the list below, select only discrete, movable, clearly "
                    "identifiable objects (like furniture, appliances, fixtures). "
                    "Exclude architectural elements (doors, walls, floors, outlets, "
                    "doorways, moldings) and large surfaces (countertops, islands, rugs). "
                    "Reply with ONLY the selected names, comma-separated.\n\n"
                    + ", ".join(obj_types)
                )
                resp = requests.post(
                    os.environ.get("VLM_API_BASE", "https://example.com/v1/chat/completions"),
                    headers={"Authorization": f"Bearer {api_key}",
                             "Content-Type": "application/json"},
                    json={"model": os.environ.get("VLM_MODEL", "Qwen3-VL-235B-A22B-Thinking"),
                          "messages": [{"role": "user", "content": prompt}],
                          "temperature": 0, "max_completion_tokens": 256},
                    timeout=15,
                )
                text = resp.json()["choices"][0]["message"]["content"]
                good = {t.strip().lower() for t in text.split(",")}
            except Exception:
                pass
    _csr_good_types_cache[cache_key] = good
    return good
_ROOM_PREFIX_RE = re.compile(
    r"^(?:bedroom|kitchen|living_room|bathroom|hallway|"
    r"dining_room|office|laundry|garage|closet|pantry|study)_"
)


def _normalize_type(t: str) -> str:
    """Strip trailing numeric suffix from object type (e.g. 'chair_5', 'candle2' → 'chair', 'candle')."""
    return _NUMERIC_SUFFIX_RE.sub("", t)


def _clean_label(t: str) -> str:
    """Produce a natural-language label: strip numeric suffix, room prefix, underscores."""
    t = _NUMERIC_SUFFIX_RE.sub("", t)
    t = _ROOM_PREFIX_RE.sub("", t)
    return t.replace("_", " ")


def object_label(obj: dict[str, Any]) -> str:
    """Short natural-language label for an object. Prefer type (cleaned), else id."""
    if obj.get("type"):
        return _clean_label(str(obj["type"]))
    if obj.get("id"):
        return _clean_label(str(obj["id"]))
    return f"object_{obj.get('object_index', '?')}"


def _object_meta(obj: dict, color: str) -> dict:
    """Shared object metadata dict used in all two-object question types."""
    return {
        "circle_color": color,
        "label": object_label(obj),
        "id": obj.get("id"),
        "centroid": list(obj["centroid"]),
        "distance_to_camera": obj["distance"],
        "width": obj.get("width"),
        "length": obj.get("length"),
        "height": obj.get("height"),
    }


def _to_letter_mcq(answer: str, choices: list[str]) -> tuple[str, list[str]]:
    """Convert word answer + choices to letter MCQ format (A/B/C/D)."""
    idx = choices.index(answer) if answer in choices else 0
    letter = chr(65 + idx)
    formatted = [f"{chr(65 + i)}. {c}" for i, c in enumerate(choices)]
    return letter, formatted


# ---------------------------------------------------------------------------
# Shared helpers used by multiple question types
# ---------------------------------------------------------------------------


def _compute_depth(centroid: tuple, cam_pos: tuple, look_at: tuple) -> float:
    """Depth of centroid along the camera's forward axis (metres)."""
    fx = look_at[0] - cam_pos[0]
    fy = look_at[1] - cam_pos[1]
    fz = look_at[2] - cam_pos[2]
    flen = math.sqrt(fx * fx + fy * fy + fz * fz)
    if flen < 1e-9:
        return 0.0
    fx, fy, fz = fx / flen, fy / flen, fz / flen
    return (
        (centroid[0] - cam_pos[0]) * fx
        + (centroid[1] - cam_pos[1]) * fy
        + (centroid[2] - cam_pos[2]) * fz
    )


_8DIR_NAMES = [
    "front",
    "front-right",
    "right",
    "rear-right",
    "rear",
    "rear-left",
    "left",
    "front-left",
]


def _quantize_8dir(dot_fwd: float, dot_right: float) -> str:
    """Quantize to the nearest of 8 compass directions. Always returns a value."""
    angle = math.degrees(math.atan2(dot_right, dot_fwd))
    idx = round(angle / 45.0) % 8
    return _8DIR_NAMES[idx]


# ── Camera frame helpers (Z-up QA space) ──────────────────────────────────
# All functions use the image-plane right-vector decomposition so results
# are purely relative to camera A's frame and independent of world axes.


def _cam_forward_right_2d(
    cam: dict,
) -> tuple[tuple[float, float], tuple[float, float]] | None:
    """Normalized horizontal forward and right vectors for a camera.

    Returns ((fx, fy), (rx, ry)) or None if the forward vector is degenerate.
    Right = cross(forward, Z_up) projected to 2D, i.e. the direction to the
    right of the camera image.
    """
    fx = cam["look_at"][0] - cam["position"][0]
    fy = cam["look_at"][1] - cam["position"][1]
    flen = math.hypot(fx, fy)
    if flen < 1e-9:
        return None
    fx /= flen
    fy /= flen
    return (fx, fy), (fy, -fx)


def _relative_heading_deg(cam_a: dict, cam_b: dict) -> float | None:
    """Relative heading of camera B in camera A's reference frame (degrees).

    Projects both forward vectors onto the horizontal plane (XY, Z-up),
    then decomposes B's direction using A's forward and right vectors.
    Returns angle where +90 = East (B faces to A's right in the image),
    -90 = West, 0 = North (same direction), +/-180 = South.
    Returns None if either camera has a degenerate forward vector.
    """
    frame_a = _cam_forward_right_2d(cam_a)
    frame_b = _cam_forward_right_2d(cam_b)
    if frame_a is None or frame_b is None:
        return None
    (fax, fay), (rax, ray) = frame_a
    (fbx, fby), _ = frame_b
    dot_fwd = fbx * fax + fby * fay
    dot_right = fbx * rax + fby * ray
    return math.degrees(math.atan2(dot_right, dot_fwd))


def _relative_position_in_frame(
    cam_a: dict, cam_b: dict
) -> tuple[float, float, float] | None:
    """Position of camera B in camera A's local frame.

    Returns (forward, right, up) distances in meters, or None if degenerate.
    Positive forward = B is in front of A, positive right = B is to A's right.
    """
    frame = _cam_forward_right_2d(cam_a)
    if frame is None:
        return None
    (fx, fy), (rx, ry) = frame
    dx = cam_b["position"][0] - cam_a["position"][0]
    dy = cam_b["position"][1] - cam_a["position"][1]
    dz = cam_b["position"][2] - cam_a["position"][2]
    return (dx * fx + dy * fy, dx * rx + dy * ry, dz)


def _relative_transform_5dof(
    cam_a: dict, cam_b: dict
) -> dict[str, float] | None:
    """5-DOF relative transformation from cam_a to cam_b in A's local frame.

    Returns dict with keys: forward, right, up (translation in meters),
    yaw (degrees, +right), pitch (degrees, +up).  None if degenerate.
    """
    pos = _relative_position_in_frame(cam_a, cam_b)
    if pos is None:
        return None
    t_fwd, t_right, t_up = pos

    heading = _relative_heading_deg(cam_a, cam_b)
    if heading is None:
        return None

    # Pitch: vertical tilt difference
    fax = cam_a["look_at"][0] - cam_a["position"][0]
    fay = cam_a["look_at"][1] - cam_a["position"][1]
    faz = cam_a["look_at"][2] - cam_a["position"][2]
    fbx = cam_b["look_at"][0] - cam_b["position"][0]
    fby = cam_b["look_at"][1] - cam_b["position"][1]
    fbz = cam_b["look_at"][2] - cam_b["position"][2]
    hlen_a = math.hypot(fax, fay)
    hlen_b = math.hypot(fbx, fby)
    pitch_a = math.degrees(math.atan2(faz, hlen_a)) if hlen_a > 1e-9 else 0.0
    pitch_b = math.degrees(math.atan2(fbz, hlen_b)) if hlen_b > 1e-9 else 0.0

    return {
        "forward": t_fwd,
        "right": t_right,
        "up": t_up,
        "yaw": heading,
        "pitch": pitch_b - pitch_a,
    }


def _build_imagined_pose(
    move_to: dict,
    face_toward: dict,
    cam_height: float,
) -> tuple[tuple, tuple, tuple] | None:
    """
    Construct a camera pose at move_to's centroid (at cam_height) facing
    face_toward's centroid.  Returns (position, forward, right) unit vectors,
    or None if the vertical tilt > 60 degrees.
    """
    cx, cy, cz = move_to["centroid"]
    pos = (cx, cy, cam_height)
    tx, ty, tz = face_toward["centroid"]
    fx, fy, fz = tx - pos[0], ty - pos[1], tz - pos[2]
    flen = math.sqrt(fx * fx + fy * fy + fz * fz)
    if flen < 1e-6:
        return None
    fx, fy, fz = fx / flen, fy / flen, fz / flen
    if abs(fz) > 0.866:
        return None
    up0 = (0.0, 0.0, 1.0)
    rx = fy * up0[2] - fz * up0[1]
    ry = fz * up0[0] - fx * up0[2]
    rz = fx * up0[1] - fy * up0[0]
    rlen = math.sqrt(rx * rx + ry * ry + rz * rz)
    if rlen < 1e-6:
        return None
    rx, ry, rz = rx / rlen, ry / rlen, rz / rlen
    ux = ry * fz - rz * fy
    uy = rz * fx - rx * fz
    uz = rx * fy - ry * fx
    return pos, (fx, fy, fz), (rx, ry, rz), (ux, uy, uz)


def _relation_oc(
    pose: tuple,
    obj: dict,
    threshold: float = 0.3,
) -> tuple[str, str, str] | None:
    """
    Compute (left/right, above/below, front/behind) of obj relative to
    the imagined observer.  Returns None if any axis is within threshold.
    """
    pos, fwd, right, up = pose
    dx = obj["centroid"][0] - pos[0]
    dy = obj["centroid"][1] - pos[1]
    dz = obj["centroid"][2] - pos[2]
    dot_r = dx * right[0] + dy * right[1] + dz * right[2]
    dot_f = dx * fwd[0] + dy * fwd[1] + dz * fwd[2]
    lr = "right" if dot_r > threshold else ("left" if dot_r < -threshold else "")
    ud = "above" if dz > threshold else ("below" if dz < -threshold else "")
    fb = "front" if dot_f > threshold else ("behind" if dot_f < -threshold else "")
    if not lr or not ud or not fb:
        return None
    return (lr, ud, fb)


_AREA_MAP = {
    "bed": ("sleeping area", "the bed"),
    "bunk_bed": ("sleeping area", "the bunk bed"),
    "desk": ("working area", "the desk"),
    "writing_desk": ("working area", "the writing desk"),
    "office_chair": ("working area", "the office chair"),
    "sofa": ("seating area", "the sofa"),
    "couch": ("seating area", "the couch"),
    "armchair": ("seating area", "the armchair"),
    "chair": ("seating area", "the chair"),
    "seat": ("seating area", "the seat"),
    "bench": ("seating area", "the bench"),
    "recliner": ("seating area", "the recliner"),
    "dining_table": ("dining area", "the dining table"),
    "table": ("dining area", "the table"),
    "kitchen_counter": ("cooking area", "the kitchen counter"),
    "stove": ("cooking area", "the stove"),
    "oven": ("cooking area", "the oven"),
    "wardrobe": ("storage area", "the wardrobe"),
    "dresser": ("storage area", "the dresser"),
    "bookcase": ("storage area", "the bookcase"),
    "bookshelf": ("storage area", "the bookshelf"),
    "shelf": ("storage area", "the shelf"),
    "sideboard": ("storage area", "the sideboard"),
    "cabinet": ("storage area", "the cabinet"),
    "tv": ("entertainment area", "the TV"),
    "monitor": ("entertainment area", "the monitor"),
    "nightstand": ("bedside area", "the nightstand"),
    "bedside_table": ("bedside area", "the bedside table"),
}


def _strip_room_prefix(raw: str) -> str:
    parts = raw.split("_", 1)
    if len(parts) > 1 and parts[0] in (
        "bedroom",
        "bathroom",
        "hallway",
        "closet",
        "garage",
        "laundry",
    ):
        return parts[1]
    if raw.startswith(("living_room_", "dining_room_")):
        return raw.split("_", 2)[2] if raw.count("_") >= 2 else raw
    return raw


def _identify_areas(
    visible_objects: list[dict],
    seg_mask_dir: str | None = None,
    cam_name: str | None = None,
    instance_id_map: dict | None = None,
    min_anchor_px: int = 10000,
) -> dict[str, dict]:
    """Map area names to {centroid, anchor_desc, anchor_ids} from visible objects.

    When seg_mask_dir/instance_id_map are provided, only keeps areas whose
    anchor objects have at least min_anchor_px pixels in the camera's seg mask
    (ensures the area is clearly identifiable in the image).
    """
    area_data: dict[str, dict] = {}
    for obj in visible_objects:
        raw = _normalize_type(obj.get("type", "")).lower()
        label = _strip_room_prefix(raw).replace(" ", "_")
        for keyword, (area_name, anchor_desc) in _AREA_MAP.items():
            if keyword in label:
                if area_name not in area_data:
                    area_data[area_name] = {
                        "points": [],
                        "anchor": anchor_desc,
                        "anchor_ids": [],
                    }
                _c = obj.get("centroid")
                _pt = (_c[0], _c[1]) if _c else (obj.get("x", 0), obj.get("y", 0))
                area_data[area_name]["points"].append(_pt)
                area_data[area_name]["anchor_ids"].append(obj.get("id"))
                break

    # Visibility check: anchor objects must be clearly visible in frame
    import os as _os
    import numpy as _np
    anchor_px: dict[str, int] = {}
    if seg_mask_dir and cam_name and instance_id_map:
        _seg_path = _os.path.join(seg_mask_dir, f"{cam_name}_seg.npy")
        if _os.path.isfile(_seg_path):
            _mask = _np.load(_seg_path)
            for name, data in area_data.items():
                best = 0
                for aid in data["anchor_ids"]:
                    iid = instance_id_map.get(aid, -1)
                    if iid >= 0:
                        best = max(best, int((_mask == iid).sum()))
                anchor_px[name] = best

    result = {}
    for name, data in area_data.items():
        pts = data["points"]
        if len(pts) >= 2:
            cx = sum(p[0] for p in pts) / len(pts)
            cy = sum(p[1] for p in pts) / len(pts)
            spread = max(math.sqrt((p[0] - cx) ** 2 + (p[1] - cy) ** 2) for p in pts)
            if spread > 3.0:
                continue
        if anchor_px and anchor_px.get(name, 0) < min_anchor_px:
            continue
        cx = sum(p[0] for p in pts) / len(pts)
        cy = sum(p[1] for p in pts) / len(pts)
        result[name] = {
            "centroid": (cx, cy),
            "anchor": data["anchor"],
            "anchor_ids": data["anchor_ids"],
        }
    return result


_CARDINAL_8DIR = [
    "North",
    "Northeast",
    "East",
    "Southeast",
    "South",
    "Southwest",
    "West",
    "Northwest",
]


def _quantize_cardinal_8dir(
    dot_fwd: float, dot_right: float, tolerance: float = 22.0
) -> str | None:
    """Quantize to one of 8 cardinal directions, or None if outside tolerance."""
    angle = math.degrees(math.atan2(dot_right, dot_fwd))
    idx = round(angle / 45.0) % 8
    nearest = idx * 45
    diff = (angle - nearest + 180) % 360 - 180
    if abs(diff) > tolerance:
        return None
    return _CARDINAL_8DIR[idx]


def _project_bbox_2d(
    obj: dict,
    cam_pos: tuple,
    look_at: tuple,
    fov: float,
) -> tuple[int, int, int, int] | None:
    """Project object's 3D extent to a 2D bbox in 0-1000 image space."""
    cx, cy, cz = obj["centroid"]
    hw = (obj.get("width") or 0.3) / 2
    hl = (obj.get("length") or 0.3) / 2
    hh = (obj.get("height") or 0.3) / 2
    corners = [
        (cx + sx * hw, cy + sy * hl, cz + sz * hh)
        for sx in (-1, 1)
        for sy in (-1, 1)
        for sz in (-1, 1)
    ]
    us, vs = [], []
    for pt in corners:
        uv = project_world_to_fraction(pt, cam_pos, look_at, horizontal_fov_deg=fov)
        if uv is None:
            return None
        us.append(uv[0])
        vs.append(uv[1])
    x_min = max(0, int(min(us) * 1000))
    y_min = max(0, int(min(vs) * 1000))
    x_max = min(1000, int(max(us) * 1000))
    y_max = min(1000, int(max(vs) * 1000))
    if x_max - x_min < 10 or y_max - y_min < 10:
        return None
    return (x_min, y_min, x_max, y_max)


def _longest_dim(obj: dict) -> float:
    return max(obj.get("width") or 0, obj.get("length") or 0, obj.get("height") or 0)


# ---------------------------------------------------------------------------
# Per-type curate function imports (dynamic — module names start with digits)
# ---------------------------------------------------------------------------

import importlib as _importlib


def _import_curate(module_name: str, func_name: str):
    return getattr(_importlib.import_module(f"src.tasks.{module_name}"), func_name)


# Distance & Depth
curate_closest_object_questions = _import_curate(
    "distance_and_depth.000_closest_object", "curate_closest_object_questions"
)
curate_depth_estimation_questions = _import_curate(
    "distance_and_depth.001_depth_estimation", "curate_depth_estimation_questions"
)
curate_distance_estimation_questions = _import_curate(
    "distance_and_depth.002_distance_estimation", "curate_distance_estimation_questions"
)
curate_object_distance_questions = _import_curate(
    "distance_and_depth.003_object_distance", "curate_object_distance_questions"
)
curate_depth_ordering_questions = _import_curate(
    "distance_and_depth.006_depth_ordering", "curate_depth_ordering_questions"
)
curate_depth_difference_questions = _import_curate(
    "distance_and_depth.020_depth_difference", "curate_depth_difference_questions"
)

# Direction & Position
curate_relative_direction_questions = _import_curate(
    "direction_and_position.004_relative_direction",
    "curate_relative_direction_questions",
)
curate_camera_object_position_questions = _import_curate(
    "direction_and_position.005_camera_object_position",
    "curate_camera_object_position_questions",
)
curate_camera_region_position_questions = _import_curate(
    "direction_and_position.027_camera_region_position",
    "curate_camera_region_position_questions",
)
curate_object_region_position_questions = _import_curate(
    "direction_and_position.028_object_region_position",
    "curate_object_region_position_questions",
)
curate_region_region_position_questions = _import_curate(
    "direction_and_position.029_region_region_position",
    "curate_region_region_position_questions",
)

# Size & Count
curate_object_size_questions = _import_curate(
    "size_and_count.007_object_size", "curate_object_size_questions"
)
curate_object_count_questions = _import_curate(
    "size_and_count.008_object_count", "curate_object_count_questions"
)
curate_room_size_questions = _import_curate(
    "size_and_count.009_room_size", "curate_room_size_questions"
)
curate_object_category_questions = _import_curate(
    "size_and_count.022_object_category", "curate_object_category_questions"
)
curate_object_size_qualitative_questions = _import_curate(
    "size_and_count.023_object_size_qualitative",
    "curate_object_size_qualitative_questions",
)

# Grounding & Placement
curate_object_grounding_questions = _import_curate(
    "grounding_and_placement.010_object_grounding", "curate_object_grounding_questions"
)
curate_object_placement_questions = _import_curate(
    "grounding_and_placement.017_object_placement", "curate_object_placement_questions"
)
curate_free_space_questions = _import_curate(
    "grounding_and_placement.031_free_space", "curate_free_space_questions"
)
curate_object_matching_mv_questions = _import_curate(
    "grounding_and_placement.019_object_matching_mv",
    "curate_object_matching_mv_questions",
)
curate_object_grounding_bbox_questions = _import_curate(
    "grounding_and_placement.021_object_grounding_bbox",
    "curate_object_grounding_bbox_questions",
)
curate_comparative_spatial_grounding_questions = _import_curate(
    "grounding_and_placement.024_comparative_spatial_grounding",
    "curate_comparative_spatial_grounding_questions",
)
curate_ordinal_grounding_questions = _import_curate(
    "grounding_and_placement.025_ordinal_grounding",
    "curate_ordinal_grounding_questions",
)
curate_object_object_position_mv_questions = _import_curate(
    "grounding_and_placement.026_object_object_position_mv",
    "curate_object_object_position_mv_questions",
)

# Camera & Viewpoint
curate_camera_relative_position_questions = _import_curate(
    "camera_and_viewpoint.011_camera_relative_position",
    "curate_camera_relative_position_questions",
)
curate_camera_facing_direction_questions = _import_curate(
    "camera_and_viewpoint.012_camera_facing_direction",
    "curate_camera_facing_direction_questions",
)
curate_camera_motion_questions = _import_curate(
    "camera_and_viewpoint.013_camera_motion", "curate_camera_motion_questions"
)
curate_viewpoint_change_questions = _import_curate(
    "camera_and_viewpoint.014_viewpoint_change", "curate_viewpoint_change_questions"
)

# Complex Reasoning
curate_spatial_imagination_questions = _import_curate(
    "complex_reasoning.015_spatial_imagination", "curate_spatial_imagination_questions"
)
curate_compound_spatial_referring_questions = _import_curate(
    "complex_reasoning.016_compound_spatial_referring",
    "curate_compound_spatial_referring_questions",
)
curate_spatial_compatibility_questions = _import_curate(
    "complex_reasoning.018_spatial_compatibility",
    "curate_spatial_compatibility_questions",
)
curate_route_planning_questions = _import_curate(
    "complex_reasoning.030_route_planning", "curate_route_planning_questions"
)


# ---------------------------------------------------------------------------
# Top-level entry point
# ---------------------------------------------------------------------------


def curate_questions(
    camera_pose: dict,
    all_objects: list[dict],
    question_type: str,
    *,
    max_questions: int = 1,
    min_distance_gap: float = MIN_DISTANCE_GAP,
    seed: int = 42,
    visibility_kwargs: dict | None = None,
    all_cameras: list[dict] | None = None,
    room_bounds: list[float] | None = None,
    aux_data_dir: str | None = None,
    pre_visible: list[dict] | None = None,
    cam_visibility: dict | None = None,
    scene_type: str = "sage",
    seg_mask_dir: str | None = None,
    instance_id_map: dict | None = None,
    mask_vis_cache: dict[str, set[str]] | None = None,
) -> list[dict]:
    """
    Run visibility filtering then dispatch to the appropriate curator.

    pre_visible: if provided, skip geometric filtering and use these objects
                 directly (e.g. from precomputed seg-mask visibility).
    cam_visibility: cam_name -> [obj_id, …] map; forwarded to multi-view
                    curators so they can resolve per-camera visibility without
                    re-scanning masks.
    """
    rng = random.Random(seed)
    if pre_visible is not None:
        visible = pre_visible
    else:
        vis_kw = dict(visibility_kwargs or {})
        if question_type in DOT_ANNOTATED_TYPES:
            vis_kw.setdefault("min_visible_fraction", 0.9)
        elif question_type == "object_size":
            vis_kw.setdefault("min_visible_fraction", 0.6)
        visible = filter_visible_objects(camera_pose, all_objects, **vis_kw)

    if question_type in DOT_ANNOTATED_TYPES:
        visible = [
            o
            for o in visible
            if max(o.get("width", 0), o.get("length", 0), o.get("height", 0)) >= 0.2
        ]

    # Filter by visual prominence (mask pixel area)
    _SKIP_MASK_FILTER = {"compound_spatial_referring", "route_planning", "spatial_compatibility"}
    if question_type not in _SKIP_MASK_FILTER:
        cam_name = camera_pose.get("name", "")
        if mask_vis_cache and cam_name in mask_vis_cache:
            _allowed = mask_vis_cache[cam_name]
            visible = [o for o in visible if o.get("id") in _allowed]
        elif seg_mask_dir and instance_id_map:
            import os as _os
            import numpy as _np
            _min_mask_px = 5000
            _seg_path = _os.path.join(seg_mask_dir, f"{cam_name}_seg.npy")
            if _os.path.isfile(_seg_path):
                _mask = _np.load(_seg_path)
                visible = [
                    o for o in visible
                    if int((_mask == instance_id_map.get(o.get("id"), -1)).sum()) >= _min_mask_px
                ]

    _ANNOTATED_NAME_TYPES = {"object_size", "object_grounding_bbox", "object_size_qualitative"}
    if question_type in NAME_ONLY_TYPES:
        if question_type in _ANNOTATED_NAME_TYPES:
            seen_labels: set[str] = set()
            deduped = []
            for o in visible:
                lbl = object_label(o)
                if lbl not in seen_labels:
                    seen_labels.add(lbl)
                    deduped.append(o)
            visible = deduped
        else:
            # Count visible instances per label using seg mask (GPU)
            # Keep only objects whose label has exactly 1 visible instance
            if seg_mask_dir and instance_id_map:
                import os as _os
                import numpy as _np
                _seg_path = _os.path.join(seg_mask_dir, f"{camera_pose['name']}_seg.npy")
                if _os.path.isfile(_seg_path):
                    try:
                        import torch as _torch
                        _mask_t = _torch.from_numpy(_np.load(_seg_path)).to(
                            device="cuda", dtype=_torch.int32)
                        _vis_min_px = 1000
                        _label_vis_count: dict[str, int] = {}
                        _label_vis_obj: dict[str, list] = {}
                        for o in visible:
                            lbl = object_label(o)
                            iid = instance_id_map.get(o.get("id"), -1)
                            px = int((_mask_t == iid).sum().item())
                            if px >= _vis_min_px:
                                _label_vis_count[lbl] = _label_vis_count.get(lbl, 0) + 1
                                _label_vis_obj.setdefault(lbl, []).append(o)
                        visible = [
                            _label_vis_obj[lbl][0]
                            for lbl, cnt in _label_vis_count.items()
                            if cnt == 1
                        ]
                        del _mask_t
                    except Exception:
                        label_counts: dict[str, int] = {}
                        for o in visible:
                            lbl = object_label(o)
                            label_counts[lbl] = label_counts.get(lbl, 0) + 1
                        visible = [o for o in visible if label_counts.get(object_label(o), 0) == 1]
                else:
                    label_counts = {}
                    for o in visible:
                        lbl = object_label(o)
                        label_counts[lbl] = label_counts.get(lbl, 0) + 1
                    visible = [o for o in visible if label_counts.get(object_label(o), 0) == 1]
            else:
                label_counts = {}
                for o in all_objects:
                    lbl = object_label(o)
                    label_counts[lbl] = label_counts.get(lbl, 0) + 1
                visible = [o for o in visible if label_counts.get(object_label(o), 0) == 1]
                visible = [
                    o for o in visible
                    if max(o.get("width", 0), o.get("length", 0), o.get("height", 0)) >= 0.3
                ]

    if question_type == "closest_object":
        result = curate_closest_object_questions(
            camera_pose,
            visible,
            max_questions=max_questions,
            min_distance_gap=min_distance_gap,
            rng=rng,
        )
    elif question_type == "depth_estimation":
        result = curate_depth_estimation_questions(
            camera_pose,
            visible,
            max_questions=max_questions,
            rng=rng,
        )
    elif question_type == "distance_estimation":
        result = curate_distance_estimation_questions(
            camera_pose,
            visible,
            max_questions=max_questions,
            rng=rng,
        )
    elif question_type == "object_distance":
        result = curate_object_distance_questions(
            camera_pose,
            visible,
            max_questions=max_questions,
            rng=rng,
        )
    elif question_type == "relative_direction":
        result = curate_relative_direction_questions(
            camera_pose,
            visible,
            max_questions=max_questions,
            rng=rng,
        )
    elif question_type == "camera_object_position":
        result = curate_camera_object_position_questions(
            camera_pose,
            visible,
            max_questions=max_questions,
            rng=rng,
        )
    elif question_type == "depth_ordering":
        result = curate_depth_ordering_questions(
            camera_pose,
            visible,
            max_questions=max_questions,
            rng=rng,
        )
    elif question_type == "object_size":
        result = curate_object_size_questions(
            camera_pose,
            visible,
            max_questions=max_questions,
            rng=rng,
        )
    elif question_type == "object_count":
        if scene_type == "marble":
            return []
        _count_kw = {}
        result = curate_object_count_questions(
            camera_pose,
            visible,
            max_questions=max_questions,
            rng=rng,
            **_count_kw,
        )
    elif question_type == "room_size":
        if scene_type == "marble":
            return []
        if room_bounds is None:
            raise ValueError("room_bounds is required for room_size")
        result = curate_room_size_questions(camera_pose, room_bounds)
    elif question_type == "object_grounding":
        result = curate_object_grounding_questions(
            camera_pose,
            visible,
            max_questions=max_questions,
            rng=rng,
        )
    elif question_type == "camera_relative_position":
        if not all_cameras:
            raise ValueError("all_cameras is required for camera_relative_position")
        _cam_kw = {}
        if scene_type == "marble":
            _cam_kw["max_separation"] = 1.5
        result = curate_camera_relative_position_questions(
            all_cameras,
            max_questions=max_questions,
            rng=rng,
            **_cam_kw,
        )
    elif question_type == "camera_facing_direction":
        if not all_cameras:
            raise ValueError("all_cameras is required for camera_facing_direction")
        _cam_kw = {}
        if scene_type == "marble":
            _cam_kw["max_separation"] = 1.5
            _cam_kw["seg_mask_dir"] = seg_mask_dir
            _cam_kw["instance_id_map"] = instance_id_map
        result = curate_camera_facing_direction_questions(
            all_cameras,
            max_questions=max_questions,
            rng=rng,
            **_cam_kw,
        )
    elif question_type == "camera_motion":
        if not all_cameras:
            raise ValueError("all_cameras is required for camera_motion")
        result = curate_camera_motion_questions(
            camera_pose,
            all_cameras,
            max_questions=max_questions,
            rng=rng,
        )
    elif question_type == "viewpoint_change":
        if not all_cameras:
            raise ValueError("all_cameras is required for viewpoint_change")
        _cam_kw = {}
        if scene_type == "marble":
            _cam_kw["max_separation"] = 1.5
            _cam_kw["seg_mask_dir"] = seg_mask_dir
            _cam_kw["instance_id_map"] = instance_id_map
        result = curate_viewpoint_change_questions(
            all_cameras,
            max_questions=max_questions,
            rng=rng,
            **_cam_kw,
        )
    elif question_type == "spatial_imagination":
        result = curate_spatial_imagination_questions(
            camera_pose,
            visible,
            max_questions=max_questions,
            all_cameras=all_cameras,
            seg_mask_dir=seg_mask_dir,
            instance_id_map=instance_id_map,
            rng=rng,
        )
    elif question_type == "compound_spatial_referring":
        _csr_visible = visible
        if scene_type == "marble":
            _lc: dict[str, int] = {}
            for o in visible:
                lbl = object_label(o)
                _lc[lbl] = _lc.get(lbl, 0) + 1
            _unique = [o for o in visible if _lc.get(object_label(o), 0) == 1]
            _good_types = _get_csr_good_types(all_objects)
            _csr_visible = [o for o in _unique if _normalize_type(o.get("type", "")) in _good_types]
            if seg_mask_dir and instance_id_map:
                import os as _os2
                import numpy as _np2
                _seg_p = _os2.path.join(seg_mask_dir, f"{camera_pose['name']}_seg.npy")
                if _os2.path.isfile(_seg_p):
                    _m = _np2.load(_seg_p)
                    _total_px = _m.shape[0] * _m.shape[1]
                    _csr_visible = [
                        o for o in _csr_visible
                        if int((_m == instance_id_map.get(o.get("id"), -1)).sum()) / _total_px < 0.15
                    ]
        result = curate_compound_spatial_referring_questions(
            camera_pose,
            _csr_visible,
            max_questions=max_questions,
            all_cameras=all_cameras,
            seg_mask_dir=seg_mask_dir,
            instance_id_map=instance_id_map,
            rng=rng,
        )
    elif question_type == "object_placement":
        result = curate_object_placement_questions(
            camera_pose,
            visible,
            all_objects,
            aux_data_dir=aux_data_dir,
            max_questions=max_questions,
            rng=rng,
        )
    elif question_type == "free_space":
        result = curate_free_space_questions(
            camera_pose,
            visible,
            all_objects,
            aux_data_dir=aux_data_dir,
            max_questions=max_questions,
            rng=rng,
        )
    elif question_type == "spatial_compatibility":
        result = curate_spatial_compatibility_questions(
            camera_pose,
            visible,
            all_objects,
            aux_data_dir=aux_data_dir,
            max_questions=max_questions,
            room_bounds=room_bounds,
            rng=rng,
        )
    elif question_type == "object_matching_mv":
        if not all_cameras:
            raise ValueError("all_cameras is required for object_matching_mv")
        result = curate_object_matching_mv_questions(
            all_cameras,
            all_objects,
            max_questions=max_questions,
            rng=rng,
            visibility_kwargs=visibility_kwargs,
            cam_visibility=cam_visibility,
            seg_mask_dir=seg_mask_dir,
            instance_id_map=instance_id_map,
        )
    elif question_type == "depth_difference":
        result = curate_depth_difference_questions(
            camera_pose,
            visible,
            max_questions=max_questions,
            rng=rng,
        )
    elif question_type == "object_grounding_bbox":
        result = curate_object_grounding_bbox_questions(
            camera_pose,
            visible,
            max_questions=max_questions,
            rng=rng,
        )
    elif question_type == "object_category":
        result = curate_object_category_questions(
            camera_pose,
            visible,
            max_questions=max_questions,
            rng=rng,
        )
    elif question_type == "object_size_qualitative":
        result = curate_object_size_qualitative_questions(
            camera_pose,
            visible,
            max_questions=max_questions,
            rng=rng,
        )
    elif question_type == "comparative_spatial_grounding":
        result = curate_comparative_spatial_grounding_questions(
            camera_pose,
            visible,
            max_questions=max_questions,
            all_objects=all_objects,
            rng=rng,
        )
    elif question_type == "ordinal_grounding":
        if scene_type == "marble":
            return []
        _pose = dict(camera_pose)
        result = curate_ordinal_grounding_questions(
            _pose,
            visible,
            max_questions=max_questions,
            scene_type=scene_type,
            rng=rng,
        )
    elif question_type == "object_object_position_mv":
        if not all_cameras:
            raise ValueError("all_cameras is required for object_object_position_mv")
        result = curate_object_object_position_mv_questions(
            all_cameras,
            all_objects,
            max_questions=max_questions,
            rng=rng,
            visibility_kwargs=visibility_kwargs,
            cam_visibility=cam_visibility,
            seg_mask_dir=seg_mask_dir,
            instance_id_map=instance_id_map,
        )
    elif question_type == "camera_region_position":
        result = curate_camera_region_position_questions(
            camera_pose,
            visible,
            max_questions=max_questions,
            seg_mask_dir=seg_mask_dir,
            instance_id_map=instance_id_map,
            rng=rng,
        )
    elif question_type == "object_region_position":
        result = curate_object_region_position_questions(
            camera_pose,
            visible,
            max_questions=max_questions,
            seg_mask_dir=seg_mask_dir,
            instance_id_map=instance_id_map,
            rng=rng,
        )
    elif question_type == "region_region_position":
        result = curate_region_region_position_questions(
            camera_pose,
            visible,
            max_questions=max_questions,
            seg_mask_dir=seg_mask_dir,
            instance_id_map=instance_id_map,
            rng=rng,
        )
    elif question_type == "route_planning":
        _rp_visible = visible
        if scene_type == "marble":
            _rp_lc: dict[str, int] = {}
            for o in all_objects:
                lbl = object_label(o)
                _rp_lc[lbl] = _rp_lc.get(lbl, 0) + 1
            _rp_visible = [o for o in visible if _rp_lc.get(object_label(o), 0) == 1]
        result = curate_route_planning_questions(
            camera_pose,
            _rp_visible,
            max_questions=max_questions,
            rng=rng,
        )
    else:
        raise ValueError(
            f"Unknown question_type: {question_type!r}. "
            f"Choose one of: {ALL_QUESTION_TYPES}"
        )

    # Post-process: convert MCQ word answers to letter format
    for q in result:
        if (
            "choices" in q
            and isinstance(q["answer"], str)
            and q["answer"] in q["choices"]
        ):
            letter, formatted = _to_letter_mcq(q["answer"], q["choices"])
            q["answer"] = letter
            q["choices"] = formatted
    return result
