"""Object-placement question curator: find valid placement locations on surfaces."""

from __future__ import annotations

import random

import numpy as np

from src.tasks import object_label, _MAX_TRIALS
from src.tasks.templates import pick_template
from src.utils.projection import project_world_to_fraction
from src.utils.surface import (
    identify_surface_objects,
    surface_top_z,
    objects_on_surface,
    load_aux_data,
    find_empty_surface_pixels,
    footprint_to_pixel_kernel,
    find_valid_placement_centers,
    check_placement_feasibility,
)


def _obj_footprint(obj: dict) -> tuple[float, float]:
    return (obj.get("width", 0.1), obj.get("length", 0.1))


_SEAT_FRACTION_KEYWORDS: list[tuple[str, float]] = [
    ("armchair", 0.50),
    ("chair", 0.50),
    ("sofa", 0.50),
    ("couch", 0.50),
    ("recliner", 0.50),
    ("throne", 0.50),
]


def _placement_surface_z(obj: dict) -> float:
    """Estimate Z of the usable horizontal surface (seat level for seating furniture)."""
    z = obj.get("z", 0.0)
    h = obj.get("height", 0.0)
    obj_type = str(obj.get("type", obj.get("id", ""))).lower()
    for kw, frac in _SEAT_FRACTION_KEYWORDS:
        if kw in obj_type:
            return z + h * frac
    return z + h


def _placement_obj_meta(obj: dict) -> dict:
    return {
        "label": object_label(obj),
        "id": obj.get("id"),
        "centroid": [obj["x"], obj["y"], obj.get("z", 0) + obj.get("height", 0) / 2],
        "surface_point": [obj["x"], obj["y"], _placement_surface_z(obj)],
        "width": obj.get("width"),
        "length": obj.get("length"),
        "height": obj.get("height"),
    }


def _find_placement_uv(
    footprint: tuple[float, float],
    surface: dict,
    all_objects: list[dict],
    cam_pos: tuple,
    look_at: tuple,
    fov: float,
    aux: dict | None,
    rng: random.Random,
) -> tuple[float, float] | None:
    """Find a valid placement [u,v] on surface using pixel path or geometry fallback."""
    surf_id = surface.get("id", "")

    # Pixel path: erosion-based feasibility using normals + instance seg
    if aux is not None and surf_id:
        mask = find_empty_surface_pixels(aux, surf_id)
        if mask.any():
            surf_center = (
                surface["x"],
                surface["y"],
                surface.get("z", 0.0) + surface.get("height", 0.0),
            )
            img_h, img_w = aux["seg"].shape[:2]
            kernel = footprint_to_pixel_kernel(
                footprint,
                surf_center,
                cam_pos,
                look_at,
                img_w,
                img_h,
                horizontal_fov_deg=fov,
            )
            if kernel is not None:
                valid = find_valid_placement_centers(mask, kernel[0], kernel[1])
                ys, xs = np.nonzero(valid)
                if len(ys) > 0:
                    np_rng = np.random.default_rng(rng.randint(0, 2**31))
                    idx = np_rng.integers(len(ys))
                    return (float(xs[idx]) / img_w, float(ys[idx]) / img_h)

    # Geometry fallback
    on_surface = objects_on_surface(surface, all_objects)
    sx, sy = surface["x"], surface["y"]
    sw, sl = surface.get("width", 0), surface.get("length", 0)
    for _ in range(_MAX_TRIALS):
        tx = sx + (rng.random() - 0.5) * sw * 0.8
        ty = sy + (rng.random() - 0.5) * sl * 0.8
        tz = surface_top_z(surface)
        if check_placement_feasibility(surface, (tx, ty), footprint, on_surface):
            uv = project_world_to_fraction(
                (tx, ty, tz), cam_pos, look_at, horizontal_fov_deg=fov
            )
            if uv and 0.0 <= uv[0] <= 1.0 and 0.0 <= uv[1] <= 1.0:
                return uv
    return None


def curate_object_placement_questions(
    camera_pose: dict,
    visible_objects: list[dict],
    all_objects: list[dict] | None = None,
    *,
    aux_data_dir: str | None = None,
    max_questions: int = 1,
    rng: random.Random,
) -> list[dict]:
    """
    Ask where on a surface a scene object can be placed.

    Uses pixel-based erosion (normals + instance seg) when aux data is
    available, otherwise falls back to geometry checks.
    Only positive placements (object fits) are emitted.

    Image annotation: yellow dot on moveable, blue dot on target surface.
    Answer: [u, v] normalized image coordinates.
    """
    surfaces = identify_surface_objects(visible_objects)
    if not surfaces:
        return []

    surface_ids = {s.get("id") for s in surfaces}
    candidates = [
        o
        for o in visible_objects
        if o.get("id") not in surface_ids
        and o.get("width", 0) > 0
        and o.get("length", 0) > 0
    ]
    if not candidates:
        return []

    if all_objects is None:
        all_objects = visible_objects

    fov = camera_pose.get("horizontal_fov_deg", 82.0)
    cam_pos = tuple(camera_pose["position"])
    look_at = tuple(camera_pose["look_at"])

    aux = load_aux_data(aux_data_dir, camera_pose["name"]) if aux_data_dir else None

    rng.shuffle(candidates)
    rng.shuffle(surfaces)
    result: list[dict] = []
    used_pairs: set[tuple[str, str]] = set()

    for moveable in candidates:
        if len(result) >= max_questions:
            break
        mov_id = moveable.get("id", "")
        footprint = _obj_footprint(moveable)

        for surf in surfaces:
            if len(result) >= max_questions:
                break
            surf_id = surf.get("id", "")
            if (mov_id, surf_id) in used_pairs:
                continue

            placement_uv = _find_placement_uv(
                footprint, surf, all_objects, cam_pos, look_at, fov, aux, rng
            )
            if placement_uv is None:
                continue

            used_pairs.add((mov_id, surf_id))
            result.append(
                {
                    "type": "object_placement",
                    "question": pick_template(
                        "object_placement", rng, label=object_label(surf)
                    ),
                    "answer": f"[({round(placement_uv[0], 3)}, {round(placement_uv[1], 3)})]",
                    "camera_name": camera_pose["name"],
                    "camera_position": list(cam_pos),
                    "camera_look_at": list(look_at),
                    "moveable_object": _placement_obj_meta(moveable),
                    "surface": _placement_obj_meta(surf),
                    "footprint": [round(footprint[0], 4), round(footprint[1], 4)],
                }
            )
            break

    return result
