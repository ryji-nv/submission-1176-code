"""Depth-estimation question curator: depth along camera Z-axis to a marked object."""

from __future__ import annotations

import random

from src.tasks import object_label, _compute_depth
from src.tasks.templates import pick_template


def curate_depth_estimation_questions(
    camera_pose: dict,
    visible_objects: list[dict],
    *,
    max_questions: int = 1,
    rng: random.Random,
) -> list[dict]:
    """Depth along the camera Z-axis from camera to marked object (metres)."""
    if not visible_objects:
        return []

    pool = list(visible_objects)
    rng.shuffle(pool)
    chosen: list[dict] = []
    seen_types: set[str] = set()
    for obj in pool:
        if len(chosen) >= max_questions:
            break
        if obj.get("type", "") not in seen_types:
            chosen.append(obj)
            seen_types.add(obj.get("type", ""))
    for obj in pool:
        if len(chosen) >= max_questions:
            break
        if obj not in chosen:
            chosen.append(obj)

    cam_pos = tuple(camera_pose["position"])
    look_at = tuple(camera_pose["look_at"])
    result = []
    for obj in chosen:
        label = object_label(obj)
        depth = _compute_depth(tuple(obj["centroid"]), cam_pos, look_at)
        result.append(
            {
                "type": "depth_estimation",
                "question": pick_template("depth_estimation", rng, label=label),
                "answer": str(round(depth, 2)),
                "camera_name": camera_pose["name"],
                "camera_position": list(cam_pos),
                "camera_look_at": list(look_at),
                "object": {
                    "label": label,
                    "id": obj.get("id"),
                    "centroid": list(obj["centroid"]),
                    "depth": round(depth, 2),
                    "width": obj.get("width"),
                    "length": obj.get("length"),
                    "height": obj.get("height"),
                },
            }
        )
    return result
