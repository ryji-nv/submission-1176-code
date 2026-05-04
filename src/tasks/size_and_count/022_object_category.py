"""Object-category question curator: identify an object's category from a bbox."""

from __future__ import annotations

import random

from src.tasks import _normalize_type, _clean_label
from src.tasks.templates import pick_template

_UNINFORMATIVE_TYPES = {
    "object",
    "mesh",
    "prim",
    "xform",
    "scope",
    "material",
    "light",
    "lighting",
}


def curate_object_category_questions(
    camera_pose: dict,
    visible_objects: list[dict],
    *,
    max_questions: int = 1,
    rng: random.Random,
) -> list[dict]:
    """Mark an object with a bbox and ask what category it belongs to."""
    pool = [
        o
        for o in visible_objects
        if o.get("type")
        and _normalize_type(o["type"]).lower() not in _UNINFORMATIVE_TYPES
    ]
    if not pool:
        return []

    rng.shuffle(pool)
    chosen: list[dict] = []
    seen_types: set[str] = set()
    for obj in pool:
        if len(chosen) >= max_questions:
            break
        t = _normalize_type(obj.get("type", ""))
        if t not in seen_types:
            chosen.append(obj)
            seen_types.add(t)
    for obj in pool:
        if len(chosen) >= max_questions:
            break
        if obj not in chosen:
            chosen.append(obj)

    cam_pos = list(camera_pose["position"])
    cam_look_at = list(camera_pose["look_at"])
    result = []
    for obj in chosen:
        label = _clean_label(obj["type"])
        result.append(
            {
                "type": "object_category",
                "question": pick_template("object_category", rng),
                "answer": label,
                "camera_name": camera_pose["name"],
                "camera_position": cam_pos,
                "camera_look_at": cam_look_at,
                "object": {
                    "label": label,
                    "id": obj.get("id"),
                    "centroid": list(obj["centroid"]),
                    "width": obj.get("width"),
                    "length": obj.get("length"),
                    "height": obj.get("height"),
                },
            }
        )
    return result
