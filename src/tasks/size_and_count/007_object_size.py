"""Object-size question curator: longest dimension of a marked object."""

from __future__ import annotations

import random

from src.tasks import object_label
from src.tasks.templates import pick_template


def curate_object_size_questions(
    camera_pose: dict,
    visible_objects: list[dict],
    *,
    max_questions: int = 1,
    rng: random.Random,
) -> list[dict]:
    """Ask for the longest dimension (width, length, or height) of a marked object."""
    candidates = [
        o
        for o in visible_objects
        if o.get("width") and o.get("length") and o.get("height")
    ]
    if not candidates:
        return []

    pool = list(candidates)
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
        longest = round(max(obj["width"], obj["length"], obj["height"]), 2)
        result.append(
            {
                "type": "object_size",
                "question": pick_template("object_size", rng, label=label),
                "answer": str(round(longest * 100, 1)),
                "camera_name": camera_pose["name"],
                "camera_position": list(cam_pos),
                "camera_look_at": list(look_at),
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
