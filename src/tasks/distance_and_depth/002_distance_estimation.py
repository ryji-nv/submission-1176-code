"""Distance-estimation question curator: Euclidean distance from camera to a marked object."""

from __future__ import annotations

import random

from src.tasks import object_label
from src.tasks.templates import pick_template


def curate_distance_estimation_questions(
    camera_pose: dict,
    visible_objects: list[dict],
    *,
    max_questions: int = 1,
    rng: random.Random,
) -> list[dict]:
    """
    Randomly pick up to max_questions visible objects, preferring type variety.
    Each question has a numeric answer (float, 2 dp).
    """
    if not visible_objects:
        return []

    pool = list(visible_objects)
    rng.shuffle(pool)

    # One per type first, then fill remainder
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

    camera_pos = list(camera_pose["position"])
    camera_look_at = list(camera_pose["look_at"])

    result = []
    for obj in chosen:
        centroid = obj["centroid"]
        dist = obj["distance"]
        label = object_label(obj)
        result.append(
            {
                "type": "distance_estimation",
                "question": pick_template(
                    "distance_estimation", rng, label=label
                ),
                "answer": str(round(dist, 2)),
                "camera_name": camera_pose["name"],
                "camera_position": camera_pos,
                "camera_look_at": camera_look_at,
                "object": {
                    "label": label,
                    "id": obj.get("id"),
                    "centroid": list(centroid),
                    "distance": dist,
                    "width": obj.get("width"),
                    "length": obj.get("length"),
                    "height": obj.get("height"),
                },
            }
        )
    return result
