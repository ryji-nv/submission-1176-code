"""Object-grounding-bbox question curator: locate an object via bounding box."""

from __future__ import annotations

import random

from src.tasks import object_label, _normalize_type, _project_bbox_2d
from src.tasks.templates import pick_template


def curate_object_grounding_bbox_questions(
    camera_pose: dict,
    visible_objects: list[dict],
    *,
    max_questions: int = 1,
    min_bbox_size: float = 0.03,
    rng: random.Random,
) -> list[dict]:
    """
    Locate a unique-type object via bounding box [x_min, y_min, x_max, y_max] in 0-1.
    Same candidate selection as object_grounding (unique-type filter).
    """
    if not visible_objects:
        return []

    fov = camera_pose.get("horizontal_fov_deg", 82.0)
    cam_pos = tuple(camera_pose["position"])
    look_at = tuple(camera_pose["look_at"])

    type_counts: dict[str, int] = {}
    for obj in visible_objects:
        t = _normalize_type(obj.get("type", ""))
        type_counts[t] = type_counts.get(t, 0) + 1

    candidates = []
    for obj in visible_objects:
        if type_counts.get(_normalize_type(obj.get("type", "")), 0) != 1:
            continue
        bbox = _project_bbox_2d(obj, cam_pos, look_at, fov)
        if bbox is None:
            continue
        x_min, y_min, x_max, y_max = bbox
        # Convert from 0-1000 space to 0-1
        nx_min = round(x_min / 1000, 3)
        ny_min = round(y_min / 1000, 3)
        nx_max = round(x_max / 1000, 3)
        ny_max = round(y_max / 1000, 3)
        if (nx_max - nx_min) < min_bbox_size or (ny_max - ny_min) < min_bbox_size:
            continue
        candidates.append((obj, (nx_min, ny_min, nx_max, ny_max)))

    if not candidates:
        return []

    rng.shuffle(candidates)
    chosen = candidates[:max_questions]

    result = []
    for obj, (bx_min, by_min, bx_max, by_max) in chosen:
        label = object_label(obj)
        result.append(
            {
                "type": "object_grounding_bbox",
                "question": pick_template(
                    "object_grounding_bbox", rng, label=label
                ),
                "answer": f"[{bx_min}, {by_min}, {bx_max}, {by_max}]",
                "camera_name": camera_pose["name"],
                "camera_position": list(cam_pos),
                "camera_look_at": list(look_at),
                "object": {
                    "label": label,
                    "id": obj.get("id"),
                    "centroid": list(obj["centroid"]),
                    "bbox_2d": [bx_min, by_min, bx_max, by_max],
                    "width": obj.get("width"),
                    "length": obj.get("length"),
                    "height": obj.get("height"),
                },
            }
        )
    return result
