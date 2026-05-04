"""Depth-difference question curator: depth gap between two objects along camera axis."""

from __future__ import annotations

import random

from src.tasks import (
    object_label,
    _object_meta,
    _normalize_type,
    _compute_depth,
    _MAX_TRIALS,
)
from src.tasks.templates import pick_template


def curate_depth_difference_questions(
    camera_pose: dict,
    visible_objects: list[dict],
    *,
    max_questions: int = 1,
    min_depth_gap: float = 0.5,
    rng: random.Random,
) -> list[dict]:
    """Depth difference between two visible objects along the camera forward axis (metres)."""
    if len(visible_objects) < 2:
        return []

    cam_pos = tuple(camera_pose["position"])
    look_at = tuple(camera_pose["look_at"])

    # Pre-compute depths
    depths = []
    for obj in visible_objects:
        d = _compute_depth(tuple(obj["centroid"]), cam_pos, look_at)
        depths.append(d)

    indices = list(range(len(visible_objects)))
    result: list[dict] = []
    used_pairs: set[tuple[int, int]] = set()

    for _ in range(max_questions):
        found = None
        for _ in range(_MAX_TRIALS):
            i, j = rng.sample(indices, 2)
            pair = (min(i, j), max(i, j))
            if pair in used_pairs:
                continue
            if abs(depths[i] - depths[j]) < min_depth_gap:
                continue
            # Prefer type-diverse pairs
            a, b = visible_objects[i], visible_objects[j]
            if _normalize_type(a.get("type", "")) != _normalize_type(b.get("type", "")):
                found = (i, j)
                used_pairs.add(pair)
                break
            found = (i, j)
            used_pairs.add(pair)
        if found is None:
            continue
        i, j = found
        obj_a, obj_b = visible_objects[i], visible_objects[j]
        label_a = object_label(obj_a)
        label_b = object_label(obj_b)
        diff = abs(depths[i] - depths[j])
        result.append(
            {
                "type": "depth_difference",
                "question": pick_template(
                    "depth_difference",
                    rng,
                    label_a=label_a,
                    label_b=label_b,
                ),
                "answer": str(round(diff, 2)),
                "camera_name": camera_pose["name"],
                "camera_position": list(cam_pos),
                "camera_look_at": list(look_at),
                "blue": _object_meta(obj_a, "blue"),
                "red": _object_meta(obj_b, "red"),
            }
        )

    return result
