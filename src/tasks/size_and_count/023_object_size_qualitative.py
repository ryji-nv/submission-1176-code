"""Object-size-qualitative question curator: classify as small / medium / large."""

from __future__ import annotations

import random

from src.tasks import object_label, _longest_dim
from src.tasks.templates import pick_template


def curate_object_size_qualitative_questions(
    camera_pose: dict,
    visible_objects: list[dict],
    *,
    max_questions: int = 1,
    rng: random.Random,
) -> list[dict]:
    """Classify an object as small / medium / large based on its longest dimension."""
    _SMALL_MAX = 0.4
    _MEDIUM_MAX = 1.2
    _MARGIN = 0.1

    candidates = []
    for o in visible_objects:
        longest = _longest_dim(o)
        if longest <= 0:
            continue
        # Skip objects near boundaries
        if _SMALL_MAX - _MARGIN < longest < _SMALL_MAX + _MARGIN:
            continue
        if _MEDIUM_MAX - _MARGIN < longest < _MEDIUM_MAX + _MARGIN:
            continue
        if longest < _SMALL_MAX:
            cat = "A"
        elif longest < _MEDIUM_MAX:
            cat = "B"
        else:
            cat = "C"
        candidates.append((o, cat, longest))

    if not candidates:
        return []

    rng.shuffle(candidates)
    # Try to balance across bins
    by_cat: dict[str, list] = {"A": [], "B": [], "C": []}
    for item in candidates:
        by_cat[item[1]].append(item)
    chosen = []
    for _ in range(max_questions):
        for cat in ("A", "B", "C"):
            if by_cat[cat] and len(chosen) < max_questions:
                chosen.append(by_cat[cat].pop(0))
    if not chosen and candidates:
        chosen = candidates[:max_questions]

    cam_pos = list(camera_pose["position"])
    cam_look_at = list(camera_pose["look_at"])
    result = []
    for obj, cat, longest in chosen:
        label = object_label(obj)
        result.append(
            {
                "type": "object_size_qualitative",
                "question": pick_template(
                    "object_size_qualitative", rng, label=label
                ),
                "choices": ["small", "medium", "large"],
                "answer": {"A": "small", "B": "medium", "C": "large"}[cat],
                "camera_name": camera_pose["name"],
                "camera_position": cam_pos,
                "camera_look_at": cam_look_at,
                "object": {
                    "label": label,
                    "id": obj.get("id"),
                    "centroid": list(obj["centroid"]),
                    "longest_dim": round(longest, 3),
                    "width": obj.get("width"),
                    "length": obj.get("length"),
                    "height": obj.get("height"),
                },
            }
        )
    return result
