"""Depth-ordering question curator: order multiple objects from nearest to farthest."""

from __future__ import annotations

import itertools
import random

from src.tasks import _object_meta, _compute_depth, _MAX_TRIALS
from src.tasks.templates import pick_template


def curate_depth_ordering_questions(
    camera_pose: dict,
    visible_objects: list[dict],
    *,
    max_questions: int = 1,
    n_objects: int = 3,
    min_depth_spread: float = 1.0,
    min_depth_ratio: float = 1.3,
    rng: random.Random,
) -> list[dict]:
    """
    Pick n_objects visible objects and ask to order them from nearest to farthest.
    Answer and choices are ordered lists of color labels (blue, red, green).
    """
    _COLORS = ["blue", "red", "green"]
    if len(visible_objects) < n_objects or n_objects > len(_COLORS):
        return []

    cam_pos = tuple(camera_pose["position"])
    look_at = tuple(camera_pose["look_at"])
    indices = list(range(len(visible_objects)))
    result = []

    for _ in range(max_questions):
        found = None
        for _ in range(_MAX_TRIALS):
            chosen_idx = rng.sample(indices, n_objects)
            objs = [visible_objects[i] for i in chosen_idx]
            depths = [
                _compute_depth(tuple(o["centroid"]), cam_pos, look_at) for o in objs
            ]
            d_min, d_max = min(depths), max(depths)
            if d_min <= 0:
                continue
            if d_max - d_min < min_depth_spread and d_max / d_min < min_depth_ratio:
                continue
            found = (objs, depths)
            break
        if not found:
            continue

        objs, depths = found
        color_labels = _COLORS[:n_objects]
        sorted_order = sorted(range(n_objects), key=lambda i: depths[i])
        answer = ", ".join(color_labels[i] for i in sorted_order)

        all_perms = [
            ", ".join(color_labels[i] for i in p)
            for p in itertools.permutations(range(n_objects))
        ]
        wrong = [p for p in all_perms if p != answer]
        rng.shuffle(wrong)
        choices = [answer] + wrong[:3]
        rng.shuffle(choices)

        result.append(
            {
                "type": "depth_ordering",
                "question": pick_template("depth_ordering", rng),
                "choices": choices,
                "answer": answer,
                "camera_name": camera_pose["name"],
                "camera_position": list(cam_pos),
                "camera_look_at": list(look_at),
                "objects": [
                    {**_object_meta(obj, color_labels[i]), "depth": round(depths[i], 2)}
                    for i, obj in enumerate(objs)
                ],
            }
        )
    return result
