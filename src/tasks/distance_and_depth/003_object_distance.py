"""Object-distance question curator (Dist-OO): Euclidean distance between two object centroids."""

from __future__ import annotations

import math
import random

from src.tasks import object_label, _object_meta, _MAX_TRIALS
from src.tasks.templates import pick_template


def curate_object_distance_questions(
    camera_pose: dict,
    visible_objects: list[dict],
    *,
    max_questions: int = 1,
    min_object_gap: float = 0.5,
    rng: random.Random,
) -> list[dict]:
    """
    Randomly sample pairs of visible objects for object-distance QAs.
    Answer is the Euclidean distance between centroids (metres, 2 dp).
    Pairs closer than min_object_gap metres are skipped.
    Resamples up to _MAX_TRIALS times per question slot to get unique pairs.
    """
    if len(visible_objects) < 2:
        return []

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
            a, b = visible_objects[i], visible_objects[j]
            ax, ay, az = a["centroid"]
            bx, by, bz = b["centroid"]
            dist = math.sqrt((ax - bx) ** 2 + (ay - by) ** 2 + (az - bz) ** 2)
            if dist < min_object_gap:
                continue
            found = (a, b, dist)
            used_pairs.add(pair)
            break
        if found:
            obj_a, obj_b, dist = found
            result.append(
                {
                    "type": "object_distance",
                    "question": pick_template(
                        "object_distance",
                        rng,
                        label_a=object_label(obj_a),
                        label_b=object_label(obj_b),
                    ),
                    "answer": str(round(dist, 2)),
                    "camera_name": camera_pose["name"],
                    "camera_position": list(camera_pose["position"]),
                    "camera_look_at": list(camera_pose["look_at"]),
                    "blue": _object_meta(obj_a, "blue"),
                    "red": _object_meta(obj_b, "red"),
                }
            )

    return result
