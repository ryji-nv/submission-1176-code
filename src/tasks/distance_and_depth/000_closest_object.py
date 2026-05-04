"""Closest-object question curator: which of two objects is closer to the camera."""

from __future__ import annotations

import random

from src.tasks import object_label, _object_meta, _MAX_TRIALS
from src.tasks.templates import pick_template


def _build_closest_question(
    camera_pose: dict, obj_a: dict, obj_b: dict, rng: random.Random
) -> dict:
    """Build one closest-object question dict for a given pair."""
    label_a = f"{object_label(obj_a)} (blue point)"
    label_b = f"{object_label(obj_b)} (red point)"
    answer = label_a if obj_a["distance"] <= obj_b["distance"] else label_b
    return {
        "type": "closest_object",
        "question": pick_template(
            "closest_object", rng, label_a=label_a, label_b=label_b
        ),
        "choices": [label_a, label_b],
        "answer": answer,
        "camera_name": camera_pose["name"],
        "camera_position": list(camera_pose["position"]),
        "camera_look_at": list(camera_pose["look_at"]),
        "blue": _object_meta(obj_a, "blue"),
        "red": _object_meta(obj_b, "red"),
    }


def curate_closest_object_questions(
    camera_pose: dict,
    visible_objects: list[dict],
    *,
    max_questions: int = 1,
    min_distance_gap: float = 1.0,
    rng: random.Random,
) -> list[dict]:
    """
    Randomly sample pairs of visible objects for "which is closer?" QAs.

    A pair is valid when distance difference >= min_distance_gap and the objects
    have different types (preferred) or same type (fallback).
    Resamples up to _MAX_TRIALS times per question slot; gives up if no valid
    pair is found.
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
            if abs(a["distance"] - b["distance"]) < min_distance_gap:
                continue
            if object_label(a) == object_label(b):
                continue
            found = (a, b)
            used_pairs.add(pair)
            break
        if found:
            result.append(_build_closest_question(camera_pose, *found, rng))

    return result
