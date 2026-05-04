"""Object-grounding question curator: locate an object by pointing to image coordinates."""

from __future__ import annotations

import random

from src.tasks import _normalize_type, object_label
from src.tasks.templates import pick_template
from src.utils.projection import project_world_to_fraction


def curate_object_grounding_questions(
    camera_pose: dict,
    visible_objects: list[dict],
    *,
    max_questions: int = 1,
    rng: random.Random,
) -> list[dict]:
    """
    Ask the model to locate an object by pointing to its image coordinates.
    Answer is [u, v] in [0, 1] x [0, 1] (top-left origin).
    The question image is unannotated so the model must find the object itself.
    """
    if not visible_objects:
        return []

    fov = camera_pose.get("horizontal_fov_deg", 82.0)
    cam_pos = tuple(camera_pose["position"])
    look_at = tuple(camera_pose["look_at"])

    # Count how many visible objects share each normalized type — only unique types are unambiguous.
    type_counts: dict[str, int] = {}
    for obj in visible_objects:
        t = _normalize_type(obj.get("type", ""))
        type_counts[t] = type_counts.get(t, 0) + 1

    candidates = []
    for obj in visible_objects:
        if type_counts.get(_normalize_type(obj.get("type", "")), 0) != 1:
            continue  # skip: another object of the same type is visible
        uv = project_world_to_fraction(
            obj["centroid"], cam_pos, look_at, horizontal_fov_deg=fov
        )
        if uv is not None:
            candidates.append((obj, uv))

    if not candidates:
        return []

    rng.shuffle(candidates)
    chosen: list[tuple] = []
    for item in candidates:
        if len(chosen) >= max_questions:
            break
        chosen.append(item)

    result = []
    for obj, (u, v) in chosen:
        label = object_label(obj)
        result.append(
            {
                "type": "object_grounding",
                "question": pick_template("object_grounding", rng, label=label),
                "answer": f"[({round(u, 3)}, {round(v, 3)})]",
                "camera_name": camera_pose["name"],
                "camera_position": list(cam_pos),
                "camera_look_at": list(look_at),
                "object": {
                    "label": label,
                    "id": obj.get("id"),
                    "centroid": list(obj["centroid"]),
                    "image_u": round(u, 3),
                    "image_v": round(v, 3),
                    "width": obj.get("width"),
                    "length": obj.get("length"),
                    "height": obj.get("height"),
                },
            }
        )
    return result
