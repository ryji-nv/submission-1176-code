"""Object-count question curator: how many instances of a category are visible."""

from __future__ import annotations

import random

from src.tasks import _normalize_type, _clean_label, object_label


def _longest_dim(obj: dict) -> float:
    return max(obj.get("width") or 0, obj.get("length") or 0, obj.get("height") or 0)


def curate_object_count_questions(
    camera_pose: dict,
    visible_objects: list[dict],
    *,
    max_questions: int = 1,
    min_single_size: float = 0.5,
    min_obj_size: float = 0.0,
    rng: random.Random,
) -> list[dict]:
    """
    Ask how many instances of a given object category are visible.
    For singleton types (count == 1), the object's longest dimension must
    exceed min_single_size so the question targets clearly visible objects.
    The image is unannotated — the model must find and count on its own.
    """
    if not visible_objects:
        return []
    type_groups: dict[str, list[dict]] = {}
    for obj in visible_objects:
        t = _normalize_type(obj.get("type", ""))
        if not t or _longest_dim(obj) < min_obj_size:
            continue
        type_groups.setdefault(t, []).append(obj)

    multi = []
    single = []
    for t, objs in type_groups.items():
        if len(objs) >= 2:
            multi.append((t, objs))
        elif _longest_dim(objs[0]) >= min_single_size:
            single.append((t, objs))
    rng.shuffle(multi)
    rng.shuffle(single)
    candidates = multi + single
    if not candidates:
        return []

    rng.shuffle(candidates)
    chosen = candidates[:max_questions]

    cam_pos = list(camera_pose["position"])
    cam_look_at = list(camera_pose["look_at"])
    result = []
    for obj_type, objs in chosen:
        clean_type = _clean_label(obj_type)
        result.append(
            {
                "type": "object_count",
                "question": f"How many {clean_type}(s) are in this room?",
                "answer": str(len(objs)),
                "camera_name": camera_pose["name"],
                "camera_position": cam_pos,
                "camera_look_at": cam_look_at,
                "counted_type": clean_type,
                "objects": [
                    {
                        "label": object_label(o),
                        "id": o.get("id"),
                        "centroid": list(o["centroid"]),
                        "distance_to_camera": o["distance"],
                    }
                    for o in objs
                ],
            }
        )
    return result
