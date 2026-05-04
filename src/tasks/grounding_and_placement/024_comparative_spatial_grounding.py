"""Comparative-spatial-grounding question curator: point to the X closest/farthest from Y."""

from __future__ import annotations

import random

import numpy as np

from src.tasks import object_label, _normalize_type, _clean_label
from src.tasks.templates import pick_template
from src.utils.projection import project_world_to_fraction


def curate_comparative_spatial_grounding_questions(
    camera_pose: dict,
    visible_objects: list[dict],
    *,
    max_questions: int = 1,
    min_distance_gap: float = 0.5,
    all_objects: list[dict] | None = None,
    rng: random.Random,
) -> list[dict]:
    """
    'Point to the chair closest to the table.'
    Requires multi-instance target type + unique-type anchor.
    """
    if len(visible_objects) < 3:
        return []

    fov = camera_pose.get("horizontal_fov_deg", 82.0)
    cam_pos = tuple(camera_pose["position"])
    look_at = tuple(camera_pose["look_at"])

    # Group by normalized type across ALL objects (scene-wide) for anchor uniqueness
    all_type_counts: dict[str, int] = {}
    for obj in all_objects if all_objects else visible_objects:
        t = _normalize_type(obj.get("type", ""))
        if t:
            all_type_counts[t] = all_type_counts.get(t, 0) + 1

    type_groups: dict[str, list[dict]] = {}
    for obj in visible_objects:
        t = _normalize_type(obj.get("type", ""))
        if t:
            type_groups.setdefault(t, []).append(obj)

    multi_types = {t: objs for t, objs in type_groups.items() if len(objs) >= 2}
    unique_types = {t: objs[0] for t, objs in type_groups.items()
                    if len(objs) == 1 and all_type_counts.get(t, 0) == 1}

    if not multi_types or not unique_types:
        return []

    candidates = []
    for target_type, targets in multi_types.items():
        for anchor_type, anchor in unique_types.items():
            anchor_c = np.array(anchor["centroid"])
            dists = [
                (obj, np.linalg.norm(np.array(obj["centroid"]) - anchor_c))
                for obj in targets
            ]
            dists.sort(key=lambda x: x[1])
            if len(dists) < 2:
                continue
            closest, d_close = dists[0]
            _, d_second = dists[1]
            if d_second - d_close < min_distance_gap:
                continue
            # Also check farthest
            farthest, d_far = dists[-1]
            _, d_second_far = dists[-2]
            if d_far - d_second_far < min_distance_gap:
                far_valid = False
            else:
                far_valid = True
            candidates.append((target_type, anchor, closest, farthest, far_valid))

    if not candidates:
        return []

    rng.shuffle(candidates)
    result = []
    for target_type, anchor, closest, farthest, far_valid in candidates:
        if len(result) >= max_questions:
            break
        # Randomly choose "closest to" or "farthest from"
        if far_valid and rng.random() < 0.5:
            winner = farthest
            relation = "farthest from"
        else:
            winner = closest
            relation = "closest to"

        uv = project_world_to_fraction(
            winner["centroid"], cam_pos, look_at, horizontal_fov_deg=fov
        )
        if uv is None:
            continue

        target_label = _clean_label(target_type)
        anchor_label = object_label(anchor)
        result.append(
            {
                "type": "comparative_spatial_grounding",
                "question": pick_template(
                    "comparative_spatial_grounding",
                    rng,
                    target_label=target_label,
                    relation=relation,
                    anchor_label=anchor_label,
                ),
                "answer": f"[({round(uv[0], 3)}, {round(uv[1], 3)})]",
                "camera_name": camera_pose["name"],
                "camera_position": list(cam_pos),
                "camera_look_at": list(look_at),
                "object": {
                    "label": target_label,
                    "id": winner.get("id"),
                    "centroid": list(winner["centroid"]),
                    "image_u": round(uv[0], 3),
                    "image_v": round(uv[1], 3),
                    "width": winner.get("width"),
                    "length": winner.get("length"),
                    "height": winner.get("height"),
                },
                "anchor": {
                    "label": anchor_label,
                    "id": anchor.get("id"),
                    "centroid": list(anchor["centroid"]),
                },
                "relation": relation,
            }
        )

    return result
