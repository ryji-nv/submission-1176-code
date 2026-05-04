"""Object-region-position question curator: cardinal direction from object to area."""

from __future__ import annotations

import math
import random

from src.tasks import (
    object_label,
    _MAX_TRIALS,
    _CARDINAL_8DIR,
    _quantize_cardinal_8dir,
    _identify_areas,
)
from src.tasks.templates import pick_template


def curate_object_region_position_questions(
    camera_pose: dict,
    visible_objects: list[dict],
    *,
    max_questions: int = 1,
    seg_mask_dir: str | None = None,
    instance_id_map: dict | None = None,
    rng: random.Random,
) -> list[dict]:
    """
    Ask in which cardinal direction a functional area is from an object.
    Camera forward = North. Matches MMSI object-region format.
    """
    areas = _identify_areas(
        visible_objects,
        seg_mask_dir=seg_mask_dir,
        cam_name=camera_pose.get("name"),
        instance_id_map=instance_id_map,
    )
    if not areas or len(visible_objects) < 2:
        return []

    cam_pos = tuple(camera_pose["position"])
    look_at = tuple(camera_pose["look_at"])
    fx = look_at[0] - cam_pos[0]
    fy = look_at[1] - cam_pos[1]
    flen = math.sqrt(fx * fx + fy * fy)
    if flen < 1e-9:
        return []
    fx, fy = fx / flen, fy / flen
    rx, ry = fy, -fx

    result: list[dict] = []
    used: set[tuple] = set()
    indices = list(range(len(visible_objects)))
    area_list = list(areas.items())

    for _ in range(max_questions):
        found = None
        for _ in range(_MAX_TRIALS):
            i = rng.choice(indices)
            area_name, info = rng.choice(area_list)
            ax, ay = info["centroid"]
            obj = visible_objects[i]
            key = (obj.get("id"), area_name)
            if key in used:
                continue
            ox, oy = obj["centroid"][0], obj["centroid"][1]
            vx, vy = ax - ox, ay - oy
            dist = math.sqrt(vx * vx + vy * vy)
            if dist < 0.5:
                continue
            dot_fwd = vx * fx + vy * fy
            dot_right = vx * rx + vy * ry
            answer = _quantize_cardinal_8dir(dot_fwd, dot_right)
            if answer is None:
                continue
            used.add(key)
            found = (obj, area_name, info["anchor"], answer)
            break

        if found:
            obj, area_name, anchor_desc, answer = found
            label = object_label(obj)
            ans_idx = _CARDINAL_8DIR.index(answer)
            ref_offset = rng.randint(0, 7)
            ref_dir = _CARDINAL_8DIR[ref_offset]
            rotated_answer = _CARDINAL_8DIR[(ans_idx + ref_offset) % 8]
            rot_idx = _CARDINAL_8DIR.index(rotated_answer)
            far_wrong = [
                d for i, d in enumerate(_CARDINAL_8DIR)
                if min(abs(i - rot_idx), 8 - abs(i - rot_idx)) >= 2
            ]
            rng.shuffle(far_wrong)
            choices = [rotated_answer] + far_wrong[:3]
            rng.shuffle(choices)
            result.append(
                {
                    "type": "object_region_position",
                    "question": pick_template(
                        "object_region_position",
                        rng,
                        ref_dir=ref_dir.lower(),
                        area_name=area_name,
                        label=label,
                    ),
                    "choices": choices,
                    "answer": rotated_answer,
                    "camera_name": camera_pose["name"],
                    "camera_position": list(cam_pos),
                    "camera_look_at": list(look_at),
                    "object": {
                        "label": label,
                        "id": obj.get("id"),
                        "centroid": list(obj["centroid"]),
                    },
                    "area_name": area_name,
                }
            )

    return result
