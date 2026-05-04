"""Region-region-position question curator: cardinal direction between functional areas."""

from __future__ import annotations

import math
import random

from src.tasks import (
    _MAX_TRIALS,
    _CARDINAL_8DIR,
    _quantize_cardinal_8dir,
    _identify_areas,
)
from src.tasks.templates import pick_template


def curate_region_region_position_questions(
    camera_pose: dict,
    visible_objects: list[dict],
    *,
    max_questions: int = 1,
    seg_mask_dir: str | None = None,
    instance_id_map: dict | None = None,
    rng: random.Random,
) -> list[dict]:
    """
    Ask in which cardinal direction one functional area is from another.
    Camera forward = North. Matches MMSI region-region format.
    """
    areas = _identify_areas(
        visible_objects,
        seg_mask_dir=seg_mask_dir,
        cam_name=camera_pose.get("name"),
        instance_id_map=instance_id_map,
    )
    if len(areas) < 2:
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

    area_list = list(areas.items())
    result: list[dict] = []
    used: set[tuple] = set()

    for _ in range(max_questions):
        found = None
        for _ in range(_MAX_TRIALS):
            (name_a, info_a), (name_b, info_b) = rng.sample(area_list, 2)
            key = (name_a, name_b)
            if key in used:
                continue
            ax, ay = info_a["centroid"]
            bx, by = info_b["centroid"]
            vx, vy = bx - ax, by - ay
            dist = math.sqrt(vx * vx + vy * vy)
            if dist < 0.5:
                continue
            dot_fwd = vx * fx + vy * fy
            dot_right = vx * rx + vy * ry
            answer = _quantize_cardinal_8dir(dot_fwd, dot_right)
            if answer is None:
                continue
            used.add(key)
            found = (name_a, info_a, name_b, info_b, answer)
            break

        if found:
            name_a, info_a, name_b, info_b, answer = found
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
                    "type": "region_region_position",
                    "question": pick_template(
                        "region_region_position",
                        rng,
                        ref_dir=ref_dir.lower(),
                        name_a=name_a,
                        name_b=name_b,
                    ),
                    "choices": choices,
                    "answer": rotated_answer,
                    "camera_name": camera_pose["name"],
                    "camera_position": list(cam_pos),
                    "camera_look_at": list(look_at),
                    "area_a": name_a,
                    "area_b": name_b,
                }
            )

    return result
