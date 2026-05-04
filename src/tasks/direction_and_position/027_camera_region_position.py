"""Camera-region-position question curator: direction to a functional area from camera."""

from __future__ import annotations

import math
import random

from src.tasks import _8DIR_NAMES, _quantize_8dir, _identify_areas
from src.tasks.templates import pick_template


def curate_camera_region_position_questions(
    camera_pose: dict,
    visible_objects: list[dict],
    *,
    max_questions: int = 1,
    seg_mask_dir: str | None = None,
    instance_id_map: dict | None = None,
    rng: random.Random,
) -> list[dict]:
    """
    Ask in which direction a functional area (near its anchor object) is
    relative to the camera. Single-room variant of MMSI camera-region.
    """
    areas = _identify_areas(
        visible_objects,
        seg_mask_dir=seg_mask_dir,
        cam_name=camera_pose.get("name"),
        instance_id_map=instance_id_map,
    )
    if not areas:
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
    rng.shuffle(area_list)
    result: list[dict] = []
    used: set[str] = set()

    for area_name, info in area_list:
        if len(result) >= max_questions:
            break
        if area_name in used:
            continue
        ax, ay = info["centroid"]
        vx, vy = ax - cam_pos[0], ay - cam_pos[1]
        dist = math.sqrt(vx * vx + vy * vy)
        if dist < 0.5:
            continue
        dot_fwd = vx * fx + vy * fy
        dot_right = vx * rx + vy * ry
        answer = _quantize_8dir(dot_fwd, dot_right)
        used.add(area_name)

        ans_idx = _8DIR_NAMES.index(answer)
        far_wrong = [
            d for i, d in enumerate(_8DIR_NAMES)
            if min(abs(i - ans_idx), 8 - abs(i - ans_idx)) >= 2
        ]
        rng.shuffle(far_wrong)
        choices = [answer] + far_wrong[:3]
        rng.shuffle(choices)

        result.append(
            {
                "type": "camera_region_position",
                "question": pick_template(
                    "camera_region_position", rng, area_name=area_name
                ),
                "choices": choices,
                "answer": answer,
                "camera_name": camera_pose["name"],
                "camera_position": list(cam_pos),
                "camera_look_at": list(look_at),
                "area_name": area_name,
                "area_centroid": [ax, ay],
            }
        )

    return result
