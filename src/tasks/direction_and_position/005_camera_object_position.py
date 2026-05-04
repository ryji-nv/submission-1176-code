"""Camera-object-position question curator: which image quadrant an object occupies."""

from __future__ import annotations

import math
import random

from src.tasks import object_label
from src.tasks.templates import pick_template
from src.utils.projection import project_world_to_fraction


def curate_camera_object_position_questions(
    camera_pose: dict,
    visible_objects: list[dict],
    *,
    max_questions: int = 1,
    min_offset: float = 0.15,
    rng: random.Random,
) -> list[dict]:
    """
    Ask which image quadrant the marked object occupies.
    min_offset: minimum distance from image center (as fraction) in both u and v
    to ensure the object is clearly in one quadrant.
    """
    if not visible_objects:
        return []

    fov = camera_pose.get("horizontal_fov_deg", 82.0)
    cam_pos = tuple(camera_pose["position"])
    look_at = tuple(camera_pose["look_at"])

    candidates = []
    for obj in visible_objects:
        uv = project_world_to_fraction(
            obj["centroid"], cam_pos, look_at, horizontal_fov_deg=fov
        )
        if uv is None:
            continue
        u, v = uv
        if abs(u - 0.5) >= min_offset and abs(v - 0.5) >= min_offset:
            candidates.append((obj, u, v))

    if not candidates:
        return []

    rng.shuffle(candidates)
    result = []
    seen_types: set[str] = set()
    for obj, u, v in candidates:
        if len(result) >= max_questions:
            break
        if obj.get("type", "") in seen_types:
            continue
        seen_types.add(obj.get("type", ""))
        # 3D camera-frame relation (SPAR convention)
        cx, cy, cz = obj["centroid"]
        dx = cx - cam_pos[0]
        dy = cy - cam_pos[1]
        dz = cz - cam_pos[2]
        _fx = look_at[0] - cam_pos[0]
        _fy = look_at[1] - cam_pos[1]
        _flen = math.sqrt(_fx * _fx + _fy * _fy)
        _fx, _fy = _fx / _flen, _fy / _flen
        _rx, _ry = _fy, -_fx
        dot_r = dx * _rx + dy * _ry
        dot_f = dx * _fx + dy * _fy
        _t = 0.1
        lr = "right" if dot_r > _t else ("left" if dot_r < -_t else "")
        ud = "above" if dz > _t else ("below" if dz < -_t else "")
        fb = "front" if dot_f > _t else ("behind" if dot_f < -_t else "")
        answer = f"{lr}, {ud}, {fb}"
        label = object_label(obj)
        _oc_axes = [
            ["left", "right", ""],
            ["above", "below", ""],
            ["front", "behind", ""],
        ]
        all_combos = [
            f"{a}, {b}, {c}"
            for a in _oc_axes[0]
            for b in _oc_axes[1]
            for c in _oc_axes[2]
        ]
        wrong = [c for c in all_combos if c != answer]
        rng.shuffle(wrong)
        choices = [answer] + wrong[:3]
        rng.shuffle(choices)
        result.append(
            {
                "type": "camera_object_position",
                "question": pick_template(
                    "camera_object_position", rng, label=label
                ),
                "choices": choices,
                "answer": answer,
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
