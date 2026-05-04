"""Spatial-compatibility question curator: can one object fit beside another."""

from __future__ import annotations

import math
import random

from src.tasks import object_label
from src.tasks.templates import pick_template
from src.utils.projection import project_world_to_fraction


def curate_spatial_compatibility_questions(
    camera_pose: dict,
    visible_objects: list[dict],
    all_objects: list[dict] | None = None,
    *,
    aux_data_dir: str | None = None,
    max_questions: int = 1,
    room_bounds: list[float] | None = None,
    rng: random.Random,
) -> list[dict]:
    """
    Ask whether one object can be placed on another.

    Any pair of visible objects can be moveable/target. Uses pixel path
    when aux data is available, otherwise geometry fallback.

    Matches RoboSpatial: uses all visible objects, specific direction
    (left/right/behind/in front of), simple collision check.
    """
    objs = [
        o for o in visible_objects if o.get("width", 0) > 0 and o.get("length", 0) > 0
    ]
    if len(objs) < 2:
        return []

    if all_objects is None:
        all_objects = visible_objects

    cam_pos = tuple(camera_pose["position"])
    look_at = tuple(camera_pose["look_at"])
    _fx = look_at[0] - cam_pos[0]
    _fy = look_at[1] - cam_pos[1]
    _flen = math.sqrt(_fx * _fx + _fy * _fy)
    if _flen < 1e-9:
        return []
    _fx, _fy = _fx / _flen, _fy / _flen
    _rx, _ry = _fy, -_fx

    _dirs = [
        ("left of", (-_rx, -_ry)),
        ("right of", (_rx, _ry)),
    ]

    pairs = [(a, b) for a in objs for b in objs if a.get("id") != b.get("id")]
    rng.shuffle(pairs)

    result: list[dict] = []
    used: set[tuple] = set()

    for moveable, target in pairs:
        if len(result) >= max_questions:
            break
        dir_name, dir_vec = rng.choice(_dirs)
        key = (moveable.get("id", ""), target.get("id", ""), dir_name)
        if key in used:
            continue

        # Skip if moveable is already in that direction from target
        dx = moveable.get("x", 0) - target.get("x", 0)
        dy = moveable.get("y", 0) - target.get("y", 0)
        dot_dir = dx * dir_vec[0] + dy * dir_vec[1]
        if dot_dir > 1.5:
            continue

        # Also skip if moveable already appears in that direction in image
        mz = moveable.get("z", 0) + moveable.get("height", 0) / 2
        tz = target.get("z", 0) + target.get("height", 0) / 2
        m_uv = project_world_to_fraction(
            (moveable.get("x", 0), moveable.get("y", 0), mz),
            cam_pos,
            look_at,
            horizontal_fov_deg=82,
        )
        t_uv = project_world_to_fraction(
            (target.get("x", 0), target.get("y", 0), tz),
            cam_pos,
            look_at,
            horizontal_fov_deg=82,
        )
        if m_uv and t_uv:
            if dir_name == "left of" and m_uv[0] < t_uv[0] - 0.03:
                continue
            if dir_name == "right of" and m_uv[0] > t_uv[0] + 0.03:
                continue

        aw = min(moveable.get("width", 0.3), moveable.get("length", 0.3))
        bw = target.get("width", 0.3)
        bl = target.get("length", 0.3)
        half_b = max(bw, bl) / 2

        fits = False
        for extra in [0.0, 0.3, 0.6, -0.3, -0.6]:
            offset = half_b + aw / 2 + 0.05 + abs(extra)
            perp_x = -dir_vec[1] * extra
            perp_y = dir_vec[0] * extra
            px = target.get("x", 0) + dir_vec[0] * offset + perp_x
            py = target.get("y", 0) + dir_vec[1] * offset + perp_y

            ok = True
            if room_bounds and len(room_bounds) >= 4:
                if (
                    px < room_bounds[0]
                    or px > room_bounds[2]
                    or py < room_bounds[1]
                    or py > room_bounds[3]
                ):
                    ok = False
            if ok:
                for o in all_objects:
                    if o.get("id") in (moveable.get("id"), target.get("id")):
                        continue
                    ohr = max(o.get("width", 0), o.get("length", 0)) / 2
                    if (
                        math.sqrt((px - o.get("x", 0)) ** 2 + (py - o.get("y", 0)) ** 2)
                        < aw / 2 + ohr
                    ):
                        ok = False
                        break
            if ok:
                fits = True
                break

        answer = "Yes" if fits else "No"
        used.add(key)
        result.append(
            {
                "type": "spatial_compatibility",
                "question": pick_template(
                    "spatial_compatibility",
                    rng,
                    moveable=object_label(moveable),
                    dir_name=dir_name,
                    target=object_label(target),
                ),
                "choices": ["Yes", "No"],
                "answer": answer,
                "camera_name": camera_pose["name"],
                "camera_position": list(cam_pos),
                "camera_look_at": list(look_at),
                "object_a": {
                    "label": object_label(moveable),
                    "id": moveable.get("id"),
                    "width": moveable.get("width"),
                    "length": moveable.get("length"),
                },
                "object_b": {
                    "label": object_label(target),
                    "id": target.get("id"),
                    "width": target.get("width"),
                    "length": target.get("length"),
                },
                "direction": dir_name,
            }
        )

    return result
