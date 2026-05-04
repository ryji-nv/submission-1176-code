"""Relative-direction question curator: spatial relation between two objects from camera's view."""

from __future__ import annotations

import math
import random

from src.tasks import object_label, _object_meta, _MAX_TRIALS
from src.tasks.templates import pick_template
from src.utils.projection import project_world_to_fraction


def curate_relative_direction_questions(
    camera_pose: dict,
    visible_objects: list[dict],
    *,
    max_questions: int = 1,
    min_separation: float = 0.10,
    rng: random.Random,
) -> list[dict]:
    """
    Randomly sample pairs of visible objects for relative-direction QAs.

    Direction is judged from the camera's perspective using image-space u
    coordinates (u=0 left, u=1 right). Pairs whose horizontal separation is
    less than min_separation (as a fraction of image width) are skipped.
    """
    if len(visible_objects) < 2:
        return []

    fov = camera_pose.get("horizontal_fov_deg", 82.0)
    cam_pos = tuple(camera_pose["position"])
    look_at = tuple(camera_pose["look_at"])

    # Camera frame vectors for 3D relation computation
    _fx = look_at[0] - cam_pos[0]
    _fy = look_at[1] - cam_pos[1]
    _flen = math.sqrt(_fx * _fx + _fy * _fy)
    if _flen < 1e-9:
        return []
    fx, fy = _fx / _flen, _fy / _flen
    rx, ry = fy, -fx

    # Pre-project all objects into image-fraction space
    projected: list[tuple[dict, tuple[float, float]]] = []
    for obj in visible_objects:
        uv = project_world_to_fraction(
            obj["centroid"], cam_pos, look_at, horizontal_fov_deg=fov
        )
        if uv is not None:
            projected.append((obj, uv))

    if len(projected) < 2:
        return []

    indices = list(range(len(projected)))
    result: list[dict] = []
    used_pairs: set[tuple[int, int]] = set()

    for _ in range(max_questions):
        found = None
        for _ in range(_MAX_TRIALS):
            i, j = rng.sample(indices, 2)
            pair = (min(i, j), max(i, j))
            if pair in used_pairs:
                continue
            obj_a, (ua, va) = projected[i]
            obj_b, (ub, vb) = projected[j]

            # Compute in 3D camera frame (SPAR convention)
            ca = obj_a["centroid"]
            cb = obj_b["centroid"]
            # Camera frame vectors
            dx = ca[0] - cb[0]
            dy = ca[1] - cb[1]
            dz = ca[2] - cb[2]
            # Project onto camera axes
            dot_r = dx * rx + dy * ry  # right axis
            dot_f = dx * fx + dy * fy  # forward axis (depth)
            dot_z = dz  # world up = above/below

            _t = 0.1  # SPAR threshold: 0.1m
            if abs(dot_r) < _t and abs(dot_z) < _t and abs(dot_f) < _t:
                continue

            lr = "right" if dot_r > _t else ("left" if dot_r < -_t else "")
            ud = "above" if dot_z > _t else ("below" if dot_z < -_t else "")
            cf = "closer" if dot_f < -_t else ("farther" if dot_f > _t else "")
            answer = f"{lr}, {ud}, {cf}"

            found = (obj_a, obj_b, answer)
            used_pairs.add(pair)
            break

        if found:
            obj_a, obj_b, answer = found
            _rel_axes = [
                ["left", "right", ""],
                ["above", "below", ""],
                ["closer", "farther", ""],
            ]
            all_combos = [
                f"{a}, {b}, {c}"
                for a in _rel_axes[0]
                for b in _rel_axes[1]
                for c in _rel_axes[2]
            ]
            wrong = [c for c in all_combos if c != answer]
            rng.shuffle(wrong)
            choices = [answer] + wrong[:3]
            rng.shuffle(choices)
            result.append(
                {
                    "type": "relative_direction",
                    "question": pick_template(
                        "relative_direction",
                        rng,
                        label_a=object_label(obj_a),
                        label_b=object_label(obj_b),
                    ),
                    "choices": choices,
                    "answer": answer,
                    "camera_name": camera_pose["name"],
                    "camera_position": list(camera_pose["position"]),
                    "camera_look_at": list(camera_pose["look_at"]),
                    "blue": _object_meta(obj_a, "blue"),
                    "red": _object_meta(obj_b, "red"),
                }
            )

    return result
