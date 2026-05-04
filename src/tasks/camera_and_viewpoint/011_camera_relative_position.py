"""Camera-relative-position question curator: 8-direction relation between two cameras."""

from __future__ import annotations

import math
import random

from src.tasks import (
    _MAX_TRIALS, _8DIR_NAMES, _quantize_8dir,
    _relative_position_in_frame,
)
from src.tasks.templates import pick_template


def curate_camera_relative_position_questions(
    all_cameras: list[dict],
    *,
    max_questions: int = 1,
    min_separation: float = 0.2,
    max_separation: float | None = None,
    rng: random.Random,
) -> list[dict]:
    """
    Sample pairs of cameras for 8-direction relative-position QAs.
    Answer indicates where camera B is relative to camera A using A's
    facing direction as front.
    """
    eligible = [c for c in all_cameras if c.get("edge_direction")]
    if len(eligible) < 2:
        return []

    # For marble (max_separation set): build controlled pairs from same base view.
    # Only pair cameras that share the exact same viewing direction:
    #   - edge_left ↔ edge_right  (pure lateral offset)
    #   - view ↔ step_forward     (pure forward offset)
    #   - view ↔ edge_left/right  (pure lateral offset)
    #   - edge ↔ step_forward     (diagonal offset)
    if max_separation is not None:
        view_lookup = {c["name"]: c for c in all_cameras}
        same_view_pairs = []
        seen_bases: set[str] = set()
        for c in eligible:
            base = c["name"].rsplit("_edge_", 1)[0]
            if base in seen_bases:
                continue
            seen_bases.add(base)
            edges = [ec for ec in eligible if ec["name"].startswith(base + "_edge_")]
            steps = [sc for sc in all_cameras
                     if sc.get("step_from") == base and sc.get("step_direction") == "forward"]
            group = list(edges)
            group.extend(steps)
            for ii in range(len(group)):
                for jj in range(ii + 1, len(group)):
                    same_view_pairs.append((group[ii], group[jj]))
        rng.shuffle(same_view_pairs)
        pair_pool = same_view_pairs
    else:
        pair_pool = None

    indices = list(range(len(eligible)))
    result: list[dict] = []
    used_pairs: set[tuple[str, str]] = set()

    for _ in range(max_questions):
        found = None
        for _ in range(_MAX_TRIALS):
            if pair_pool is not None:
                if not pair_pool:
                    break
                idx = rng.randrange(len(pair_pool))
                cam_a, cam_b = pair_pool[idx]
                if rng.random() < 0.5:
                    cam_a, cam_b = cam_b, cam_a
            else:
                i, j = rng.sample(indices, 2)
                cam_a, cam_b = eligible[i], eligible[j]

            pair_key = (cam_a["name"], cam_b["name"])
            if pair_key in used_pairs:
                continue
            if cam_a.get("room_name") != cam_b.get("room_name"):
                continue
            az = cam_a["position"][2]
            bz = cam_b["position"][2]
            if abs(az - bz) > 0.1:
                continue

            pos = _relative_position_in_frame(cam_a, cam_b)
            if pos is None:
                continue
            t_fwd, t_right, _ = pos
            dist = math.hypot(t_fwd, t_right)
            if dist < min_separation or (
                max_separation is not None and dist > max_separation
            ):
                continue

            answer = _quantize_8dir(t_fwd, t_right)
            found = (cam_a, cam_b, answer)
            used_pairs.add(pair_key)
            break

        if found:
            cam_a, cam_b, answer = found
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
                    "type": "camera_relative_position",
                    "question": pick_template("camera_relative_position", rng),
                    "choices": choices,
                    "answer": answer,
                    "camera_a": {
                        "name": cam_a["name"],
                        "position": list(cam_a["position"]),
                        "look_at": list(cam_a["look_at"]),
                    },
                    "camera_b": {
                        "name": cam_b["name"],
                        "position": list(cam_b["position"]),
                        "look_at": list(cam_b["look_at"]),
                    },
                }
            )

    return result
