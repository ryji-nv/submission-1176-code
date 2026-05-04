"""Viewpoint-change question curator: predict 5-DOF camera transformation between views."""

from __future__ import annotations

import math
import random

from src.tasks import _MAX_TRIALS, _relative_transform_5dof
from src.tasks.templates import pick_template


def _format_viewpoint_change(t: dict) -> str:
    """Format 5-DOF transform as SPAR ViewChgI answer string."""

    def _dir(val, pos, neg):
        return (pos, abs(val)) if val >= 0 else (neg, abs(val))

    lr_d, lr_v = _dir(t["right"], "right", "left")
    ud_d, ud_v = _dir(t["up"], "up", "down")
    fb_d, fb_v = _dir(t["forward"], "forward", "backward")
    pu_d, pu_v = _dir(t["pitch"], "up", "down")
    rl_d, rl_v = _dir(t["yaw"], "right", "left")
    return (
        f"move_{lr_d}:{lr_v:.1f},move_{ud_d}:{ud_v:.1f},"
        f"move_{fb_d}:{fb_v:.1f},rotate_{pu_d}:{pu_v:.0f},"
        f"rotate_{rl_d}:{rl_v:.0f}"
    )


def curate_viewpoint_change_questions(
    all_cameras: list[dict],
    *,
    max_questions: int = 1,
    min_separation: float = 0.2,
    max_separation: float | None = None,
    seg_mask_dir: str | None = None,
    instance_id_map: dict | None = None,
    min_shared_objects: int = 2,
    rng: random.Random,
) -> list[dict]:
    """
    Predict the 5-DOF camera transformation (translation + rotation)
    from view A to view B.  Matches SPAR ViewChgI format.
    """
    eligible = [c for c in all_cameras if not c.get("step_direction")]
    if len(eligible) < 2:
        return []

    # For marble: build pairs from same base view, filtered by shared objects
    if max_separation is not None:
        from collections import defaultdict
        base_groups: defaultdict[str, list] = defaultdict(list)
        for c in all_cameras:
            if c.get("edge_direction"):
                base = c["name"].rsplit("_edge_", 1)[0]
                base_groups[base].append(c)
            elif c.get("step_direction"):
                base = c.get("step_from", "")
                if base:
                    base_groups[base].append(c)

        # Compute per-camera visible objects from seg masks (25-75% height)
        _cam_objects: dict[str, set[str]] = {}
        if seg_mask_dir and instance_id_map:
            import os
            import numpy as np
            all_step_edge = [c for g in base_groups.values() for c in g]
            for cam in all_step_edge:
                seg_path = os.path.join(seg_mask_dir, f"{cam['name']}_seg.npy")
                if not os.path.isfile(seg_path):
                    continue
                mask = np.load(seg_path)
                H = mask.shape[0]
                vis = set()
                for oid, iid in instance_id_map.items():
                    match = (mask == iid)
                    if int(match.sum()) < 200:
                        continue
                    ys = np.where(match.any(axis=1))[0]
                    cy = float(ys.mean()) / H
                    if 0.25 <= cy <= 0.75:
                        vis.add(oid)
                _cam_objects[cam["name"]] = vis

        pair_pool = []
        for base, cams in base_groups.items():
            if len(cams) < 2:
                continue
            for ii in range(len(cams)):
                for jj in range(ii + 1, len(cams)):
                    if _cam_objects:
                        va = _cam_objects.get(cams[ii]["name"], set())
                        vb = _cam_objects.get(cams[jj]["name"], set())
                        if len(va & vb) < min_shared_objects:
                            continue
                    pair_pool.append((cams[ii], cams[jj]))
        # Interleave pair types for diversity
        def _pair_type(a, b):
            na, nb = a["name"], b["name"]
            for tag in ("pitch_", "up", "down"):
                if tag in na or tag in nb:
                    return "vertical"
            for tag in ("yaw_",):
                if tag in na or tag in nb:
                    return "rotation"
            for tag in ("forward",):
                if tag in na.split("step_")[-1] or tag in nb.split("step_")[-1]:
                    return "forward"
            return "lateral"
        buckets: dict[str, list] = {}
        for p in pair_pool:
            t = _pair_type(p[0], p[1])
            buckets.setdefault(t, []).append(p)
        for v in buckets.values():
            rng.shuffle(v)
        interleaved = []
        while any(buckets.values()):
            for k in list(buckets.keys()):
                if buckets[k]:
                    interleaved.append(buckets[k].pop(0))
                else:
                    del buckets[k]
        pair_pool = interleaved
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
                pair_key = tuple(sorted([cam_a["name"], cam_b["name"]]))
                if pair_key in used_pairs:
                    pair_pool.pop(idx)
                    continue
                if rng.random() < 0.5:
                    cam_a, cam_b = cam_b, cam_a
            else:
                i, j = rng.sample(indices, 2)
                cam_a, cam_b = eligible[i], eligible[j]
                pair_key = tuple(sorted([cam_a["name"], cam_b["name"]]))
                if pair_key in used_pairs:
                    continue
                if cam_a.get("room_name") != cam_b.get("room_name"):
                    continue

            t = _relative_transform_5dof(cam_a, cam_b)
            if t is None:
                if pair_pool is not None:
                    pair_pool.pop(idx)
                continue
            dist = math.sqrt(t["right"] ** 2 + t["up"] ** 2 + t["forward"] ** 2)
            has_rotation = abs(t["yaw"]) > 5 or abs(t["pitch"]) > 5
            if not has_rotation and dist < min_separation:
                if pair_pool is not None:
                    pair_pool.pop(idx)
                continue
            if max_separation is not None and dist > max_separation:
                if pair_pool is not None:
                    pair_pool.pop(idx)
                continue
            found = (cam_a, cam_b, t)
            used_pairs.add(pair_key)
            if pair_pool is not None:
                pair_pool.pop(idx)
            break

        if found:
            cam_a, cam_b, t = found
            result.append(
                {
                    "type": "viewpoint_change",
                    "question": (
                        pick_template("viewpoint_change", rng)
                        + "\nProvide the camera movement and rotation in the following format:\n"
                        "move_<right_or_left>:<meters>,move_<down_or_up>:<meters>,"
                        "move_<forward_or_back>:<meters>,"
                        "rotate_<down_or_up>:<degrees>,rotate_<right_or_left>:<degrees>\n"
                        "- The first three values are in meters.\n"
                        "- The last two values are in degrees.\n"
                        "- Use commas to separate each parameter.\n"
                        "- Do not include any additional text.\n"
                        "Example:move_left:2.6,move_down:0.1,move_forward:0.2,"
                        "rotate_up:10,rotate_left:0"
                    ),
                    "answer": _format_viewpoint_change(t),
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
