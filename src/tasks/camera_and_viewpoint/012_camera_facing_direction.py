"""Camera-facing-direction question curator: which cardinal direction camera B faces."""

from __future__ import annotations

import random

from src.tasks import _MAX_TRIALS, _relative_heading_deg
from src.tasks.templates import pick_template


_CARDINAL_DIRS = [
    ("North", 0),
    ("East", 90),
    ("South", 180),
    ("West", -90),
]
_CARDINAL_NAMES = [name for name, _ in _CARDINAL_DIRS]


def _quantize_cardinal(angle_deg: float, tolerance: float = 25.0) -> int | None:
    """Map angle to nearest cardinal index (0=N,1=E,2=S,3=W), or None if outside tolerance."""
    for idx, (_, target) in enumerate(_CARDINAL_DIRS):
        diff = (angle_deg - target + 180) % 360 - 180
        if abs(diff) <= tolerance:
            return idx
    return None


def curate_camera_facing_direction_questions(
    all_cameras: list[dict],
    *,
    max_questions: int = 1,
    max_separation: float | None = None,
    seg_mask_dir: str | None = None,
    instance_id_map: dict | None = None,
    min_shared_objects: int = 3,
    rng: random.Random,
) -> list[dict]:
    """
    Ask which direction camera B is facing, given camera A faces a random
    cardinal direction. Uses corner + edge cameras (not stepped).
    The relative facing angle is quantized to the nearest cardinal, then
    rotated by a random reference offset so A's stated direction varies.
    """
    if max_separation is not None:
        eligible = [c for c in all_cameras
                    if c.get("edge_direction") or c["name"].startswith("rand_")]
    else:
        eligible = [c for c in all_cameras if c.get("edge_direction")]
    if len(eligible) < 2:
        return []

    # For marble: precompute per-camera visible object sets from seg masks
    # then find pairs that look at the same region (high object overlap)
    _cam_obj_sets: dict[str, set[int]] = {}
    if max_separation is not None and seg_mask_dir and instance_id_map:
        import os
        import numpy as np
        min_px = 500
        iid_list = list(instance_id_map.values())
        for cam in eligible:
            seg_path = os.path.join(seg_mask_dir, f"{cam['name']}_seg.npy")
            if not os.path.isfile(seg_path):
                continue
            mask = np.load(seg_path)
            vis = set()
            for iid in iid_list:
                if int((mask == iid).sum()) >= min_px:
                    vis.add(iid)
            _cam_obj_sets[cam["name"]] = vis

    # For marble: pre-filter pairs by region overlap (Jaccard >= 0.2)
    if max_separation is not None and _cam_obj_sets:
        min_jaccard = 0.2
        valid_pairs = []
        for i in range(len(eligible)):
            for j in range(i + 1, len(eligible)):
                ca, cb = eligible[i], eligible[j]
                va = _cam_obj_sets.get(ca["name"], set())
                vb = _cam_obj_sets.get(cb["name"], set())
                if not va or not vb:
                    continue
                shared = va & vb
                if len(shared) < min_shared_objects:
                    continue
                jaccard = len(shared) / len(va | vb)
                if jaccard < min_jaccard:
                    continue
                relative = _relative_heading_deg(ca, cb)
                if abs(relative) < 30:
                    continue
                rel_idx = _quantize_cardinal(relative)
                if rel_idx is None:
                    continue
                valid_pairs.append((ca, cb, rel_idx))
        rng.shuffle(valid_pairs)
        pair_pool = valid_pairs
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
                cam_a, cam_b, rel_idx = pair_pool[idx]
                pair_key = tuple(sorted([cam_a["name"], cam_b["name"]]))
                if pair_key in used_pairs:
                    pair_pool.pop(idx)
                    continue
                if rng.random() < 0.5:
                    cam_a, cam_b = cam_b, cam_a
                    rel_idx = (4 - rel_idx) % 4 if rel_idx != 0 else 0
                found = (cam_a, cam_b, rel_idx)
                used_pairs.add(pair_key)
                pair_pool.pop(idx)
                break
            else:
                i, j = rng.sample(indices, 2)
                cam_a, cam_b = eligible[i], eligible[j]
                pair_key = (cam_a["name"], cam_b["name"])
                if pair_key in used_pairs:
                    continue
                if cam_a.get("room_name") != cam_b.get("room_name"):
                    continue
                relative = _relative_heading_deg(cam_a, cam_b)
                if not (75 <= abs(relative) <= 105):
                    continue
                rel_idx = _quantize_cardinal(relative)
                if rel_idx is None:
                    continue
                found = (cam_a, cam_b, rel_idx)
                used_pairs.add(pair_key)
                break

        if found:
            cam_a, cam_b, rel_idx = found
            ref_offset = rng.randint(0, 3)
            ref_dir = _CARDINAL_NAMES[ref_offset]
            answer = _CARDINAL_NAMES[(ref_offset + rel_idx) % 4]
            result.append(
                {
                    "type": "camera_facing_direction",
                    "question": pick_template(
                        "camera_facing_direction", rng, ref_dir=ref_dir
                    ),
                    "choices": _CARDINAL_NAMES,
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
