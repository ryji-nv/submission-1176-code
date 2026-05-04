"""Ordinal-grounding question curator: point to the Nth object from left/right."""

from __future__ import annotations

import random

from src.tasks import _normalize_type, _clean_label
from src.utils.projection import project_world_to_fraction

_ORDINALS = ["first", "second", "third", "fourth", "fifth"]


def curate_ordinal_grounding_questions(
    camera_pose: dict,
    visible_objects: list[dict],
    *,
    max_questions: int = 1,
    min_u_gap: float = 0.08,
    scene_type: str = "sage",
    rng: random.Random,
) -> list[dict]:
    """
    'Point to the second chair from the left.'
    Requires multi-instance type with horizontal spread in image space.
    """
    if len(visible_objects) < 2:
        return []

    fov = camera_pose.get("horizontal_fov_deg", 82.0)
    cam_pos = tuple(camera_pose["position"])
    look_at = tuple(camera_pose["look_at"])
    if scene_type == "marble":
        look_at = tuple(2 * p - l for p, l in zip(cam_pos, look_at))

    # Group by type, project to image space
    # For marble: use seg mask centroid for reliable image coordinates
    _seg_centroids: dict[str, tuple[float, float]] = {}
    if scene_type == "marble":
        import os
        seg_mask_dir = camera_pose.get("_seg_mask_dir")
        instance_id_map = camera_pose.get("_instance_id_map")
        if seg_mask_dir and instance_id_map:
            import numpy as np
            seg_path = os.path.join(seg_mask_dir, f"{camera_pose['name']}_seg.npy")
            if os.path.isfile(seg_path):
                mask = np.load(seg_path)
                H, W = mask.shape
                for oid, iid in instance_id_map.items():
                    ys, xs = np.where(mask == iid)
                    if len(xs) >= 200:
                        _seg_centroids[oid] = (float(xs.mean()) / W, float(ys.mean()) / H)

    type_groups: dict[str, list[tuple[dict, float, float]]] = {}
    for obj in visible_objects:
        t = _normalize_type(obj.get("type", ""))
        if not t:
            continue
        oid = obj.get("id", "")
        if oid in _seg_centroids:
            u, v = _seg_centroids[oid]
        else:
            uv = project_world_to_fraction(
                obj["centroid"], cam_pos, look_at, horizontal_fov_deg=fov
            )
            if uv is None:
                continue
            u, v = uv
        type_groups.setdefault(t, []).append((obj, u, v))

    # Keep groups with >= 2 instances and sufficient horizontal spread
    valid_groups = []
    for t, items in type_groups.items():
        if len(items) < 2:
            continue
        items.sort(key=lambda x: x[1])  # sort by u (left to right)
        # Check all consecutive u-gaps
        ok = True
        for k in range(len(items) - 1):
            if items[k + 1][1] - items[k][1] < min_u_gap:
                ok = False
                break
        if ok:
            valid_groups.append((t, items))

    if not valid_groups:
        return []

    rng.shuffle(valid_groups)
    result = []
    for t, items in valid_groups:
        if len(result) >= max_questions:
            break
        n = len(items)
        if n > len(_ORDINALS):
            n = len(_ORDINALS)
            items = items[:n]

        # Pick direction and ordinal
        from_left = rng.choice([True, False])
        ordinal_idx = rng.randint(0, n - 1)
        if from_left:
            target_obj, u, v = items[ordinal_idx]
            direction = "from the left"
        else:
            target_obj, u, v = items[n - 1 - ordinal_idx]
            direction = "from the right"

        target_label = _clean_label(t)
        ordinal_word = _ORDINALS[ordinal_idx]
        result.append(
            {
                "type": "ordinal_grounding",
                "question": (
                    f"Point to the {ordinal_word} {target_label} {direction}. "
                    "Your answer should be formatted as a list of tuples, "
                    "i.e. [(x1, y1)], where each tuple contains the x and y "
                    "coordinates of a point satisfying the conditions above. "
                    "The coordinates should be between 0 and 1, indicating the "
                    "normalized pixel locations of the points in the image."
                ),
                "answer": f"[({round(u, 3)}, {round(v, 3)})]",
                "camera_name": camera_pose["name"],
                "camera_position": list(cam_pos),
                "camera_look_at": list(look_at),
                "object": {
                    "label": target_label,
                    "id": target_obj.get("id"),
                    "centroid": list(target_obj["centroid"]),
                    "image_u": round(u, 3),
                    "image_v": round(v, 3),
                    "width": target_obj.get("width"),
                    "length": target_obj.get("length"),
                    "height": target_obj.get("height"),
                },
                "ordinal": ordinal_word,
                "direction": direction,
                "group_size": n,
            }
        )

    return result
