"""Object-object-position multi-view question curator: spatial relation across views."""

from __future__ import annotations

import math
import os
import random

import numpy as np

from src.tasks import object_label, _MAX_TRIALS
from src.tasks.templates import pick_template
from src.utils.occlusion import filter_visible_objects, object_centroid


def curate_object_object_position_mv_questions(
    all_cameras: list[dict],
    all_objects: list[dict],
    *,
    max_questions: int = 1,
    rng: random.Random,
    visibility_kwargs: dict | None = None,
    cam_visibility: dict | None = None,
    seg_mask_dir: str | None = None,
    instance_id_map: dict | None = None,
) -> list[dict]:
    """
    Two camera views, two objects visible in both.
    Ask about spatial relation between them from camera A's perspective.
    """
    corners = [
        c
        for c in all_cameras
        if not c.get("step_direction") and not c.get("edge_direction")
    ]
    if len(corners) < 2:
        return []

    if cam_visibility:
        obj_by_id = {}
        for o in all_objects:
            enriched = {**o, "centroid": object_centroid(o)}
            obj_by_id[o.get("id")] = enriched
        cam_visible = {
            c["name"]: [
                obj_by_id[oid]
                for oid in cam_visibility.get(c["name"], [])
                if oid in obj_by_id
            ]
            for c in corners
        }
    else:
        vis_kw = visibility_kwargs or {}
        cam_visible: dict[str, list[dict]] = {}
        for c in corners:
            pose = {
                "position": tuple(c["position"]),
                "look_at": tuple(c["look_at"]),
                "horizontal_fov_deg": c.get("horizontal_fov_deg", 82.0),
            }
            cam_visible[c["name"]] = filter_visible_objects(pose, all_objects, **vis_kw)

    _seg_cache: dict[str, np.ndarray] = {}

    def _get_seg(cam_name: str) -> np.ndarray | None:
        if cam_name in _seg_cache:
            return _seg_cache[cam_name]
        if not seg_mask_dir:
            return None
        p = os.path.join(seg_mask_dir, f"{cam_name}_seg.npy")
        if not os.path.isfile(p):
            return None
        m = np.load(p)
        _seg_cache[cam_name] = m
        return m

    def _mask_centroid_2d(seg: np.ndarray, iid: int) -> tuple[float, float] | None:
        """Return (cx, cy) in normalized image coords for an instance, or None."""
        ys, xs = np.where(seg == iid)
        if len(xs) < 10:
            return None
        H, W = seg.shape
        return (float(xs.mean()) / W, float(ys.mean()) / H)

    def _spatial_relation(obj_a, obj_b, cam):
        """Position of obj_a relative to obj_b from camera's viewpoint.

        Uses visible seg mask pixels for left/right and above/below so the
        answer matches what is visually apparent in the image.  Close/far
        uses 3D Euclidean distance.
        """
        seg = _get_seg(cam["name"])
        id_a = obj_a.get("id")
        id_b = obj_b.get("id")
        iid_a = instance_id_map.get(id_a) if instance_id_map and id_a else None
        iid_b = instance_id_map.get(id_b) if instance_id_map and id_b else None

        lr = None
        ud = None
        _t_px = 0.03

        if seg is not None and iid_a is not None and iid_b is not None:
            ca_2d = _mask_centroid_2d(seg, iid_a)
            cb_2d = _mask_centroid_2d(seg, iid_b)
            if ca_2d is not None and cb_2d is not None:
                dx_img = ca_2d[0] - cb_2d[0]
                dy_img = ca_2d[1] - cb_2d[1]
                if abs(dx_img) < _t_px and abs(dy_img) < _t_px:
                    return None
                lr = "right" if dx_img > _t_px else ("left" if dx_img < -_t_px else None)
                ud = "above" if dy_img < -_t_px else ("below" if dy_img > _t_px else None)

        if lr is None or ud is None:
            return None

        ca_3d = obj_a["centroid"]
        cb_3d = obj_b["centroid"]
        dx = ca_3d[0] - cb_3d[0]
        dy = ca_3d[1] - cb_3d[1]
        dz = ca_3d[2] - cb_3d[2]
        dist_3d = math.sqrt(dx * dx + dy * dy + dz * dz)
        if dist_3d > 2.0:
            cf = "far"
        elif dist_3d < 1.0:
            cf = "close"
        else:
            cf = None

        if cf:
            return f"{lr}, {ud}, {cf}"
        return f"{lr}, {ud}"

    result: list[dict] = []
    used: set[tuple] = set()

    for _ in range(max_questions):
        found = None
        for _ in range(_MAX_TRIALS):
            ca, cb = rng.sample(corners, 2)
            if ca.get("room_name") != cb.get("room_name"):
                continue
            vis_a = cam_visible[ca["name"]]
            vis_b = cam_visible[cb["name"]]
            ids_b = {o.get("id") for o in vis_b}
            shared = [o for o in vis_a if o.get("id") in ids_b]
            if len(shared) < 2:
                continue
            obj_a, obj_b = rng.sample(shared, 2)
            key = (ca["name"], cb["name"], obj_a.get("id"), obj_b.get("id"))
            if key in used:
                continue
            answer = _spatial_relation(obj_a, obj_b, ca)
            if answer is None:
                continue
            used.add(key)
            found = (ca, cb, obj_a, obj_b, answer)
            break

        if found:
            ca, cb, obj_a, obj_b, answer = found
            has_cf = ", " in answer and answer.count(", ") == 2
            if has_cf:
                all_combos = [
                    f"{a}, {b}, {c}"
                    for a in ("left", "right")
                    for b in ("above", "below")
                    for c in ("close", "far")
                ]
            else:
                all_combos = [
                    f"{a}, {b}"
                    for a in ("left", "right")
                    for b in ("above", "below")
                ]
            wrong = [c for c in all_combos if c != answer]
            rng.shuffle(wrong)
            choices = [answer] + wrong[:3]
            rng.shuffle(choices)
            result.append(
                {
                    "type": "object_object_position_mv",
                    "question": pick_template(
                        "object_object_position_mv",
                        rng,
                        label_a=object_label(obj_a),
                        label_b=object_label(obj_b),
                    ),
                    "choices": choices,
                    "answer": answer,
                    "camera_a": {
                        "name": ca["name"],
                        "position": list(ca["position"]),
                        "look_at": list(ca["look_at"]),
                    },
                    "camera_b": {
                        "name": cb["name"],
                        "position": list(cb["position"]),
                        "look_at": list(cb["look_at"]),
                    },
                    "blue": {
                        "label": object_label(obj_a),
                        "id": obj_a.get("id"),
                        "centroid": list(obj_a["centroid"]),
                        "width": obj_a.get("width"),
                        "length": obj_a.get("length"),
                        "height": obj_a.get("height"),
                    },
                    "red": {
                        "label": object_label(obj_b),
                        "id": obj_b.get("id"),
                        "centroid": list(obj_b["centroid"]),
                        "width": obj_b.get("width"),
                        "length": obj_b.get("length"),
                        "height": obj_b.get("height"),
                    },
                }
            )

    return result
