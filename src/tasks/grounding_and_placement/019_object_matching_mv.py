"""Object-matching multi-view question curator: identify an object's bbox across views."""

from __future__ import annotations

import random

from src.tasks import object_label, _MAX_TRIALS
from src.tasks.templates import pick_template
from src.utils.occlusion import filter_visible_objects, object_centroid
from src.utils.projection import project_world_to_fraction


def _project_bbox_2d(
    obj: dict,
    cam_pos: tuple,
    look_at: tuple,
    fov: float,
) -> tuple[int, int, int, int] | None:
    """Project object's 3D extent to a 2D bbox in 0-1000 image space."""
    cx, cy, cz = obj["centroid"]
    hw = (obj.get("width") or 0.3) / 2
    hl = (obj.get("length") or 0.3) / 2
    hh = (obj.get("height") or 0.3) / 2
    corners = [
        (cx + sx * hw, cy + sy * hl, cz + sz * hh)
        for sx in (-1, 1)
        for sy in (-1, 1)
        for sz in (-1, 1)
    ]
    us, vs = [], []
    for pt in corners:
        uv = project_world_to_fraction(pt, cam_pos, look_at, horizontal_fov_deg=fov)
        if uv is None:
            return None
        us.append(uv[0])
        vs.append(uv[1])
    x_min = max(0, int(min(us) * 1000))
    y_min = max(0, int(min(vs) * 1000))
    x_max = min(1000, int(max(us) * 1000))
    y_max = min(1000, int(max(vs) * 1000))
    if x_max - x_min < 10 or y_max - y_min < 10:
        return None
    return (x_min, y_min, x_max, y_max)


def curate_object_matching_mv_questions(
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
    Given an object marked in image A, identify its bounding box in image B.
    Matches SPAR PosMatch format: MCQ with 4 bbox choices in [x,y,x,y] 0-1000.
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
            if not shared:
                continue
            target = rng.choice(shared)
            key = (ca["name"], cb["name"], target.get("id"))
            if key in used:
                continue

            fov_b = cb.get("horizontal_fov_deg", 82.0)
            cam_b_pos = tuple(cb["position"])
            cam_b_look = tuple(cb["look_at"])

            def _get_bbox(obj, cam_name):
                if seg_mask_dir and instance_id_map:
                    import os, numpy as np
                    seg_path = os.path.join(seg_mask_dir, f"{cam_name}_seg.npy")
                    if os.path.isfile(seg_path):
                        mask = np.load(seg_path)
                        iid = instance_id_map.get(obj.get("id"), -1)
                        match = (mask == iid)
                        if match.sum() < 100:
                            return None
                        ys, xs = np.where(match)
                        H, W = mask.shape
                        x1 = max(0, int(xs.min() / W * 1000))
                        y1 = max(0, int(ys.min() / H * 1000))
                        x2 = min(1000, int(xs.max() / W * 1000))
                        y2 = min(1000, int(ys.max() / H * 1000))
                        if x2 - x1 < 10 or y2 - y1 < 10:
                            return None
                        return (x1, y1, x2, y2)
                return _project_bbox_2d(obj, cam_b_pos, cam_b_look, fov_b)

            bbox = _get_bbox(target, cb["name"])
            if bbox is None:
                continue

            distractors = []
            for o in vis_b:
                if o.get("id") == target.get("id"):
                    continue
                db = _get_bbox(o, cb["name"])
                if db:
                    distractors.append(db)
            if len(distractors) < 3:
                continue

            rng.shuffle(distractors)
            wrong = distractors[:3]
            raw_choices = [list(bbox)] + [list(w) for w in wrong]
            rng.shuffle(raw_choices)
            answer_idx = raw_choices.index(list(bbox))
            answer = chr(ord("A") + answer_idx)
            choices = [f"{chr(ord('A') + i)}. {c}" for i, c in enumerate(raw_choices)]

            used.add(key)
            found = (ca, cb, target, choices, answer)
            break

        if found:
            ca, cb, target, choices, answer = found
            label = object_label(target)
            opts = "\n".join(choices)
            result.append(
                {
                    "type": "object_matching_mv",
                    "question": pick_template(
                        "object_matching_mv", rng, label=label, opts=opts
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
                    "object": {
                        "label": label,
                        "id": target.get("id"),
                        "centroid": list(target["centroid"]),
                        "width": target.get("width"),
                        "length": target.get("length"),
                        "height": target.get("height"),
                    },
                }
            )

    return result
