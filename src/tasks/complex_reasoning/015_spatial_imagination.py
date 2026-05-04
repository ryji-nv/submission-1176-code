"""Spatial-imagination question curator: describe spatial relations after hypothetical movement."""

from __future__ import annotations

import itertools
import math
import random
from typing import Any

import os
import numpy as np

from src.tasks import (
    object_label,
    _object_meta,
    _MAX_TRIALS,
    _build_imagined_pose,
    _relation_oc,
)
from src.tasks.templates import pick_template

_si_vis_cache: dict[str, dict[str, set[str]]] = {}


def _get_si_visibility(seg_mask_dir: str, instance_id_map: dict) -> dict[str, set[str]]:
    """Precompute per-camera visible object sets from seg masks (cached, GPU-batched)."""
    if seg_mask_dir in _si_vis_cache:
        return _si_vis_cache[seg_mask_dir]

    seg_files = sorted(f for f in os.listdir(seg_mask_dir) if f.endswith("_seg.npy"))
    if not seg_files:
        _si_vis_cache[seg_mask_dir] = {}
        return {}

    cam_names = [f.replace("_seg.npy", "") for f in seg_files]
    oid_list = list(instance_id_map.keys())
    iid_list = list(instance_id_map.values())

    try:
        import torch
        masks = np.stack([np.load(os.path.join(seg_mask_dir, f)) for f in seg_files])
        masks_t = torch.from_numpy(masks).to(device="cuda", dtype=torch.int32)
        iids_t = torch.tensor(iid_list, device="cuda", dtype=torch.int32)
        counts = (masks_t.unsqueeze(1) == iids_t.view(1, -1, 1, 1)).sum(dim=(2, 3))
        result = {}
        counts_cpu = counts.cpu().numpy()
        for ci, name in enumerate(cam_names):
            result[name] = {oid_list[j] for j in range(len(oid_list)) if counts_cpu[ci, j] >= 200}
        del masks_t, iids_t, counts
        torch.cuda.empty_cache()
    except Exception:
        result = {}
        for f, name in zip(seg_files, cam_names):
            mask = np.load(os.path.join(seg_mask_dir, f))
            vis = set()
            for oid, iid in instance_id_map.items():
                if int((mask == iid).sum()) >= 200:
                    vis.add(oid)
            result[name] = vis

    _si_vis_cache[seg_mask_dir] = result
    return result


def _relation_oo(
    pose: tuple,
    obj_c: dict,
    obj_d: dict,
    threshold: float = 0.3,
) -> tuple[str, str, str] | None:
    """
    Compute relative position of obj_c w.r.t. obj_d from the imagined
    observer's frame: (left/right, above/below, farther/closer).
    """
    pos, fwd, right, up = pose

    def _project(o):
        dx = o["centroid"][0] - pos[0]
        dy = o["centroid"][1] - pos[1]
        dz = o["centroid"][2] - pos[2]
        return (
            dx * right[0] + dy * right[1] + dz * right[2],
            dx * up[0] + dy * up[1] + dz * up[2],
            dx * fwd[0] + dy * fwd[1] + dz * fwd[2],
        )

    rc, uc, fc = _project(obj_c)
    rd, ud, fd = _project(obj_d)
    dr = rc - rd
    dz = obj_c["centroid"][2] - obj_d["centroid"][2]
    dist_c = math.sqrt(rc * rc + uc * uc + fc * fc)
    dist_d = math.sqrt(rd * rd + ud * ud + fd * fd)
    dd = dist_c - dist_d
    lr = "right" if dr > threshold else ("left" if dr < -threshold else "")
    ab = "above" if dz > threshold else ("below" if dz < -threshold else "")
    fc_label = "farther" if dd > threshold else ("closer" if dd < -threshold else "")
    if not lr or not ab or not fc_label:
        return None
    return (lr, ab, fc_label)


def _format_relation(rel: tuple[str, ...]) -> str:
    return ", ".join(r if r else "" for r in rel)


def _generate_wrong_choices(
    correct: tuple[str, ...],
    axes: list[list[str]],
    rng: random.Random,
    n: int = 3,
) -> list[str]:
    """Generate n wrong MCQ choices by flipping axis values."""
    all_combos = list(itertools.product(*axes))
    wrong = [c for c in all_combos if c != correct]
    rng.shuffle(wrong)
    return [_format_relation(c) for c in wrong[:n]]


_OC_AXES = [["left", "right"], ["above", "below"], ["front", "behind"]]
_OO_AXES = [["left", "right"], ["above", "below"], ["farther", "closer"]]


def curate_spatial_imagination_questions(
    camera_pose: dict,
    visible_objects: list[dict],
    *,
    max_questions: int = 1,
    all_cameras: list[dict] | None = None,
    seg_mask_dir: str | None = None,
    instance_id_map: dict | None = None,
    rng: random.Random,
) -> list[dict]:
    """
    Spatial imagination: after hypothetically moving to object A and facing
    object B, describe the spatial relation of the query object(s).

    Generates a mix of OC (object-camera, 3 objects) and OO (object-object,
    4 objects) sub-variants.  When multiple camera views are available,
    multi-view variants are included (extra images for context).
    """
    if len(visible_objects) < 3:
        return []

    cam_height = camera_pose["position"][2]
    can_oo = len(visible_objects) >= 4
    mv_cameras: list[dict] = []
    if all_cameras:
        candidates = [
            c
            for c in all_cameras
            if not c.get("step_direction")
            and not c.get("edge_direction")
            and c["name"] != camera_pose["name"]
        ]
        if seg_mask_dir and instance_id_map:
            import os
            cam_vis_cache = _get_si_visibility(seg_mask_dir, instance_id_map)
            main_vis = cam_vis_cache.get(camera_pose["name"], set())
            for c in candidates:
                ctx_vis = cam_vis_cache.get(c["name"], set())
                if len(main_vis & ctx_vis) >= 2:
                    mv_cameras.append(c)
        else:
            mv_cameras = candidates

    indices = list(range(len(visible_objects)))
    result: list[dict] = []
    used: set[tuple] = set()

    for _ in range(max_questions):
        found = None
        for _ in range(_MAX_TRIALS):
            do_oo = can_oo and rng.random() < 0.5
            n_pick = 4 if do_oo else 3
            pick = tuple(rng.sample(indices, n_pick))
            if pick in used:
                continue
            objs = [visible_objects[k] for k in pick]
            labels = [object_label(o) for o in objs]
            if len(set(labels)) < len(labels):
                continue
            move_to, face_toward, obj_red = objs[0], objs[1], objs[2]
            pose = _build_imagined_pose(move_to, face_toward, cam_height)
            if pose is None:
                continue
            if do_oo:
                obj_yellow = objs[3]
                rel = _relation_oo(pose, obj_red, obj_yellow)
                if rel is None:
                    continue
            else:
                rel = _relation_oc(pose, obj_red)
                if rel is None:
                    continue
            used.add(pick)
            found = (do_oo, objs, rel, pose)
            break

        if not found:
            continue

        do_oo, objs, rel, pose = found
        move_to, face_toward, obj_red = objs[0], objs[1], objs[2]
        is_mv = bool(mv_cameras) and rng.random() < 0.5

        correct = _format_relation(rel)
        axes = _OO_AXES if do_oo else _OC_AXES
        wrong = _generate_wrong_choices(rel, axes, rng)
        choices = [correct] + wrong
        rng.shuffle(choices)

        green_label = object_label(move_to)
        blue_label = object_label(face_toward)
        red_label = object_label(obj_red)

        if do_oo:
            obj_yellow = objs[3]
            yellow_label = object_label(obj_yellow)
            question = pick_template(
                "spatial_imagination_oo",
                rng,
                red_label=red_label,
                yellow_label=yellow_label,
                green_label=green_label,
                blue_label=blue_label,
            )
            sub_type = "oo_mv" if is_mv else "oo"
        else:
            question = pick_template(
                "spatial_imagination_oc",
                rng,
                red_label=red_label,
                green_label=green_label,
                blue_label=blue_label,
            )
            sub_type = "oc_mv" if is_mv else "oc"

        q: dict[str, Any] = {
            "type": "spatial_imagination",
            "sub_type": sub_type,
            "question": question,
            "choices": choices,
            "answer": correct,
            "camera_name": camera_pose["name"],
            "camera_position": list(camera_pose["position"]),
            "camera_look_at": list(camera_pose["look_at"]),
            "imagined_position": list(pose[0]),
            "imagined_forward": list(pose[1]),
            "move_to": _object_meta(move_to, "green"),
            "face_toward": _object_meta(face_toward, "blue"),
            "query_object": _object_meta(obj_red, "red"),
        }
        if do_oo:
            q["reference_object"] = _object_meta(obj_yellow, "yellow")
        if is_mv:
            ctx = rng.choice(mv_cameras)
            q["context_camera"] = {
                "name": ctx["name"],
                "position": list(ctx["position"]),
                "look_at": list(ctx["look_at"]),
            }
        result.append(q)

    return result
