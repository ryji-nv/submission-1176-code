"""Compound-spatial-referring question curator: multi-step spatial reasoning."""

from __future__ import annotations

import math
import random
from typing import Any

from src.tasks import (
    _normalize_type,
    _clean_label,
    object_label,
    _object_meta,
    _MAX_TRIALS,
    _build_imagined_pose,
    _relation_oc,
)
from src.tasks.templates import pick_template
from src.utils.projection import project_world_to_fraction


_SPATIAL_RELATIONS = [
    ("closest to", lambda d: d, True),
    ("farthest from", lambda d: -d, True),
    ("to the left of", lambda u: u, False),
    ("to the right of", lambda u: -u, False),
]


def _pick_by_relation(
    targets: list[tuple[dict, tuple[float, float]]],
    anchor: dict,
    relation_name: str,
    key_fn,
    is_distance: bool,
    cam_pos: tuple,
    look_at: tuple,
    fov: float,
) -> tuple[dict, tuple[float, float]] | None:
    """Pick the target that uniquely matches the spatial relation to anchor."""
    if is_distance:
        ac = anchor["centroid"]
        scored = []
        for t, uv in targets:
            tc = t["centroid"]
            d = math.sqrt(sum((a - b) ** 2 for a, b in zip(tc, ac)))
            scored.append((key_fn(d), t, uv))
    else:
        anchor_uv = project_world_to_fraction(
            anchor["centroid"], cam_pos, look_at, horizontal_fov_deg=fov
        )
        if anchor_uv is None:
            return None
        scored = []
        for t, uv in targets:
            diff = uv[0] - anchor_uv[0]
            scored.append((key_fn(diff), t, uv))
        # For directional relations, require exactly one target on that side
        on_side = [s for s in scored if s[0] < 0]
        if len(on_side) != 1:
            return None
    if not scored:
        return None
    scored.sort(key=lambda x: x[0])
    return scored[0][1], scored[0][2]


def curate_compound_spatial_referring_questions(
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
    Compound spatial referring with two sub-types:
      - 'referring': point to an object using a chain of spatial anchors
        (RefSpatial compound format, coordinate answer)
      - 'multi_step': multi-image direction reasoning
        (MMSI MSR-inspired, MCQ answer)
    """
    fov = camera_pose.get("horizontal_fov_deg", 82.0)
    cam_pos = tuple(camera_pose["position"])
    look_at = tuple(camera_pose["look_at"])
    cam_height = camera_pose["position"][2]

    # Group visible objects by normalized type, project to image coords
    type_groups: dict[str, list[tuple[dict, tuple[float, float]]]] = {}
    all_projected: list[tuple[dict, tuple[float, float]]] = []
    for obj in visible_objects:
        t = _normalize_type(obj.get("type", ""))
        if not t:
            continue
        uv = project_world_to_fraction(
            obj["centroid"], cam_pos, look_at, horizontal_fov_deg=fov
        )
        if uv is not None:
            type_groups.setdefault(t, []).append((obj, uv))
            all_projected.append((obj, uv))

    # Candidates for 'referring': types with 2+ instances (need disambiguation)
    referring_types = [(t, objs) for t, objs in type_groups.items() if len(objs) >= 2]
    # Anchors: types with exactly 1 instance (unambiguous reference)
    anchor_pool = [objs[0][0] for t, objs in type_groups.items() if len(objs) == 1]

    # Multi-step candidates: need 3+ visible objects total
    can_msr = len(visible_objects) >= 3
    mv_cameras: list[dict] = []
    _cam_vis: dict[str, set[str]] = {}
    if all_cameras and can_msr:
        candidates = [
            c
            for c in all_cameras
            if not c.get("step_direction")
            and not c.get("edge_direction")
            and c["name"] != camera_pose["name"]
        ]
        if seg_mask_dir and instance_id_map:
            import os
            import numpy as np
            _min_px = 200
            for c in candidates:
                seg_path = os.path.join(seg_mask_dir, f"{c['name']}_seg.npy")
                if not os.path.isfile(seg_path):
                    continue
                mask = np.load(seg_path)
                vis = set()
                for oid, iid in instance_id_map.items():
                    if int((mask == iid).sum()) >= _min_px:
                        vis.add(oid)
                if vis:
                    _cam_vis[c["name"]] = vis
                    mv_cameras.append(c)
        else:
            mv_cameras = candidates

    result: list[dict] = []
    used: set = set()

    for _ in range(max_questions):
        # Decide sub-type: try referring first, fall back to multi_step
        do_referring = (
            bool(referring_types) and bool(anchor_pool) and rng.random() < 0.6
        )
        if do_referring:
            rng.shuffle(referring_types)
            found = None
            for t, targets in referring_types:
                rng.shuffle(anchor_pool)
                for anchor in anchor_pool:
                    rel_name, key_fn, is_dist = rng.choice(_SPATIAL_RELATIONS)
                    pick = _pick_by_relation(
                        targets,
                        anchor,
                        rel_name,
                        key_fn,
                        is_dist,
                        cam_pos,
                        look_at,
                        fov,
                    )
                    if pick is None:
                        continue
                    obj, (u, v) = pick
                    desc_key = (obj.get("id"), anchor.get("id"), rel_name)
                    if desc_key in used:
                        continue
                    used.add(desc_key)
                    anchor_label = object_label(anchor)
                    target_label = _clean_label(obj.get("type", ""))
                    found = (obj, u, v, target_label, rel_name, anchor_label)
                    break
                if found:
                    break
            if found:
                obj, u, v, target_label, rel_name, anchor_label = found
                result.append(
                    {
                        "type": "compound_spatial_referring",
                        "sub_type": "referring",
                        "question": pick_template(
                            "compound_spatial_referring",
                            rng,
                            target_label=target_label,
                            rel_name=rel_name,
                            anchor_label=anchor_label,
                        ),
                        "answer": f"[({round(u, 3)}, {round(v, 3)})]",
                        "camera_name": camera_pose["name"],
                        "camera_position": list(cam_pos),
                        "camera_look_at": list(look_at),
                        "object": {
                            "label": object_label(obj),
                            "id": obj.get("id"),
                            "centroid": list(obj["centroid"]),
                            "image_u": round(u, 3),
                            "image_v": round(v, 3),
                        },
                    }
                )
                continue

        # Multi-step sub-type
        if can_msr and len(visible_objects) >= 3:
            indices = list(range(len(visible_objects)))
            found_msr = None
            for _ in range(_MAX_TRIALS):
                pick = tuple(rng.sample(indices, 3))
                if pick in used:
                    continue
                move_to, face_toward, query = (
                    visible_objects[pick[0]],
                    visible_objects[pick[1]],
                    visible_objects[pick[2]],
                )
                if (
                    len(
                        {
                            object_label(move_to),
                            object_label(face_toward),
                            object_label(query),
                        }
                    )
                    < 3
                ):
                    continue
                # All 3 objects must project into the main image
                _all_in_frame = True
                for _obj in (move_to, face_toward, query):
                    _uv = project_world_to_fraction(
                        _obj["centroid"], cam_pos, look_at,
                        horizontal_fov_deg=fov,
                    )
                    if _uv is None or not (0.02 < _uv[0] < 0.98 and 0.02 < _uv[1] < 0.98):
                        _all_in_frame = False
                        break
                if not _all_in_frame:
                    continue
                pose = _build_imagined_pose(move_to, face_toward, cam_height)
                if pose is None:
                    continue
                rel = _relation_oc(pose, query)
                if rel is None:
                    continue
                used.add(pick)
                found_msr = (move_to, face_toward, query, rel, pose)
                break
            if found_msr:
                move_to, face_toward, query, rel, pose = found_msr
                lr, ud, fb = rel
                _DIR_LABELS = {
                    ("left", "front"): "To my front-left",
                    ("right", "front"): "To my front-right",
                    ("left", "behind"): "To my rear-left",
                    ("right", "behind"): "To my rear-right",
                }
                answer = _DIR_LABELS.get((lr, fb), f"To my {fb}-{lr}")
                wrong = [v for v in _DIR_LABELS.values() if v != answer]
                rng.shuffle(wrong)
                choices = [answer] + wrong[:3]
                rng.shuffle(choices)

                q: dict[str, Any] = {
                    "type": "compound_spatial_referring",
                    "sub_type": "multi_step",
                    "question": pick_template(
                        "compound_spatial_multi_step",
                        rng,
                        move_to=object_label(move_to),
                        face_toward=object_label(face_toward),
                        query=object_label(query),
                    ),
                    "choices": choices,
                    "answer": answer,
                    "camera_name": camera_pose["name"],
                    "camera_position": list(cam_pos),
                    "camera_look_at": list(look_at),
                    "move_to": _object_meta(move_to, "green"),
                    "face_toward": _object_meta(face_toward, "blue"),
                    "query_object": _object_meta(query, "red"),
                }
                if mv_cameras:
                    obj_ids = {
                        move_to.get("id"),
                        face_toward.get("id"),
                        query.get("id"),
                    } - {None}
                    if _cam_vis:
                        relevant = [
                            c for c in mv_cameras
                            if len(obj_ids & _cam_vis.get(c["name"], set())) >= 2
                        ]
                    else:
                        relevant = mv_cameras
                    if relevant:
                        ctx = rng.choice(relevant)
                        q["context_camera"] = {
                            "name": ctx["name"],
                            "position": list(ctx["position"]),
                            "look_at": list(ctx["look_at"]),
                        }
                result.append(q)

    return result
