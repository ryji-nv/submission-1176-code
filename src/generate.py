"""
Generate physics-understanding questions from a scene JSON.

Output structure:
  {output_root}/{scene_id}/tasks/{idx}_{question_type}/question_0.json
  {output_root}/{scene_id}/tasks/{idx}_{question_type}/question_image_0.jpg

Called via process_scene() from src.sage.pipeline and src.scenesmith.pipeline.
"""

import hashlib
import json
import os
import random as _random
import shutil
import sys

import numpy as np

from src.utils.annotate import write_question_image
from src.tasks import curate_questions
from src.utils.constants import EDGE_CAMERA_TYPES, MULTI_VIEW_TYPES, TASK_DIR_NAME
from src.utils.occlusion import enrich_objects
from src.utils.parallel import run_parallel


def _generate_qtype(qtype: str, ctx: dict) -> bool:
    """Core logic: curate, annotate, and save all questions for one type."""
    scene_id = ctx["scene_id"]
    output_root = ctx["output_root"]
    room_objects = ctx["room_objects"]
    camera_pool = ctx["camera_pool"]
    edge_camera_pool = ctx["edge_camera_pool"]
    room_cameras = ctx["room_cameras"]
    room_bounds = ctx["room_bounds"]
    num_questions = ctx["num_questions"]
    force_rerun = ctx["force_rerun"]
    base_seed = ctx["base_seed"]
    images_dir = ctx["images_dir"]
    layout_path = ctx["layout_path"]
    visibility_kwargs = ctx.get("visibility_kwargs")
    seg_mask_dir = ctx.get("seg_mask_dir")
    instance_id_map = ctx.get("instance_id_map")
    _mask_vis_cache = ctx.get("_mask_vis_cache")

    out_dir = os.path.join(output_root, scene_id, "tasks", TASK_DIR_NAME[qtype])
    if (
        not force_rerun
        and os.path.isdir(out_dir)
        and any(f.endswith(".json") for f in os.listdir(out_dir))
    ):
        print(f"  [skip] {scene_id}/{qtype} (already exists)")
        return True
    if force_rerun and os.path.isdir(out_dir):
        shutil.rmtree(out_dir)

    pool = (
        edge_camera_pool
        if (qtype in EDGE_CAMERA_TYPES and edge_camera_pool)
        else camera_pool
    )

    type_seed = base_seed ^ (int(hashlib.md5(qtype.encode()).hexdigest(), 16) % (2**31))
    cam_rng = _random.Random(type_seed)

    all_questions: list[dict] = []
    seen_keys: set[str] = set()

    if qtype in MULTI_VIEW_TYPES:
        cam = cam_rng.choice(pool)
        camera_pose = {
            "name": cam["name"],
            "position": tuple(cam["position"]),
            "look_at": tuple(cam["look_at"]),
            "horizontal_fov_deg": cam.get("horizontal_fov_deg", 82.0),
        }
        meta_dir = os.path.join(output_root, scene_id, "meta")
        qs = curate_questions(
            camera_pose,
            room_objects,
            qtype,
            max_questions=num_questions,
            all_cameras=room_cameras,
            room_bounds=room_bounds,
            seed=type_seed,
            aux_data_dir=meta_dir,
            visibility_kwargs=visibility_kwargs,
            cam_visibility=ctx.get("cam_visibility"),
            scene_type=ctx.get("scene_type", "sage"),
            seg_mask_dir=seg_mask_dir,
            instance_id_map=instance_id_map,
            mask_vis_cache=_mask_vis_cache,
        )
        for q in qs:
            q["_cam"] = cam
            if "camera_a" in q:
                q["_cam"] = q["camera_a"]
            all_questions.append(q)
    else:
        MAX_PER_ENTITY = 2
        entity_counts: dict = {}
        exclude_ids: set[str] = set()
        attempts = 0
        stale = 0
        while len(all_questions) < num_questions and attempts < num_questions * 8:
            prev_count = len(all_questions)
            cam = cam_rng.choice(pool)
            fov = cam.get("horizontal_fov_deg", 82.0)
            camera_pose = {
                "name": cam["name"],
                "position": tuple(cam["position"]),
                "look_at": tuple(cam["look_at"]),
                "horizontal_fov_deg": fov,
            }
            seed = int(
                hashlib.md5(
                    (scene_id + cam["name"] + str(attempts)).encode()
                ).hexdigest(),
                16,
            ) % (2**31)
            meta_dir = os.path.join(output_root, scene_id, "meta")
            cam_vis = ctx.get("cam_visibility")
            pre_vis = None
            if cam_vis and cam["name"] in cam_vis:
                id_set = set(cam_vis[cam["name"]]) - exclude_ids
                pre_vis = enrich_objects(
                    [o for o in room_objects if o.get("id") in id_set],
                    tuple(camera_pose["position"]),
                )
            qs = curate_questions(
                camera_pose,
                room_objects,
                qtype,
                max_questions=1,
                all_cameras=room_cameras,
                room_bounds=room_bounds,
                seed=seed,
                aux_data_dir=meta_dir,
                visibility_kwargs=visibility_kwargs,
                pre_visible=pre_vis,
                scene_type=ctx.get("scene_type", "sage"),
                seg_mask_dir=seg_mask_dir,
                instance_id_map=instance_id_map,
                mask_vis_cache=_mask_vis_cache,
            )
            for q in qs:
                blue, red = q.get("blue"), q.get("red")
                obj = q.get("object")
                if blue and red:
                    bid = blue["id"] if isinstance(blue, dict) else blue
                    rid = red["id"] if isinstance(red, dict) else red
                    ekey = tuple(sorted([bid, rid]))
                    eids = {bid, rid}
                elif obj:
                    eid = obj["id"] if isinstance(obj, dict) else obj
                    ekey = eid
                    eids = {eid}
                else:
                    ekey = None
                    eids = set()

                if ekey is not None and entity_counts.get(ekey, 0) >= MAX_PER_ENTITY:
                    continue

                key = str(q.get("question", ""))
                if qtype not in ("room_size", "object_placement"):
                    key += str(q.get("answer", ""))
                if q.get("camera_a"):
                    key += q["camera_a"].get("name", "")
                if q.get("camera_b"):
                    key += q["camera_b"].get("name", "")
                if key in seen_keys:
                    if ekey is not None:
                        entity_counts[ekey] = entity_counts.get(ekey, 0) + 1
                        if entity_counts[ekey] >= MAX_PER_ENTITY:
                            exclude_ids.update(eids)
                    continue
                seen_keys.add(key)

                if ekey is not None:
                    entity_counts[ekey] = entity_counts.get(ekey, 0) + 1
                    if entity_counts[ekey] >= MAX_PER_ENTITY:
                        exclude_ids.update(eids)

                q["_cam"] = cam
                all_questions.append(q)

            attempts += 1
            stale = stale + 1 if len(all_questions) == prev_count else 0
            if stale >= len(pool) * 2:
                break

    if not all_questions:
        return True

    os.makedirs(out_dir, exist_ok=True)
    print(f"  {scene_id}/{qtype}  ({len(all_questions)} question(s))")

    ok = True
    written = 0
    for q in all_questions:
        cam = q.pop("_cam")
        q["scene_id"] = scene_id
        q["layout_path"] = layout_path
        src_image = os.path.join(images_dir, f"{cam['name']}.jpg")
        fov = cam.get("horizontal_fov_deg", 82.0)
        try:
            write_question_image(
                q,
                qtype,
                written,
                src_image,
                images_dir,
                out_dir,
                fov,
                seg_mask_dir=seg_mask_dir,
                instance_id_map=instance_id_map,
            )
        except Exception as e:
            print(f"    [error] {qtype}: {e}", file=sys.stderr)
            ok = False
            continue
        with open(os.path.join(out_dir, f"question_{written}.json"), "w") as f:
            json.dump(q, f, indent=2)
        written += 1

    return ok


def process_scene(
    scene_path: str,
    question_types: list[str],
    output_root: str,
    num_questions: int,
    force_rerun: bool = False,
    workers: int = 1,
    **extra_ctx,
) -> bool:
    """Generate questions for one scene and all requested types.

    Pipelines can pass extra context (seg_mask_dir, instance_id_map,
    cam_visibility, visibility_kwargs, …) via **extra_ctx — these are
    forwarded to workers as-is.

    workers: number of parallel workers per scene.
             1 = sequential (default), N = explicit, 0 = auto (cpu_count).
    """
    with open(scene_path) as f:
        scene = json.load(f)

    scene_id = (
        scene.get("scene_id") or os.path.splitext(os.path.basename(scene_path))[0]
    )
    objects = scene["objects"]
    cameras = scene["cameras"]
    scene_room_bounds = scene.get("room_bounds")

    if not cameras:
        print(f"  [skip] {scene_id}: no cameras in scene JSON.", file=sys.stderr)
        return False

    images_dir = os.path.join(output_root, scene_id, "meta", "images")

    # Load VLM view filter results if available; skip cameras marked BAD.
    filter_path = os.path.join(os.path.dirname(scene_path), "view_filter.json")
    bad_cameras: set[str] = set()
    if os.path.isfile(filter_path):
        with open(filter_path) as f:
            filter_results = json.load(f)
        bad_cameras = {
            r["camera_name"] for r in filter_results if r.get("quality") == "BAD"
        }
        if bad_cameras:
            print(
                f"  [filter] skipping {len(bad_cameras)} BAD camera(s): {sorted(bad_cameras)}"
            )

    seen_rooms: dict = {}
    for c in cameras:
        if c.get("step_direction") or c.get("edge_direction"):
            continue
        if c["name"] in bad_cameras:
            continue
        room = c.get("room_name")
        key = room if room is not None else "__default__"
        if key not in seen_rooms:
            seen_rooms[key] = c

    if not seen_rooms:
        print(f"  [skip] {scene_id}: no non-stepped cameras found.", file=sys.stderr)
        return False

    if len(seen_rooms) > 1:

        def _count_objects_in_room(cam: dict) -> int:
            rb = cam.get("room_bounds")
            if not rb:
                return 0
            x_min, y_min, x_max, y_max = rb[0], rb[1], rb[2], rb[3]
            return sum(
                1
                for o in objects
                if x_min <= o.get("x", 0) <= x_max and y_min <= o.get("y", 0) <= y_max
            )

        best_cam = max(seen_rooms.values(), key=_count_objects_in_room)
        target_room = best_cam.get("room_name")
        print(
            f"  [camera] selected room '{target_room}' ({_count_objects_in_room(best_cam)} objects)"
        )
    else:
        target_room = list(seen_rooms.values())[0].get("room_name")

    camera_pool = []
    edge_camera_pool = []
    for c in cameras:
        if c.get("step_direction"):
            continue
        if c["name"] in bad_cameras:
            continue
        if c.get("room_name") != target_room and target_room is not None:
            continue
        img_path = os.path.join(images_dir, f"{c['name']}.jpg")
        if not os.path.isfile(img_path):
            continue
        try:
            from PIL import Image as _PILImage

            _img = _PILImage.open(img_path)
            if np.asarray(_img).mean() < 5.0:
                print(f"  [skip] {c['name']}: black image", file=sys.stderr)
                continue
        except Exception:
            pass
        if c.get("edge_direction"):
            edge_camera_pool.append(c)
        else:
            camera_pool.append(c)

    if not camera_pool:
        print(f"  [skip] {scene_id}: no camera images found.", file=sys.stderr)
        return False

    first_cam = camera_pool[0]
    if first_cam.get("room_bounds"):
        rb = first_cam["room_bounds"]
        x_min, y_min, x_max, y_max = rb[0], rb[1], rb[2], rb[3]
        room_objects = [
            o
            for o in objects
            if x_min <= o.get("x", 0) <= x_max and y_min <= o.get("y", 0) <= y_max
        ]
    else:
        room_objects = objects

    room_name = first_cam.get("room_name")
    room_cameras = (
        [c for c in cameras if c.get("room_name") == room_name]
        if room_name
        else cameras
    )
    room_bounds = first_cam.get("room_bounds") or scene_room_bounds
    layout_path = scene.get("layout_path") or scene.get("usd_path")
    base_seed = int(hashlib.md5(scene_id.encode()).hexdigest(), 16) % (2**31)

    # Precompute per-camera mask visibility on GPU (avoids per-worker I/O)
    _mask_vis_cache: dict[str, set[str]] | None = None
    _seg_dir = extra_ctx.get("seg_mask_dir")
    _id_map = extra_ctx.get("instance_id_map")
    if _seg_dir and _id_map:
        import numpy as _np
        _min_px = 5000
        _mask_vis_cache = {}
        all_cams = camera_pool + edge_camera_pool
        seg_files, seg_names = [], []
        for cam in all_cams:
            seg_path = os.path.join(_seg_dir, f"{cam['name']}_seg.npy")
            if os.path.isfile(seg_path):
                seg_files.append(seg_path)
                seg_names.append(cam["name"])
        if seg_files:
            oid_list = list(_id_map.keys())
            iid_list = list(_id_map.values())
            try:
                import torch as _torch
                from concurrent.futures import ThreadPoolExecutor as _TPE
                with _TPE(max_workers=8) as _pool:
                    _loaded = list(_pool.map(_np.load, seg_files))
                masks = _np.stack(_loaded)
                masks_t = _torch.from_numpy(masks).to(device="cuda", dtype=_torch.int32)
                iids_t = _torch.tensor(iid_list, device="cuda", dtype=_torch.int32)
                counts = (masks_t.unsqueeze(1) == iids_t.view(1, -1, 1, 1)).sum(dim=(2, 3))
                counts_cpu = counts.cpu().numpy()
                for ci, name in enumerate(seg_names):
                    _mask_vis_cache[name] = {
                        oid_list[j] for j in range(len(oid_list))
                        if counts_cpu[ci, j] >= _min_px
                    }
                del masks_t, iids_t, counts, masks
                _torch.cuda.empty_cache()
            except Exception:
                for path, name in zip(seg_files, seg_names):
                    mask = _np.load(path)
                    vis = set()
                    for oid, iid in _id_map.items():
                        if int((mask == iid).sum()) >= _min_px:
                            vis.add(oid)
                    _mask_vis_cache[name] = vis

    ctx = {
        "scene_id": scene_id,
        "scene_type": scene.get("scene_type", "sage"),
        "output_root": output_root,
        "room_objects": room_objects,
        "camera_pool": camera_pool,
        "edge_camera_pool": edge_camera_pool,
        "room_cameras": room_cameras,
        "room_bounds": room_bounds,
        "num_questions": num_questions,
        "force_rerun": force_rerun,
        "base_seed": base_seed,
        "images_dir": images_dir,
        "layout_path": layout_path,
        "_mask_vis_cache": _mask_vis_cache,
        **extra_ctx,
    }

    return run_parallel(_generate_qtype, question_types, ctx, workers=workers)
