"""
Marble end-to-end pipeline: image -> Marble API -> segment -> label -> cameras -> render -> QA.

Called by run.py via ``src.scenes.marble.pipeline.run(args)``.

All pipeline modules live in this package and are called directly.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path


def _call(label: str, fn, argv: list[str]) -> int:
    """Call a script's main() with synthetic sys.argv. Returns 0 on success."""
    saved = sys.argv
    try:
        sys.argv = argv
        print(f"  [{label}] {' '.join(argv[:3])}...")
        fn()
        return 0
    except SystemExit as e:
        rc = e.code if isinstance(e.code, int) else 1
        if rc != 0:
            print(f"  [{label}] FAILED (rc={rc})", file=sys.stderr)
        return rc
    except Exception as e:
        print(f"  [{label}] FAILED: {e}", file=sys.stderr)
        return 1
    finally:
        sys.argv = saved


import threading

_scene_id_lock = threading.Lock()


def _find_or_create_scene_id(output_root: str, scene_name: str) -> str:
    """Find existing scene folder or create a new indexed one (thread-safe)."""
    with _scene_id_lock:
        if os.path.isdir(output_root):
            for name in sorted(os.listdir(output_root)):
                if name.split("_", 1)[-1] == scene_name:
                    return name
        existing = []
        if os.path.isdir(output_root):
            for name in os.listdir(output_root):
                parts = name.split("_", 1)
                if parts[0].isdigit():
                    existing.append(int(parts[0]))
        idx = max(existing, default=-1) + 1
        scene_id = f"{idx:04d}_{scene_name}"
        os.makedirs(os.path.join(output_root, scene_id), exist_ok=True)
        return scene_id


def _warmup_gsplat():
    """Pre-compile gsplat CUDA extension with a file lock to prevent races."""
    import fcntl
    lock_path = os.path.join(os.environ.get("TMPDIR", "/tmp"), "gsplat_compile.lock")
    with open(lock_path, "w") as lf:
        fcntl.flock(lf, fcntl.LOCK_EX)
        try:
            from gsplat import rasterization  # noqa: F401
        finally:
            fcntl.flock(lf, fcntl.LOCK_UN)


def run(args) -> int:
    _warmup_gsplat()

    image_path = getattr(args, "image", None)
    text_prompt = getattr(args, "text_prompt", None)
    scene_id_arg = getattr(args, "scene_id", None)

    _STEP_ORDER = ["1", "2", "3", "4c", "4", "4b", "5", "5b", "5c", "5e", "6", "7"]
    start_from = getattr(args, "start_from", None)
    if start_from:
        if start_from not in _STEP_ORDER:
            print(f"Error: --start-from {start_from!r} invalid. "
                  f"Valid: {', '.join(_STEP_ORDER)}", file=sys.stderr)
            return 1
        _rerun_steps = set(_STEP_ORDER[_STEP_ORDER.index(start_from):])
    else:
        _rerun_steps = set()

    if not image_path and not text_prompt and not scene_id_arg:
        print("Error: --image, --text-prompt, or --scene-id required for marble.",
              file=sys.stderr)
        return 1
    if image_path and not os.path.isfile(image_path):
        print(f"Error: --image {image_path!r} not found.", file=sys.stderr)
        return 1

    api_key = os.environ.get("WLT_API_KEY", "")
    output_root = os.path.abspath(args.output_root)

    force_rerun = getattr(args, "force_rerun", False)
    force_all = getattr(args, "force_rerun_all", False)
    force_pipeline = force_all

    def _should_rerun(step: str) -> bool:
        return step in _rerun_steps

    if scene_id_arg:
        scene_id = scene_id_arg
    elif image_path:
        scene_name = Path(image_path).stem
        scene_id = _find_or_create_scene_id(output_root, scene_name)
    else:
        import hashlib as _hl
        scene_name = "txt_" + _hl.md5(text_prompt.encode()).hexdigest()[:12]
        scene_id = _find_or_create_scene_id(output_root, scene_name)
    qa_dir = os.path.join(output_root, scene_id)
    meta_dir = os.path.join(qa_dir, "meta")
    scene_json_path = os.path.join(meta_dir, "scene.json")
    images_dir = os.path.join(meta_dir, "images")

    os.makedirs(meta_dir, exist_ok=True)

    import shutil
    if image_path:
        dst = os.path.join(qa_dir, "input" + Path(image_path).suffix)
        if not os.path.isfile(dst):
            shutil.copy2(image_path, dst)
    input_json_path = os.path.join(qa_dir, "input.json")
    if not os.path.isfile(input_json_path):
        full_entry = getattr(args, "_input_entry", None)
        if full_entry:
            input_meta = dict(full_entry)
        else:
            input_meta = {}
            if text_prompt:
                input_meta["prompt"] = text_prompt
            if image_path:
                input_meta["image"] = image_path
        with open(input_json_path, "w") as f:
            json.dump(input_meta, f, indent=2)

    if args.question_type is None:
        question_types = None
    elif args.question_type == "all":
        from src.utils.constants import ALL_QUESTION_TYPES
        question_types = ALL_QUESTION_TYPES
    else:
        question_types = [args.question_type]

    input_desc = image_path or (f"text: {text_prompt[:60]}" if text_prompt else f"scene-id: {scene_id}")
    print(f"\n{'='*60}")
    print(f"Marble pipeline: {scene_id}")
    print(f"  input:        {input_desc}")
    print(f"  qa_dir:       {qa_dir}")
    print(f"{'='*60}")

    # ── Step 1: Marble API -> download assets ─────────────────────────────
    freshly_generated = False
    world_json = os.path.join(meta_dir, "world_full.json")
    if os.path.isfile(world_json) and not force_pipeline and not _should_rerun("1"):
        print("\n[1/7] Marble assets: cached")
    else:
        if not api_key:
            print("Error: WLT_API_KEY env var required for Marble generation.", file=sys.stderr)
            return 1
        print("\n[1/7] Calling Marble API...")
        from src.scenes.marble.marble_api import main as marble_api_main
        api_argv = ["marble_api"]
        if image_path:
            api_argv.append(image_path)
        else:
            api_argv.append("__text_only__")
        api_argv.extend(["--api-key", api_key,
                         "--output-dir", meta_dir, "--skip-render"])
        if text_prompt:
            api_argv.extend(["--text-prompt", text_prompt])
        rc = _call("marble-api", marble_api_main, api_argv)
        if rc != 0:
            return 1
        freshly_generated = True

    # ── Step 2: SPZ -> PLY ────────────────────────────────────────────────
    ply_path = os.path.join(meta_dir, "scene.ply")
    spz_path = os.path.join(meta_dir, "splat_full_res.spz")
    if os.path.isfile(ply_path) and not force_pipeline and not _should_rerun("2"):
        print("\n[2/7] PLY: cached")
    elif os.path.isfile(spz_path):
        print("\n[2/7] Converting SPZ -> PLY...")
        from gaussforge import GaussForge
        gf = GaussForge()
        data = Path(spz_path).read_bytes()
        result = gf.convert(data, "spz", "ply")
        Path(ply_path).write_bytes(result["data"])
    else:
        print(f"Error: neither PLY nor SPZ found in {meta_dir}", file=sys.stderr)
        return 1

    if getattr(args, "_marble_api_only", False):
        print(f"\n{'='*60}\nAPI done: {scene_id}\n{'='*60}")
        return 0

    generation_only = getattr(args, "generation_only", False)

    seg_dir = os.path.join(meta_dir, "segmentation")
    instance_info_path = os.path.join(seg_dir, "instance_info.json")

    # ── Steps 3-4: segmentation + labeling ───────────────────────────────
    seg_scene = os.path.join(seg_dir, "scene.json")
    if os.path.isfile(seg_scene) and not force_pipeline and not _should_rerun("3"):
        print("\n[3/7] Segmentation: cached")
    else:
        print("\n[3/7] Running segmentation...")
        from src.scenes.marble.segment_scene import main as segment_main
        rc = _call("segment", segment_main,
                    ["segment_scene", "--marble-dir", meta_dir])
        if rc != 0:
            return 1

    # ── Step 4c: VLM 2D merge + label (before 3D labeling) ─────────────
    relabel_marker = os.path.join(seg_dir, "relabel.done")
    if os.path.isfile(relabel_marker) and not force_pipeline and not _should_rerun("4c"):
        print("\n[4c/7] VLM 2D merge+label: cached")
    else:
        print("\n[4c/7] VLM 2D merge + label (multi-round)...")
        from src.scenes.marble.scene import vlm_merge_and_label_2d
        vlm_merge_and_label_2d(seg_dir)
        Path(relabel_marker).touch()

    # ── Step 4: 3D labeling (uses VLM-corrected labels) ───────────────
    labels_path = os.path.join(seg_dir, "gaussian_labels.npy")
    if os.path.isfile(instance_info_path) and os.path.isfile(labels_path) and not force_pipeline and not _should_rerun("4"):
        print("\n[4/7] 3D labeling: cached")
    else:
        print("\n[4/7] Running 3D labeling...")
        import glob as _glob
        for _old in _glob.glob(os.path.join(images_dir, "*_seg.npy")):
            os.remove(_old)
        for _old in _glob.glob(os.path.join(images_dir, "*_seg_vis.png")):
            os.remove(_old)
        from src.scenes.marble.label_gaussians import main as label_main
        rc = _call("label3d", label_main,
                    ["label_gaussians", "--marble-dir", meta_dir, "--single-gpu"])
        if rc != 0:
            return 1

    # ── Step 4b: postprocess (split components only, no VLM merge) ──────
    postprocess_marker = os.path.join(seg_dir, "postprocess.done")
    if os.path.isfile(postprocess_marker) and not force_pipeline and not _should_rerun("4b"):
        print("\n[4b/7] Postprocessing: cached")
    else:
        print("\n[4b/7] Running postprocessing (split + VLM 3D merge)...")
        from src.scenes.marble.postprocess import main as postprocess_main
        rc = _call("postprocess", postprocess_main,
                    ["postprocess", "--marble-dir", meta_dir])
        if rc != 0:
            print("  [postprocess] WARNING: non-zero exit", file=sys.stderr)
        else:
            Path(postprocess_marker).touch()

    # ── Step 5: Build QA scene.json + cameras + render ───────────────────
    if os.path.isfile(scene_json_path) and not force_pipeline and not _should_rerun("5"):
        print("\n[5/7] scene.json + cameras + images: cached")
        with open(instance_info_path) as f:
            instance_info = json.load(f)
        # Ensure all non-rand cameras are rendered (may be missing from earlier runs)
        _scene_data = json.load(open(scene_json_path))
        _all_named = [c for c in _scene_data["cameras"]
                      if not c["name"].startswith("rand_")]
        _need = [c for c in _all_named
                 if not os.path.isfile(os.path.join(images_dir, f"{c['name']}.jpg"))]
        if _need:
            from src.scenes.marble.scene import _ground_offset, render_cameras
            _C = _ground_offset(os.path.join(seg_dir, "views", "cameras.json"))
            print(f"  Rendering {len(_need)} missing edge/step cameras...")
            render_cameras(
                ply_path, _need, images_dir, _C,
                labels_path=labels_path,
                instance_info=instance_info,
            )
    else:
        print("\n[5/7] Building QA scene.json + cameras + rendering...")
        import glob as _glob
        for _old in _glob.glob(os.path.join(images_dir, "rand_*")):
            os.remove(_old)
        for _old in _glob.glob(os.path.join(images_dir, "*_seg.npy")):
            os.remove(_old)
        for _old in _glob.glob(os.path.join(images_dir, "*_seg_vis.png")):
            os.remove(_old)

        from src.scenes.marble.scene import (
            _ground_offset, generate_cameras, render_cameras,
            build_scene_json, sample_random_cameras,
        )
        from src.scenes.marble.quality import filter_and_smooth

        TARGET_RAND = 64
        MAX_RETRIES = 5

        seg_cams_json = os.path.join(seg_dir, "views", "cameras.json")
        C = _ground_offset(seg_cams_json)
        with open(instance_info_path) as f:
            instance_info = json.load(f)
        seg_cameras = json.load(open(seg_cams_json))

        import numpy as _np
        from plyfile import PlyData as _PlyData
        _ply = _PlyData.read(ply_path)
        _v = _ply["vertex"]
        _gauss_xyz = _np.stack([_v["x"], _v["y"], _v["z"]], axis=-1).astype(_np.float32)
        _gauss_labels = _np.load(labels_path).astype(_np.int32)

        tmp_scene = build_scene_json(
            instance_info, [], "tmp", C,
            gaussian_xyz=_gauss_xyz, gaussian_labels=_gauss_labels,
        )
        objects = tmp_scene["objects"]

        # View + edge + step cameras (always kept)
        cameras = generate_cameras(
            seg_cameras, C, objects=objects, num_random=0, seed=42,
        )

        # Render view cameras first
        render_cameras(
            ply_path, cameras, images_dir, C,
            labels_path=labels_path,
            instance_info=instance_info,
        )

        BATCH_SIZE_CAM = TARGET_RAND * 5  # 320: oversample 5x, keep best 64

        # Retry loop: generate 32 rand cameras each round, filter, accumulate
        good_rand: list[dict] = []
        total_generated = 0
        gen_seed = 42
        for attempt in range(MAX_RETRIES):
            if len(good_rand) >= TARGET_RAND:
                break
            batch = sample_random_cameras(
                objects, BATCH_SIZE_CAM,
                start_index=total_generated, seed=gen_seed,
                existing_cameras=good_rand,
            )
            total_generated += len(batch)
            render_cameras(
                ply_path, batch, images_dir, C,
                labels_path=labels_path,
                instance_info=instance_info,
            )
            kept_batch, dropped_batch = filter_and_smooth(images_dir, batch)
            rand_kept = [c for c in kept_batch if c["name"].startswith("rand_")]
            good_rand.extend(rand_kept)
            gen_seed += 1000
            print(f"  [quality] attempt {attempt + 1}: "
                  f"+{len(rand_kept)} good, {len(dropped_batch)} bad "
                  f"({len(good_rand)}/{TARGET_RAND} total)")
            if not dropped_batch:
                break

        # Truncate to target and clean up excess files
        for cam in good_rand[TARGET_RAND:]:
            for ext in (".jpg", "_seg.npy"):
                p = os.path.join(images_dir, f"{cam['name']}{ext}")
                if os.path.isfile(p):
                    os.remove(p)
        good_rand = good_rand[:TARGET_RAND]

        # Renumber contiguously
        for i, cam in enumerate(good_rand):
            old = cam["name"]
            new = f"rand_{i:03d}"
            if old != new:
                for ext in (".jpg", "_seg.npy"):
                    old_p = os.path.join(images_dir, f"{old}{ext}")
                    new_p = os.path.join(images_dir, f"{new}{ext}")
                    if os.path.isfile(old_p):
                        os.replace(old_p, new_p)
                cam["name"] = new

        cameras.extend(good_rand)

        scene = build_scene_json(
            instance_info, cameras, scene_id, C,
            gaussian_xyz=_gauss_xyz, gaussian_labels=_gauss_labels,
        )
        del _gauss_xyz, _gauss_labels
        with open(scene_json_path, "w") as f:
            json.dump(scene, f, indent=2)
        n_rand = len(good_rand)
        n_view = len(cameras) - n_rand
        print(f"  {len(scene['objects'])} objects, "
              f"{n_view} view + {n_rand} rand cameras "
              f"-> {scene_json_path}")

    # ── Step 5b: Precompute per-camera visibility from seg masks (GPU) ──
    visibility_path = os.path.join(meta_dir, "visibility.json")
    if os.path.isfile(visibility_path) and not force_pipeline and not _should_rerun("5b"):
        print("\n[5b/7] visibility.json: cached")
    else:
        import time as _time
        _t0 = _time.perf_counter()
        import numpy as _np
        import torch as _torch

        with open(instance_info_path) as f:
            instance_info = json.load(f)
        id_map = {v["label"]: int(k) for k, v in instance_info.items()}

        min_px = 200
        edge_pad = 0.05
        min_dist = 0.3
        max_obj_dim = 2.0

        scene_data = json.load(open(scene_json_path))
        oversized = {
            o["id"] for o in scene_data["objects"]
            if max(o.get("width", 0), o.get("length", 0), o.get("height", 0)) > max_obj_dim
        }
        valid_objs = [(oid, iid) for oid, iid in id_map.items() if oid not in oversized]
        obj_ids = [oid for oid, _ in valid_objs]
        iids = [iid for _, iid in valid_objs]
        n_obj = len(obj_ids)

        cam_files = sorted(f for f in os.listdir(images_dir) if f.endswith("_seg.npy"))
        cam_names = [f.replace("_seg.npy", "") for f in cam_files]
        n_cam = len(cam_files)

        masks_list = [_np.load(os.path.join(images_dir, f)) for f in cam_files]
        H, W = masks_list[0].shape
        masks_gpu = _torch.from_numpy(_np.stack(masks_list)).to(
            device="cuda", dtype=_torch.int32)
        del masks_list

        iids_t = _torch.tensor(iids, device="cuda", dtype=_torch.int32)
        x_coords = _torch.arange(W, device="cuda", dtype=_torch.float32)
        y_coords = _torch.arange(H, device="cuda", dtype=_torch.float32)
        edge_xl, edge_xh = W * edge_pad, W * (1 - edge_pad)
        edge_yl, edge_yh = H * edge_pad, H * (1 - edge_pad)

        vis_mask = _torch.zeros(n_cam, n_obj, device="cuda", dtype=_torch.bool)
        for oi in range(n_obj):
            match = (masks_gpu == iids_t[oi])
            counts = match.sum(dim=(1, 2))
            has_enough = counts >= min_px
            if not has_enough.any():
                continue
            match_f = match.float()
            cx = (match_f * x_coords.view(1, 1, W)).sum(dim=(1, 2)) / counts.clamp(min=1).float()
            cy = (match_f * y_coords.view(1, H, 1)).sum(dim=(1, 2)) / counts.clamp(min=1).float()
            in_bounds = (cx >= edge_xl) & (cx <= edge_xh) & (cy >= edge_yl) & (cy <= edge_yh)
            vis_mask[:, oi] = has_enough & in_bounds

        del masks_gpu, iids_t
        _torch.cuda.empty_cache()

        # Batched 3D projection check
        cam_lookup = {c["name"]: c for c in scene_data["cameras"]}
        obj_lookup = {o["id"]: o for o in scene_data["objects"]}
        cam_indices = []
        w2c_list = []
        intrinsics = []
        for ci, cn in enumerate(cam_names):
            cam = cam_lookup.get(cn)
            if cam and "c2w" in cam:
                w2c_list.append(_np.linalg.inv(_np.array(cam["c2w"], dtype=_np.float32)))
                intrinsics.append((cam["fx"], cam["fy"], cam["cx"], cam["cy"]))
                cam_indices.append(ci)

        if w2c_list:
            w2c_t = _torch.from_numpy(_np.stack(w2c_list)).cuda()
            fx_t = _torch.tensor([i[0] for i in intrinsics], device="cuda")
            fy_t = _torch.tensor([i[1] for i in intrinsics], device="cuda")
            cx_t = _torch.tensor([i[2] for i in intrinsics], device="cuda")
            cy_t = _torch.tensor([i[3] for i in intrinsics], device="cuda")

            ply_centers = []
            for oid in obj_ids:
                o = obj_lookup.get(oid)
                pc = o["ply_center"] if o and o.get("ply_center") else [0, 0, 0]
                ply_centers.append(pc + [1.0])
            pts = _torch.tensor(ply_centers, device="cuda", dtype=_torch.float32)

            nc = len(w2c_list)
            proj = _torch.bmm(w2c_t, pts.T.unsqueeze(0).expand(nc, -1, -1))
            depth = proj[:, 2, :]
            u = fx_t.unsqueeze(1) * proj[:, 0, :] / depth.clamp(min=1e-6) + cx_t.unsqueeze(1)
            v = fy_t.unsqueeze(1) * proj[:, 1, :] / depth.clamp(min=1e-6) + cy_t.unsqueeze(1)
            proj_ok = (depth >= min_dist) & (u >= 0) & (u < W) & (v >= 0) & (v < H)

            for vi, ci in enumerate(cam_indices):
                vis_mask[ci] &= proj_ok[vi]

            del w2c_t, proj, depth, u, v, proj_ok, pts
            _torch.cuda.empty_cache()

        vis_cpu = vis_mask.cpu().numpy()
        vis: dict[str, list[str]] = {}
        for ci, cn in enumerate(cam_names):
            vis[cn] = [obj_ids[oi] for oi in range(n_obj) if vis_cpu[ci, oi]]
        del vis_mask, vis_cpu

        with open(visibility_path, "w") as f:
            json.dump(vis, f, indent=2)
        total_pairs = sum(len(v) for v in vis.values())
        elapsed = _time.perf_counter() - _t0
        print(f"\n[5b/7] visibility.json: {n_cam} cameras, "
              f"{total_pairs} pairs ({elapsed:.1f}s)")

    # ── Steps 5c + 5e: Run in parallel (VLM dim filter + VLM camera quality)
    dim_filter_path = os.path.join(meta_dir, "dimension_filter.json")

    def _run_5c():
        if os.path.isfile(dim_filter_path) and not force_pipeline and not _should_rerun("5c"):
            exc = set(json.load(open(dim_filter_path)).get("excluded", []))
            print(f"\n[5c/7] Dimension filter: cached ({len(exc)} excluded)")
            return exc
        with open(scene_json_path) as f:
            _sd = json.load(f)
        from src.scenes.marble.scene import validate_object_dimensions
        print("\n[5c/7] Running dimension filter...")
        exc = validate_object_dimensions(_sd["objects"])
        with open(dim_filter_path, "w") as f:
            json.dump({"excluded": sorted(exc)}, f, indent=2)
        return exc

    cam_filter_path = os.path.join(meta_dir, "camera_filter.json")

    def _run_5e():
        if os.path.isfile(cam_filter_path) and not force_pipeline and not _should_rerun("5e"):
            exc = set(json.load(open(cam_filter_path)).get("rejected", []))
            print(f"\n[5e/7] Camera quality VLM: cached ({len(exc)} rejected)")
            return exc
        with open(scene_json_path) as f:
            _sd = json.load(f)
        _rand_cams = [c for c in _sd["cameras"] if c["name"].startswith("rand_")]
        print(f"\n[5e/7] VLM camera quality filter...")
        from src.scenes.marble.quality import filter_cameras_vlm
        exc = filter_cameras_vlm(images_dir, _rand_cams)
        with open(cam_filter_path, "w") as f:
            json.dump({"rejected": sorted(exc)}, f, indent=2)
        return exc

    from concurrent.futures import ThreadPoolExecutor as _TPE_5
    with _TPE_5(max_workers=2) as _pool_5:
        _fut_5c = _pool_5.submit(_run_5c)
        _fut_5e = _pool_5.submit(_run_5e)
        dim_exclude: set[str] = _fut_5c.result()
        bad_cameras: set[str] = _fut_5e.result()

    # Remove rejected cameras from scene.json and clean up files
    if bad_cameras:
        with open(scene_json_path) as f:
            _sd = json.load(f)
        _sd["cameras"] = [c for c in _sd["cameras"] if c["name"] not in bad_cameras]
        with open(scene_json_path, "w") as f:
            json.dump(_sd, f, indent=2)
        for _bc in bad_cameras:
            for _ext in (".jpg", "_seg.npy"):
                _bp = os.path.join(images_dir, f"{_bc}{_ext}")
                if os.path.isfile(_bp):
                    os.remove(_bp)
        print(f"  Removed {len(bad_cameras)} bad cameras from scene.json")

    # ── Early exit if generation-only ───────────────────────────────────
    if generation_only:
        print(f"\n{'='*60}\nDone (generation only): {scene_id}\n{'='*60}")
        return 0

    # ── Step 6: Generate questions ───────────────────────────────────────
    if question_types is not None:
        print("\n[6/7] Generating questions...")
        with open(instance_info_path) as f:
            _inst = json.load(f)
        _id_map = {v["label"]: int(k) for k, v in _inst.items()}
        _cam_vis = None
        if os.path.isfile(visibility_path):
            with open(visibility_path) as f:
                _cam_vis = json.load(f)

        from src.generate import process_scene
        ok = process_scene(
            scene_path=scene_json_path,
            question_types=question_types,
            output_root=output_root,
            num_questions=args.num_questions,
            force_rerun=force_rerun or force_all or _should_rerun("6"),
            workers=getattr(args, "workers", 0),
            seg_mask_dir=images_dir,
            instance_id_map=_id_map,
            cam_visibility=_cam_vis,
            visibility_kwargs={"min_thickness": 0.01, "min_visible_fraction": 0.5},
            dimension_exclude=dim_exclude,
        )
        if not ok:
            print("  [generate] some question types failed", file=sys.stderr)
    else:
        print("\n[6/7] Question generation: skipped (no --question-type)")

    print("\n[7/7] Upload: skipped")

    print(f"\n{'='*60}\nDone: {scene_id}\n{'='*60}")
    return 0
