"""
SceneSmith end-to-end pipeline: collect → export+render → bbox → questions.

Called by run.py via src.scenesmith.pipeline.run(args).
"""

from __future__ import annotations

import os
import sys


def run(args) -> int:
    from src.scenes.scenesmith.loader import (
        list_scenesmith_scene_ids,
        get_scenesmith_usd_path_for_scene,
        get_scenesmith_scenes_dir,
    )
    from src.utils.isaac import needs_render

    output_root = os.path.abspath(args.output_root)

    if args.use_vlm_filter and not os.environ.get("VLM_API_KEY", "").strip():
        print(
            "Error: VLM_API_KEY is required with --use-vlm-filter.", file=sys.stderr
        )
        return 1

    if args.question_type is None:
        question_types = None
    elif args.question_type == "all":
        from src.tasks import ALL_QUESTION_TYPES

        question_types = ALL_QUESTION_TYPES
    else:
        question_types = [args.question_type]

    _EDGE_CAM_TYPES = {
        "camera_facing_direction",
        "camera_relative_position",
        "camera_motion",
        "camera_region_position",
        "object_region_position",
        "region_region_position",
    }
    include_edge_cameras = question_types is not None and bool(
        _EDGE_CAM_TYPES & set(question_types)
    )

    # List scenes (no download yet)
    try:
        sids = list_scenesmith_scene_ids(
            args.num_scenes, args.scenesmith_subset, seed=getattr(args, "seed", None)
        )
    except (FileNotFoundError, RuntimeError) as e:
        print(f"Error (SceneSmith): {e}", file=sys.stderr)
        return 1
    if not sids:
        print("No SceneSmith scenes to process.", file=sys.stderr)
        return 1
    if len(sids) < args.num_scenes:
        print(
            f"Warning: requested {args.num_scenes} SceneSmith scene(s), found {len(sids)}."
        )

    subset = args.scenesmith_subset

    sim_needed = any(
        not os.path.isfile(
            os.path.join(
                output_root, f"scenesmith_{subset.lower()}_{sid}", "scene.json"
            )
        )
        or needs_render(
            os.path.join(output_root, f"scenesmith_{subset.lower()}_{sid}", "images")
        )
        for sid in sids
    )

    simulation_app = None
    export_and_render_scenesmith = None
    if sim_needed:
        from src.utils.isaac import launch_sim

        simulation_app = launch_sim()
        from src.scenes.scenesmith.scene import export_and_render_scenesmith
    else:
        print("All scenes already rendered — Isaac Sim not started.")

    if question_types is not None:
        from src.generate import process_scene

    total = len(sids)
    n_ok = 0
    try:
        for i, sid in enumerate(sids):
            scene_id = f"scenesmith_{subset.lower()}_{sid}"
            scene_json = os.path.join(output_root, scene_id, "scene.json")
            images_dir = os.path.join(output_root, scene_id, "images")

            print(
                f"\n{'=' * 60}\nSceneSmith Scene {i + 1}/{total}: {scene_id}\n{'=' * 60}"
            )

            # Step 1+2 — export + render (combined; skip if outputs exist)
            if os.path.isfile(scene_json) and not needs_render(images_dir):
                print("  [export+render] skipped (outputs exist)")
            elif simulation_app is None:
                print(
                    "  [export+render] FAILED: Isaac Sim not started but outputs missing",
                    file=sys.stderr,
                )
                continue
            else:
                usd_path = get_scenesmith_usd_path_for_scene(sid, subset)
                if usd_path is None:
                    print("  FAILED: could not download scene", file=sys.stderr)
                    continue
                try:
                    os.makedirs(os.path.dirname(scene_json), exist_ok=True)
                    export_and_render_scenesmith(
                        usd_path,
                        scene_json,
                        images_dir,
                        simulation_app,
                        include_edge_cameras=include_edge_cameras,
                    )
                except Exception as e:
                    print(f"  [export+render] FAILED: {e}", file=sys.stderr)
                    continue

            # Step 3 — bbox overlays
            bbox_dir = os.path.join(output_root, scene_id, "3d_bbox_images")
            if os.path.isdir(bbox_dir) and os.listdir(bbox_dir):
                print("  [bbox] skipped (3d_bbox_images exists)")
            else:
                try:
                    from src.utils.bbox_overlay import generate_bbox_images

                    written = generate_bbox_images(scene_json, images_dir, flip_y=False)
                    for p in written:
                        print(f"  [bbox] {os.path.basename(p)}")
                except Exception as e:
                    print(f"  [bbox] FAILED: {e}", file=sys.stderr)

            # Step 4 — filter views (optional, requires --vlm-filter)
            if args.use_vlm_filter:
                try:
                    from src.utils.filter_views import filter_scene

                    filter_scene(scene_json, force_rerun=args.force_rerun)
                except Exception as e:
                    print(f"  [filter] FAILED: {e}", file=sys.stderr)

            # Step 5 — generate questions
            ok = True
            if question_types is not None:
                try:
                    ok = process_scene(
                        scene_path=scene_json,
                        question_types=question_types,
                        output_root=output_root,
                        num_questions=args.num_questions,
                        force_rerun=args.force_rerun,
                        workers=getattr(args, "workers", 1),
                    )
                except Exception as e:
                    print(f"  [generate] FAILED: {e}", file=sys.stderr)
                    ok = False

            # Step 7 — clean intermediates to free disk (only with --cleanup)
            if getattr(args, "cleanup", False):
                _scene_dir = os.path.join(output_root, scene_id)
                for d in ["images", "instance_seg", "normals", "3d_bbox_images"]:
                    p = os.path.join(_scene_dir, d)
                    if os.path.isdir(p):
                        import shutil

                        shutil.rmtree(p)
                for f in ["scene.json", "view_filter.json"]:
                    p = os.path.join(_scene_dir, f)
                    if os.path.isfile(p):
                        os.remove(p)
                src_dir = os.path.join(str(get_scenesmith_scenes_dir()), sid)
                if os.path.isdir(src_dir):
                    import shutil

                    shutil.rmtree(src_dir)
                print(f"  [cleanup] removed intermediates for {scene_id}")

            if ok:
                n_ok += 1

    finally:
        if simulation_app is not None:
            simulation_app.close()

    print(f"\n{'=' * 60}\nDone: {n_ok}/{total} scene(s) completed.\n{'=' * 60}")
    return 0 if n_ok == total else 1
