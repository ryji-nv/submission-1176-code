"""
SceneSmith scene utilities for Isaac Sim: load USD, extract room bounds + objects,
build corner cameras, and render images.

All functions require an active Isaac Sim / SimulationApp context.
Import this module only from within ./app/python.sh scripts.
"""

from __future__ import annotations

import os
import sys

ROOT = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
)
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

SCENESMITH_CEILING_HEIGHT = (
    2.7  # metres; standard in SceneSmith MuJoCo-converted scenes
)
SCENESMITH_HFOV_DEG = 82.0  # matches SAGE corner cameras


# ---------------------------------------------------------------------------
# Scene loading
# ---------------------------------------------------------------------------


def load_scenesmith_scene(usd_path: str, simulation_app) -> None:
    """Load a SceneSmith USD into a fresh stage. Adds lighting; disables physics."""
    import omni.usd
    from isaacsim.core.utils.stage import add_reference_to_stage
    from pxr import Gf, Usd, UsdGeom, UsdLux, UsdPhysics

    omni.usd.get_context().new_stage()
    simulation_app.update()

    add_reference_to_stage(usd_path, "/World/scene")
    simulation_app.update()

    stage = omni.usd.get_context().get_stage()

    # Disable rigid-body physics so objects don't fall
    for prim in Usd.PrimRange(stage.GetPrimAtPath("/World/scene")):
        if prim.HasAPI(UsdPhysics.RigidBodyAPI):
            UsdPhysics.RigidBodyAPI(prim).GetRigidBodyEnabledAttr().Set(False)
        if prim.HasAPI(UsdPhysics.ArticulationRootAPI):
            prim.RemoveAppliedSchema("PhysicsArticulationRootAPI")

    # Lighting
    dome = UsdLux.DomeLight.Define(stage, "/World/ScenesmithLight/Dome")
    dome.CreateIntensityAttr(2000)
    dome.CreateColorAttr(Gf.Vec3f(1, 0.98, 0.95))
    dl = UsdLux.DistantLight.Define(stage, "/World/ScenesmithLight/Distant")
    dl.CreateIntensityAttr(500)
    UsdGeom.Xformable(dl.GetPrim()).AddRotateXYZOp().Set(Gf.Vec3f(-40, 25, 0))


# ---------------------------------------------------------------------------
# Room bounds
# ---------------------------------------------------------------------------


def get_scenesmith_room_bounds(stage) -> list[tuple]:
    """
    Extract room bounding boxes from a loaded SceneSmith USD stage.

    Returns list of (cx, cy, hw, hh, room_name):
      cx, cy    — room centre in world XY
      hw, hh    — floor half-extents (room spans [cx-hw, cx+hw] x [cy-hh, cy+hh])
      room_name — short label e.g. 'bedroom', 'hallway'
    """
    from pxr import Usd, UsdGeom

    rooms = []
    scene_prim = stage.GetPrimAtPath("/World/scene")
    if not scene_prim.IsValid():
        return rooms

    for prim in Usd.PrimRange(scene_prim):
        name = prim.GetName()
        if "room_geometry" not in name or "body_link" in name:
            continue
        room_world = UsdGeom.Xformable(prim).ComputeLocalToWorldTransform(
            Usd.TimeCode.Default()
        )
        cx, cy = room_world[3][0], room_world[3][1]
        for body_link in prim.GetChildren():
            for child in body_link.GetChildren():
                if "floor_collision" not in child.GetName():
                    continue
                for op in UsdGeom.Xformable(child).GetOrderedXformOps():
                    if op.GetOpType() == UsdGeom.XformOp.TypeScale:
                        scale = op.Get()
                        short_name = name.split("_room_geometry_")[0]
                        rooms.append(
                            (cx, cy, float(scale[0]), float(scale[1]), short_name)
                        )
    return rooms


def room_bounds_to_dict(
    cx: float, cy: float, hw: float, hh: float, room_name: str
) -> dict:
    """Convert raw room tuple to scene.json-compatible room_bounds list + room_name."""
    return {
        "room_bounds": [
            cx - hw,
            cy - hh,
            cx - hw + 2 * hw,
            cy - hh + 2 * hh,
            SCENESMITH_CEILING_HEIGHT,
        ],
        "room_name": room_name,
    }


# ---------------------------------------------------------------------------
# Object extraction
# ---------------------------------------------------------------------------

# Prims skipped when extracting furniture objects (direct children of Geometry).
_OBJECT_SKIP_KEYWORDS = {
    "room_geometry",
    "wall",
    "floor",
    "ceiling",
    "ground_plane",
    "ground",
    "body_link",
    "collision",
}


def extract_scenesmith_objects(stage) -> list[dict]:
    """
    Extract furniture objects from /World/scene/Geometry as scene.json object dicts.

    Each dict has: id, label, type, x, y, z (floor base), width, length, height.
    Very small prims (< 0.05 m in any dimension) and structural prims are skipped.
    """
    from pxr import Usd, UsdGeom

    geometry_prim = stage.GetPrimAtPath("/World/scene/Geometry")
    if not geometry_prim.IsValid():
        geometry_prim = stage.GetPrimAtPath("/World/scene")
    if not geometry_prim.IsValid():
        return []

    bbox_cache = UsdGeom.BBoxCache(
        Usd.TimeCode.Default(),
        includedPurposes=[UsdGeom.Tokens.default_],
        useExtentsHint=True,
    )

    objects: list[dict] = []
    for prim in geometry_prim.GetChildren():
        if not prim.IsActive():
            continue
        name = prim.GetName().lower()
        if any(kw in name for kw in _OBJECT_SKIP_KEYWORDS):
            continue
        try:
            bbox = bbox_cache.ComputeWorldBound(prim)
        except Exception:
            continue
        rng = bbox.GetRange()
        if rng.IsEmpty():
            continue
        mn, mx = rng.GetMin(), rng.GetMax()
        w = float(mx[0] - mn[0])
        l = float(mx[1] - mn[1])
        h = float(mx[2] - mn[2])
        if w < 0.05 or l < 0.05 or h < 0.05:
            continue
        prim_name = prim.GetName()
        parts = prim_name.rsplit("_", 1)
        obj_type = parts[0] if len(parts) == 2 and parts[1].isdigit() else prim_name
        objects.append(
            {
                "id": prim_name,
                "label": obj_type,
                "type": obj_type,
                "x": float((mn[0] + mx[0]) / 2),
                "y": float((mn[1] + mx[1]) / 2),
                "z": float(mn[2]),
                "rotation_z": 0.0,
                "width": round(w, 4),
                "length": round(l, 4),
                "height": round(h, 4),
            }
        )

    return objects


def apply_semantic_labels(stage, objects: list[dict]) -> int:
    """Apply semantic labels to object prims for instance_segmentation annotator."""
    try:
        import Semantics as Sem
    except ImportError:
        from pxr import Semantics as Sem

    geometry_prim = stage.GetPrimAtPath("/World/scene/Geometry")
    if not geometry_prim.IsValid():
        geometry_prim = stage.GetPrimAtPath("/World/scene")

    labelled = 0
    obj_ids = {o["id"] for o in objects}

    for prim in geometry_prim.GetChildren():
        prim_name = prim.GetName()
        if prim_name not in obj_ids:
            continue
        sem = Sem.SemanticsAPI.Apply(prim, "Semantics")
        sem.CreateSemanticTypeAttr().Set("class")
        sem.CreateSemanticDataAttr().Set(prim_name)
        labelled += 1

    print(
        f"  [semantics] labelled {labelled}/{len(obj_ids)} object prims",
        file=sys.stderr,
    )
    return labelled


# ---------------------------------------------------------------------------
# Camera building
# ---------------------------------------------------------------------------


def _stage_add_camera(stage, path, focal_length=12.0, clipping_range=(0.01, 200.0)):
    from pxr import UsdGeom

    camera = UsdGeom.Camera.Define(stage, path)
    camera.CreateFocalLengthAttr().Set(focal_length)
    camera.CreateHorizontalApertureAttr().Set(20.955)
    camera.CreateVerticalApertureAttr().Set(20.955)
    camera.CreateClippingRangeAttr().Set(clipping_range)
    return camera


def _set_camera_lookat(cam_prim, cam_pos, look_pos) -> None:
    from pxr import Gf, UsdGeom

    fwd = look_pos - cam_pos
    fwd_len = fwd.GetLength()
    if fwd_len < 1e-6:
        return
    fwd /= fwd_len
    right = Gf.Cross(fwd, Gf.Vec3d(0, 0, 1))
    right_len = right.GetLength()
    if right_len < 1e-6:
        return
    right /= right_len
    up = Gf.Cross(right, fwd)
    mat = Gf.Matrix4d()
    mat.SetRow(0, Gf.Vec4d(right[0], right[1], right[2], 0))
    mat.SetRow(1, Gf.Vec4d(up[0], up[1], up[2], 0))
    mat.SetRow(2, Gf.Vec4d(-fwd[0], -fwd[1], -fwd[2], 0))
    mat.SetRow(3, Gf.Vec4d(cam_pos[0], cam_pos[1], cam_pos[2], 1))
    xf = UsdGeom.Xformable(cam_prim)
    xf.ClearXformOpOrder()
    xf.AddTransformOp().Set(mat)


def build_scenesmith_cameras(
    stage, room_bounds: list[tuple], pad: float = 0.12
) -> list[tuple]:
    """
    Build 4 corner cameras per room.  Returns list of (cam_path, cam_name).

    cam_name format: scenesmith_{room_name}_corner_{0-3}
    (consistent with camera names stored in scene.json).
    """
    from pxr import Gf

    cam_z = SCENESMITH_CEILING_HEIGHT - 0.15
    look_z = SCENESMITH_CEILING_HEIGHT * 0.35
    specs = []

    for room_idx, (cx, cy, hw, hh, room_name) in enumerate(room_bounds):
        look_target = Gf.Vec3d(cx, cy, look_z)
        corners = [
            (cx - hw + pad, cy - hh + pad),
            (cx + hw - pad, cy - hh + pad),
            (cx - hw + pad, cy + hh - pad),
            (cx + hw - pad, cy + hh - pad),
        ]
        for corner_idx, (px, py) in enumerate(corners):
            cam_name = f"scenesmith_{room_name}_corner_{corner_idx}"
            cam_path = f"/World/scenesmith_camera_{room_idx}_{corner_idx}"
            _stage_add_camera(stage, cam_path)
            prim = stage.GetPrimAtPath(cam_path)
            _set_camera_lookat(prim, Gf.Vec3d(px, py, cam_z), look_target)
            specs.append((cam_path, cam_name))

        print(
            f"  Room '{room_name}': 4 cameras at z={cam_z:.2f}, "
            f"centre=({cx:.2f}, {cy:.2f})",
            file=sys.stderr,
        )

    return specs


# ---------------------------------------------------------------------------
# Combined export + render
# ---------------------------------------------------------------------------


def export_and_render_scenesmith(
    usd_path: str,
    scene_json_path: str,
    images_dir: str,
    simulation_app,
    *,
    resolution: tuple[int, int] = (1920, 1080),
    warmup_steps: int = 8,
    include_edge_cameras: bool = False,
) -> dict:
    """
    Load a SceneSmith scene, write scene.json, and render all camera images.

    Returns the scene dict written to scene.json.
    """
    import json
    import math

    import numpy as np
    import omni.usd
    import omni.replicator.core as rep
    import PIL.Image
    from pxr import Gf

    from src.utils.cameras import get_edge_camera_poses, get_stepped_camera_poses

    os.makedirs(images_dir, exist_ok=True)
    os.makedirs(os.path.dirname(os.path.abspath(scene_json_path)) or ".", exist_ok=True)

    scene_id = os.path.basename(os.path.dirname(os.path.abspath(scene_json_path)))

    print(f"Loading SceneSmith USD: {usd_path}")
    load_scenesmith_scene(usd_path, simulation_app)
    simulation_app.update()

    stage = omni.usd.get_context().get_stage()

    room_bounds_raw = get_scenesmith_room_bounds(stage)
    if not room_bounds_raw:
        raise RuntimeError(f"No rooms found in {usd_path}")

    objects = extract_scenesmith_objects(stage)
    objects.sort(key=lambda o: str(o.get("id") or ""))
    apply_semantic_labels(stage, objects)
    camera_specs = build_scenesmith_cameras(stage, room_bounds_raw)
    simulation_app.update()

    # Build camera list for scene.json (one entry per camera with per-room bounds)
    _HFOV = round(math.degrees(2 * math.atan(20.955 / (2 * 12.0))), 2)
    _PAD = 0.12
    _CAM_Z = SCENESMITH_CEILING_HEIGHT - 0.15
    _LOOK_Z = SCENESMITH_CEILING_HEIGHT * 0.35
    room_lookup = {name: (cx, cy, hw, hh) for cx, cy, hw, hh, name in room_bounds_raw}

    cameras_list = []
    for _cam_path, cam_name in list(
        camera_specs
    ):  # snapshot: stepped cameras appended below
        # cam_name = "scenesmith_{room_name}_corner_{corner_idx}"
        without_prefix = cam_name[len("scenesmith_") :]  # "{room_name}_corner_{n}"
        room_name, corner_str = without_prefix.rsplit("_corner_", 1)
        corner_idx = int(corner_str)
        cx, cy, hw, hh = room_lookup[room_name]
        corners = [
            (cx - hw + _PAD, cy - hh + _PAD),
            (cx + hw - _PAD, cy - hh + _PAD),
            (cx - hw + _PAD, cy + hh - _PAD),
            (cx + hw - _PAD, cy + hh - _PAD),
        ]
        px, py = corners[corner_idx]

        cam_entry = {
            "name": cam_name,
            "position": [px, py, _CAM_Z],
            "look_at": [cx, cy, _LOOK_Z],
            "horizontal_fov_deg": _HFOV,
            "room_name": room_name,
            "room_bounds": [
                cx - hw,
                cy - hh,
                cx + hw,
                cy + hh,
                SCENESMITH_CEILING_HEIGHT,
            ],
        }
        cameras_list.append(cam_entry)

        # Add edge cameras for corner_0's room (once per room)
        if corner_idx == 0 and include_edge_cameras:
            rb = (cx - hw, cy - hh, 2 * hw, 2 * hh, SCENESMITH_CEILING_HEIGHT)
            edge_poses = get_edge_camera_poses(rb, prefix=f"scenesmith_{room_name}")
            for ep in edge_poses:
                ep_path = f"/World/scenesmith_edge_{room_name}_{ep['edge_direction']}"
                _stage_add_camera(stage, ep_path)
                _set_camera_lookat(
                    stage.GetPrimAtPath(ep_path),
                    Gf.Vec3d(*ep["position"]),
                    Gf.Vec3d(*ep["look_at"]),
                )
                camera_specs.append((ep_path, ep["name"]))
                cameras_list.append(
                    {
                        "name": ep["name"],
                        "position": list(ep["position"]),
                        "look_at": list(ep["look_at"]),
                        "horizontal_fov_deg": _HFOV,
                        "room_name": room_name,
                        "room_bounds": list(cam_entry["room_bounds"]),
                        "edge_direction": ep["edge_direction"],
                    }
                )

        # Add stepped cameras from edge_south (or corner_0 fallback)
        if corner_idx == 0:
            rb = (cx - hw, cy - hh, 2 * hw, 2 * hh, SCENESMITH_CEILING_HEIGHT)
            edge_south = next(
                (
                    e
                    for e in cameras_list
                    if e.get("edge_direction") == "south"
                    and e.get("room_name") == room_name
                ),
                None,
            )
            step_base = edge_south or cam_entry
            stepped = get_stepped_camera_poses(
                rb,
                {
                    "name": step_base["name"],
                    "position": tuple(step_base["position"]),
                    "look_at": tuple(step_base["look_at"]),
                },
            )
            for sc in stepped:
                # Build USD camera prim for stepped pose
                sc_path = (
                    f"/World/scenesmith_stepped_{room_name}_{sc['step_direction']}"
                )
                _stage_add_camera(stage, sc_path)
                _set_camera_lookat(
                    stage.GetPrimAtPath(sc_path),
                    Gf.Vec3d(*sc["position"]),
                    Gf.Vec3d(*sc["look_at"]),
                )
                camera_specs.append((sc_path, sc["name"]))
                cameras_list.append(
                    {
                        "name": sc["name"],
                        "position": list(sc["position"]),
                        "look_at": list(sc["look_at"]),
                        "horizontal_fov_deg": _HFOV,
                        "room_name": room_name,
                        "room_bounds": list(cam_entry["room_bounds"]),
                        "step_direction": sc["step_direction"],
                        "step_from": sc["step_from"],
                    }
                )

    scene_data = {
        "scene_id": scene_id,
        "scene_type": "scenesmith",
        "usd_path": os.path.abspath(usd_path),
        "room_bounds": cameras_list[0]["room_bounds"] if cameras_list else [],
        "objects": objects,
        "cameras": cameras_list,
    }
    with open(scene_json_path, "w") as f:
        json.dump(scene_data, f, indent=2)
    print(
        f"[export] {len(objects)} objects, {len(cameras_list)} cameras → {scene_json_path}"
    )

    # Render
    normals_dir = os.path.join(os.path.dirname(images_dir), "normals")
    instance_seg_dir = os.path.join(os.path.dirname(images_dir), "instance_seg")
    os.makedirs(normals_dir, exist_ok=True)
    os.makedirs(instance_seg_dir, exist_ok=True)

    camera_slots = []
    for cam_path, cam_name in camera_specs:
        rp = rep.create.render_product(cam_path, resolution, force_new=True)
        color_ann = rep.AnnotatorRegistry.get_annotator("LdrColor")
        color_ann.attach(rp)
        normals_ann = rep.AnnotatorRegistry.get_annotator("normals")
        normals_ann.attach(rp)
        instance_ann = rep.AnnotatorRegistry.get_annotator("instance_segmentation")
        instance_ann.attach(rp)
        camera_slots.append((color_ann, normals_ann, instance_ann, cam_name))

    for _ in range(warmup_steps):
        simulation_app.update()
        rep.orchestrator.step(rt_subframes=2, delta_time=0.0, pause_timeline=False)

    obj_id_set = {o["id"] for o in objects}

    saved = 0
    for color_ann, normals_ann, instance_ann, cam_name in camera_slots:
        frame = color_ann.get_data()
        if frame is None or getattr(frame, "size", 0) == 0:
            print(f"  [warn] no data for {cam_name}", file=sys.stderr)
            continue
        frame = np.asarray(frame)
        if frame.ndim == 3 and frame.shape[2] >= 3:
            frame = frame[:, :, :3]
        if frame.dtype in (np.float32, np.float64):
            frame = (np.clip(frame, 0, 1) * 255).astype(np.uint8)
        else:
            frame = frame.astype(np.uint8)
        out_path = os.path.join(images_dir, f"{cam_name}.jpg")
        PIL.Image.fromarray(frame).save(out_path, quality=95)
        print(f"  Saved {out_path}")

        nrm_data = normals_ann.get_data()
        if nrm_data is not None:
            nrm_arr = np.asarray(nrm_data, dtype=np.float32)
            if nrm_arr.ndim == 3 and nrm_arr.shape[2] >= 3:
                nrm_arr = nrm_arr[:, :, :3]
            np.save(os.path.join(normals_dir, f"{cam_name}.npy"), nrm_arr)

        inst_data = instance_ann.get_data()
        if inst_data is not None and isinstance(inst_data, dict):
            seg_arr = np.asarray(inst_data.get("data", inst_data), dtype=np.uint32)
            np.save(os.path.join(instance_seg_dir, f"{cam_name}.npy"), seg_arr)
            info = inst_data.get("info", {})
            id_to_labels = info.get("idToLabels", {})
            id_to_obj = {}
            for sid_str, labels in id_to_labels.items():
                prim = labels if isinstance(labels, str) else str(labels)
                for oid in obj_id_set:
                    if oid in prim:
                        id_to_obj[sid_str] = oid
                        break
            seg_info = {
                "idToLabels": id_to_labels,
                "primPaths": info.get("primPaths", []),
                "idToObjectId": id_to_obj,
            }
            with open(
                os.path.join(instance_seg_dir, f"{cam_name}_info.json"), "w"
            ) as fj:
                json.dump(seg_info, fj, indent=2)

        saved += 1

    print(f"  Done: {saved}/{len(camera_slots)} images saved")
    return scene_data
