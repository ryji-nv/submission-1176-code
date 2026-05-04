"""
SAGE scene utilities: layout loading, export, lighting, cameras, and rendering.

External dependencies (all Isaac Sim built-ins, no MobilityGen required):
  omni.usd, pxr (Gf, UsdGeom, UsdLux) — available in any Isaac Sim Python env.
  actor-simulation kits — needed only by load_sage_layout (isaacsim_utils).
"""

from __future__ import annotations

import json
import math
import os
import sys

import omni.usd
from pxr import Gf, UsdGeom, UsdLux

ROOT = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
)
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from .loader import get_room_bounds, load_layout_objects
from src.utils.cameras import (
    get_corner_camera_poses,
    get_edge_camera_poses,
    get_stepped_camera_poses,
)
from src.tasks import object_label


ACTOR_SIMULATION_KITS = os.path.join(os.path.dirname(__file__), "kits")

# Camera parameters — must match build_sage_corner_cameras + stage_add_camera defaults
_FOCAL_LENGTH_MM = 12.0
_HORIZONTAL_APERTURE_MM = 20.955
_CAMERA_HFOV_DEG = round(
    math.degrees(2 * math.atan(_HORIZONTAL_APERTURE_MM / (2 * _FOCAL_LENGTH_MM))), 2
)


# ---------------------------------------------------------------------------
# Layout loading (actor-simulation)
# ---------------------------------------------------------------------------


def load_sage_layout(layout_json_path: str, kits_path: str = None) -> list:
    """
    Load a SAGE scene from a layout JSON using actor-simulation's get_layout_scene_loaded.
    Replaces the current USD stage with the layout (walls, floor, doors, furniture under /World).
    Returns track_ids (list of tracked object mesh ids).
    """
    kits_path = kits_path or ACTOR_SIMULATION_KITS
    if not os.path.isdir(kits_path):
        raise FileNotFoundError(f"actor-simulation kits not found: {kits_path}")
    layout_json_path = os.path.abspath(layout_json_path)
    if not os.path.isfile(layout_json_path):
        raise FileNotFoundError(f"Layout JSON not found: {layout_json_path}")
    if kits_path not in sys.path:
        sys.path.insert(0, kits_path)
    import isaacsim_utils

    return isaacsim_utils.get_layout_scene_loaded(layout_json_path)


def apply_sage_semantic_labels(stage, objects: list[dict]) -> int:
    """Apply semantic labels to SAGE prims so instance_segmentation annotator works."""
    from pxr import Usd

    try:
        import Semantics as Sem
    except ImportError:
        from pxr import Semantics as Sem

    obj_ids = {str(o.get("id", "")) for o in objects if o.get("id")}
    labelled = 0
    world_prim = stage.GetPrimAtPath("/World")
    if not world_prim.IsValid():
        return 0

    for prim in Usd.PrimRange(world_prim):
        prim_name = prim.GetName()
        if prim_name in obj_ids:
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
# Export (layout JSON → scene.json)
# ---------------------------------------------------------------------------


def scene_to_dict(layout_path: str, *, include_edge_cameras: bool = False) -> dict:
    """Build a JSON-serializable dict with scene_id, layout_path, room_bounds, objects, cameras."""
    layout_path = os.path.abspath(layout_path)
    scene_id = os.path.basename(os.path.dirname(layout_path))
    room_bounds = get_room_bounds(layout_path)
    objects_raw = load_layout_objects(layout_path)
    cameras = get_corner_camera_poses(room_bounds)
    if include_edge_cameras:
        cameras = cameras + get_edge_camera_poses(room_bounds, prefix="sage")
    base_cam = next((c for c in cameras if c.get("edge_direction") == "south"), None)
    if base_cam is None:
        base_cam = next(
            (c for c in cameras if c["name"] == "sage_corner_camera_0"), None
        )
    if base_cam is not None:
        cameras = cameras + get_stepped_camera_poses(room_bounds, base_cam)

    objects = []
    for o in objects_raw:
        obj = {
            "id": o.get("id"),
            "label": object_label(o),
            "type": o.get("type"),
            "x": o["x"],
            "y": o["y"],
            "z": o["z"],
            "rotation_z": float((o.get("rotation") or {}).get("z", 0.0)),
            "width": o.get("width"),
            "length": o.get("length"),
            "height": o.get("height"),
        }
        objects.append(obj)

    cameras_list = []
    for c in cameras:
        entry = {
            "name": c["name"],
            "position": list(c["position"]),
            "look_at": list(c["look_at"]),
            "horizontal_fov_deg": _CAMERA_HFOV_DEG,
        }
        if c.get("step_direction"):
            entry["step_direction"] = c["step_direction"]
            entry["step_from"] = c["step_from"]
        if c.get("edge_direction"):
            entry["edge_direction"] = c["edge_direction"]
        cameras_list.append(entry)

    objects.sort(key=lambda o: str(o.get("id") or ""))

    return {
        "scene_id": scene_id,
        "layout_path": layout_path,
        "room_bounds": list(room_bounds),
        "objects": objects,
        "cameras": cameras_list,
    }


def export_scene(
    layout_path: str, output_path: str, *, include_edge_cameras: bool = False
) -> dict:
    """Export one scene to JSON. Returns the written dict."""
    data = scene_to_dict(layout_path, include_edge_cameras=include_edge_cameras)
    out_path = os.path.abspath(output_path)
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(data, f, indent=2)
    print(
        f"[export] {len(data['objects'])} objects, {len(data['cameras'])} cameras → {out_path}"
    )
    return data


# ---------------------------------------------------------------------------
# Lighting
# ---------------------------------------------------------------------------


def add_sage_lighting(stage, room_bounds=None, light_intensity=30000.0):
    """
    Add lighting to the stage.
    - room_bounds given: 4 corner SphereLight points (reference SAGE setup).
    - room_bounds None: DomeLight + DistantLight fallback.
    room_bounds: (rx, ry, rw, rl, rh).
    """
    if room_bounds is not None:
        rx, ry, rw, rl, rh = room_bounds

        def _sph(tag, pos):
            s = UsdLux.SphereLight.Define(stage, f"/World/Lights/{tag}")
            s.CreateIntensityAttr(light_intensity)
            s.CreateColorAttr(Gf.Vec3f(0.75, 0.75, 0.75))
            s.CreateTreatAsPointAttr(True)
            UsdGeom.Xformable(s.GetPrim()).AddTranslateOp().Set(Gf.Vec3d(*pos))

        _sph("corner_1", (rx + 0.2, ry + 0.2, rh - 0.2))
        _sph("corner_2", (rx + rw - 0.2, ry + 0.2, rh - 0.2))
        _sph("corner_3", (rx + 0.2, ry + rl - 0.2, rh - 0.2))
        _sph("corner_4", (rx + rw - 0.2, ry + rl - 0.2, rh - 0.2))
        return

    dome = UsdLux.DomeLight.Define(stage, "/World/DomeLight")
    dome.CreateIntensityAttr(5000)
    dome.CreateColorAttr(Gf.Vec3f(1, 0.98, 0.95))
    dl = UsdLux.DistantLight.Define(stage, "/World/DistantLight")
    dl.CreateIntensityAttr(4000)
    UsdGeom.Xformable(dl.GetPrim()).AddRotateXYZOp().Set(Gf.Vec3f(-40, 25, 0))


# ---------------------------------------------------------------------------
# Corner cameras
# ---------------------------------------------------------------------------


def _stage_add_camera(
    stage,
    path: str,
    focal_length: float = 35.0,
    horizontal_aperture: float = 20.955,
    vertical_aperture: float = 20.955,
    clipping_range=(0.1, 100000),
) -> UsdGeom.Camera:
    """Define a USD camera prim with the given intrinsics."""
    camera = UsdGeom.Camera.Define(stage, path)
    camera.CreateFocalLengthAttr().Set(focal_length)
    camera.CreateHorizontalApertureAttr().Set(horizontal_aperture)
    camera.CreateVerticalApertureAttr().Set(vertical_aperture)
    camera.CreateClippingRangeAttr().Set(clipping_range)
    return camera


def _set_camera_lookat(cam_prim, cam_pos: Gf.Vec3d, look_pos: Gf.Vec3d) -> None:
    """Orient the camera prim to look from cam_pos toward look_pos."""
    fwd = look_pos - cam_pos
    fwd_len = fwd.GetLength()
    if fwd_len < 1e-6:
        return
    fwd /= fwd_len
    world_up = Gf.Vec3d(0, 0, 1)
    right = Gf.Cross(fwd, world_up)
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


def build_sage_corner_cameras(stage, room_bounds, corner_cameras_config=None):
    """
    Build eight fixed corner cameras: four at top corners, four at mid height.
    All look at the room centroid.

    room_bounds: (rx, ry, rw, rl, rh).
    Returns list of (cam_path, camera_name).
    """
    rx, ry, rw, rl, rh = room_bounds
    cfg = corner_cameras_config if isinstance(corner_cameras_config, dict) else {}
    pad = float(cfg.get("pad", 0.12))
    look_at_height_frac = float(cfg.get("look_at_height_frac", 0.35))
    room_cx = rx + rw / 2.0
    room_cy = ry + rl / 2.0
    look_target = Gf.Vec3d(room_cx, room_cy, rh * look_at_height_frac)

    corner_xy = [
        (rx + pad, ry + pad),
        (rx + rw - pad, ry + pad),
        (rx + pad, ry + rl - pad),
        (rx + rw - pad, ry + rl - pad),
    ]
    cam_z_top = rh - 0.15
    cam_z_mid = rh / 2.0
    focal_length = float(cfg.get("focal_length", 12.0))
    clipping_range = (0.01, 200.0)

    specs = []
    for i, (cx, cy) in enumerate(corner_xy):
        for z, name_suffix in ((cam_z_top, str(i)), (cam_z_mid, f"mid_{i}")):
            cam_pos = Gf.Vec3d(cx, cy, z)
            idx = len(specs)
            cam_path = f"/World/sage_corner_camera_{idx}"
            state_name = f"sage_corner_camera_{name_suffix}"
            _stage_add_camera(
                stage,
                cam_path,
                focal_length=focal_length,
                clipping_range=clipping_range,
            )
            prim = stage.GetPrimAtPath(cam_path)
            _set_camera_lookat(prim, cam_pos, look_target)
            specs.append((cam_path, state_name))

    print(
        f"  SAGE corner cameras: 8 (4 top z={cam_z_top:.2f}, 4 mid z={cam_z_mid:.2f})"
        f" looking at ({room_cx:.2f}, {room_cy:.2f}, {rh * look_at_height_frac:.2f})",
        file=sys.stderr,
    )
    return specs


def build_edge_cameras(stage, room_bounds, *, focal_length: float = 12.0):
    """Build USD camera prims at wall midpoints. Returns list of (cam_path, cam_name)."""
    poses = get_edge_camera_poses(room_bounds, prefix="sage")
    specs = []
    for p in poses:
        cam_path = f"/World/{p['name']}"
        _stage_add_camera(
            stage, cam_path, focal_length=focal_length, clipping_range=(0.01, 200.0)
        )
        prim = stage.GetPrimAtPath(cam_path)
        _set_camera_lookat(prim, Gf.Vec3d(*p["position"]), Gf.Vec3d(*p["look_at"]))
        specs.append((cam_path, p["name"]))
    print(f"  Edge cameras: {len(specs)}", file=sys.stderr)
    return specs


def build_stepped_cameras(
    stage,
    room_bounds,
    base_cam_name: str = "sage_corner_camera_0",
    *,
    step_size: float = 0.5,
    focal_length: float = 12.0,
):
    """
    Build stepped USD camera prims for forward/left/right steps from base_cam_name.
    room_bounds: (rx, ry, rw, rl, rh).
    Returns list of (cam_path, camera_name).
    """
    edge_poses = get_edge_camera_poses(room_bounds, prefix="sage")
    base_camera = next(
        (c for c in edge_poses if c.get("edge_direction") == "south"), None
    )
    if base_camera is None:
        corner_poses = get_corner_camera_poses(room_bounds)
        base_camera = next(
            (c for c in corner_poses if c["name"] == base_cam_name), None
        )
    if base_camera is None:
        return []

    stepped = get_stepped_camera_poses(room_bounds, base_camera, step_size=step_size)
    specs = []
    for sc in stepped:
        cam_path = f"/World/{sc['name']}"
        _stage_add_camera(
            stage, cam_path, focal_length=focal_length, clipping_range=(0.01, 200.0)
        )
        prim = stage.GetPrimAtPath(cam_path)
        _set_camera_lookat(prim, Gf.Vec3d(*sc["position"]), Gf.Vec3d(*sc["look_at"]))
        specs.append((cam_path, sc["name"]))

    print(f"  Stepped cameras: {len(specs)} from {base_cam_name}", file=sys.stderr)
    return specs


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------


def render_scene(
    layout_path: str,
    output_dir: str,
    simulation_app,
    *,
    resolution: tuple[int, int] = (1920, 1080),
    warmup_steps: int = 8,
    include_edge_cameras: bool = False,
) -> int:
    """
    Render camera images for one scene. simulation_app must already be running.
    Returns the number of images saved.
    """
    import numpy as np
    import omni.replicator.core as rep
    import PIL.Image

    os.makedirs(output_dir, exist_ok=True)

    omni.usd.get_context().new_stage()
    simulation_app.update()

    print(f"Loading layout: {layout_path}")
    load_sage_layout(layout_path)
    simulation_app.update()

    stage = omni.usd.get_context().get_stage()
    room_bounds = get_room_bounds(layout_path)
    add_sage_lighting(stage, room_bounds=room_bounds)

    scene_data = scene_to_dict(layout_path, include_edge_cameras=include_edge_cameras)
    apply_sage_semantic_labels(stage, scene_data.get("objects", []))

    camera_specs = build_sage_corner_cameras(stage, room_bounds)
    if include_edge_cameras:
        camera_specs = camera_specs + build_edge_cameras(stage, room_bounds)
    camera_specs = camera_specs + build_stepped_cameras(stage, room_bounds)
    simulation_app.update()

    if not camera_specs:
        print(
            "Error: no corner cameras built — check layout has valid room bounds.",
            file=sys.stderr,
        )
        return 0

    normals_dir = os.path.join(os.path.dirname(output_dir), "normals")
    instance_seg_dir = os.path.join(os.path.dirname(output_dir), "instance_seg")
    os.makedirs(normals_dir, exist_ok=True)
    os.makedirs(instance_seg_dir, exist_ok=True)

    obj_id_set = {o["id"] for o in scene_data.get("objects", [])}

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
        out_path = os.path.join(output_dir, f"{cam_name}.jpg")
        PIL.Image.fromarray(frame).save(out_path, quality=95)
        print(f"Saved {out_path}")

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

    print(f"Done: {saved}/{len(camera_slots)} images saved to {output_dir}")
    return saved
