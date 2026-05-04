#!/usr/bin/env python3
"""
World Labs Marble -> Isaac Sim Pipeline
========================================
Single image -> 3D world generation -> USD scene -> Isaac Sim headless render

Every stage caches its output.  Re-runs skip expensive work automatically.
Pass --no-cache to force a full re-run, or delete the output directory.

Usage:
    python marble_to_isaacsim.py path/to/image.jpg
    python marble_to_isaacsim.py path/to/image.jpg --skip-render
    python marble_to_isaacsim.py path/to/image.jpg --world-id <id>   # resume from existing world
    python marble_to_isaacsim.py path/to/image.jpg --no-cache         # ignore all caches

Requires:
    - WLT_API_KEY env var or --api-key flag
    - OMNI_KIT_ACCEPT_EULA=yes for Isaac Sim headless (set automatically)
"""

import argparse
import base64
import hashlib
import json
import logging
import math
import os
import sys
import time
from pathlib import Path

import numpy as np
import requests
import trimesh
from pxr import Gf, Sdf, Usd, UsdGeom, UsdLux, UsdShade, Vt

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("marble2isaacsim")

API_BASE = "https://api.worldlabs.ai/marble/v1"
CACHE_MANIFEST = ".cache_manifest.json"

VALID_MODELS = [
    "marble-1.0-draft",
    "marble-1.0",
    "marble-1.1",
    "marble-1.1-plus",
]


# ── Cache helpers ────────────────────────────────────────────────────────────


def _image_hash(path: Path) -> str:
    """Fast content hash of the input image for cache-key derivation."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 16), b""):
            h.update(chunk)
    return h.hexdigest()[:16]


def load_cache(out: Path) -> dict:
    p = out / CACHE_MANIFEST
    if p.exists():
        with open(p) as f:
            return json.load(f)
    return {}


def save_cache(out: Path, cache: dict):
    p = out / CACHE_MANIFEST
    with open(p, "w") as f:
        json.dump(cache, f, indent=2)


# ── World Labs Marble API ────────────────────────────────────────────────────


class MarbleClient:
    def __init__(self, api_key: str):
        self.session = requests.Session()
        self.session.headers.update(
            {"WLT-Api-Key": api_key, "Content-Type": "application/json"}
        )

    def _req(self, method: str, path: str, max_retries: int = 5, **kwargs):
        import time as _time
        url = f"{API_BASE}/{path}"
        for attempt in range(max_retries):
            resp = self.session.request(method, url, **kwargs)
            if resp.status_code == 429:
                wait = min(30, 2 ** attempt * 5)
                log.warning("Rate limited (429), retrying in %ds...", wait)
                _time.sleep(wait)
                continue
            if not resp.ok:
                body = resp.text[:500]
                raise RuntimeError(f"API {method} {path} → {resp.status_code}: {body}")
            return resp.json() if resp.content else {}
        raise RuntimeError(f"API {method} {path} → 429 after {max_retries} retries")

    def generate_from_image(
        self,
        image_path: str | None = None,
        text_prompt: str | None = None,
        model: str = "marble-1.1",
        display_name: str | None = None,
        seed: int | None = None,
    ) -> dict:
        if image_path and os.path.isfile(image_path):
            with open(image_path, "rb") as f:
                image_b64 = base64.b64encode(f.read()).decode()
            ext = Path(image_path).suffix.lstrip(".").lower()
            payload = {
                "display_name": display_name or Path(image_path).stem,
                "model": model,
                "world_prompt": {
                    "type": "image",
                    "image_prompt": {
                        "source": "data_base64",
                        "data_base64": image_b64,
                        "extension": ext,
                    },
                },
            }
            if text_prompt:
                payload["world_prompt"]["text_prompt"] = text_prompt
        elif text_prompt:
            payload = {
                "display_name": display_name or text_prompt[:40],
                "model": model,
                "world_prompt": {
                    "type": "text",
                    "text_prompt": text_prompt,
                },
            }
        else:
            raise ValueError("Either image_path or text_prompt is required")

        if seed is not None:
            payload["seed"] = seed

        return self._req("POST", "worlds:generate", json=payload)

    def get_operation(self, operation_id: str) -> dict:
        return self._req("GET", f"operations/{operation_id}")

    def get_world(self, world_id: str) -> dict:
        return self._req("GET", f"worlds/{world_id}")

    def wait_for_completion(
        self, operation_id: str, poll_sec: int = 10, timeout_sec: int = 600
    ) -> dict:
        t0 = time.time()
        while True:
            op = self.get_operation(operation_id)
            meta = op.get("metadata") or {}
            progress = meta.get("progress") or {}
            status = progress.get("status", "UNKNOWN")
            desc = progress.get("description", "")
            elapsed = int(time.time() - t0)
            log.info("  [%3ds] %s – %s", elapsed, status, desc)

            if op.get("done"):
                if op.get("error"):
                    raise RuntimeError(f"Generation failed: {op['error']}")
                return op

            if time.time() - t0 > timeout_sec:
                raise TimeoutError(
                    f"World generation timed out after {timeout_sec}s"
                )
            time.sleep(poll_sec)


# ── Asset download ───────────────────────────────────────────────────────────


def download_file(url: str, dest: Path) -> int:
    if dest.exists() and dest.stat().st_size > 0:
        log.info("  CACHED  %s  (%s bytes)", dest.name, f"{dest.stat().st_size:,}")
        return dest.stat().st_size

    resp = requests.get(url, stream=True, timeout=120)
    resp.raise_for_status()
    got = 0
    with open(dest, "wb") as f:
        for chunk in resp.iter_content(8192):
            f.write(chunk)
            got += len(chunk)
    log.info("  DOWNLOAD  %s  (%s bytes)", dest.name, f"{got:,}")
    return got


def download_world_assets(assets: dict, out: Path) -> dict[str, Path]:
    """Download all available assets, skipping files already on disk."""
    paths: dict[str, Path] = {}

    mesh = assets.get("mesh") or {}
    if mesh.get("collider_mesh_url"):
        p = out / "collider_mesh.glb"
        download_file(mesh["collider_mesh_url"], p)
        paths["collider_glb"] = p

    splats = assets.get("splats") or {}
    for variant, url in (splats.get("spz_urls") or {}).items():
        p = out / f"splat_{variant}.spz"
        download_file(url, p)
        paths[f"splat_{variant}"] = p

    imagery = assets.get("imagery") or {}
    if imagery.get("pano_url"):
        p = out / "panorama.jpg"
        download_file(imagery["pano_url"], p)
        paths["panorama"] = p

    if assets.get("thumbnail_url"):
        p = out / "thumbnail.jpg"
        download_file(assets["thumbnail_url"], p)
        paths["thumbnail"] = p

    return paths


# ── GLB → USD conversion (standalone pxr, no Isaac Sim needed) ───────────────


def glb_to_usd(
    glb_path: Path,
    usd_path: Path,
    scale_factor: float = 1.0,
    ground_offset: float = 0.0,
) -> Path:
    if usd_path.exists() and usd_path.stat().st_size > 0:
        log.info("CACHED  %s", usd_path)
        return usd_path

    log.info("Loading GLB mesh: %s", glb_path)
    loaded = trimesh.load(str(glb_path), force="scene")
    if isinstance(loaded, trimesh.Scene):
        mesh = loaded.to_geometry() if hasattr(loaded, "to_geometry") else loaded.dump(concatenate=True)
    else:
        mesh = loaded

    if not isinstance(mesh, trimesh.Trimesh):
        raise ValueError(f"Could not extract mesh from GLB (got {type(mesh).__name__})")

    log.info(
        "  Mesh: %d verts, %d faces, bounds %s",
        len(mesh.vertices),
        len(mesh.faces),
        mesh.bounds.tolist(),
    )

    stage = Usd.Stage.CreateNew(str(usd_path))
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.y)
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)

    UsdGeom.Xform.Define(stage, "/World")

    mesh_xform = UsdGeom.Xform.Define(stage, "/World/Mesh")
    if scale_factor != 1.0:
        s = scale_factor
        mesh_xform.AddScaleOp().Set(Gf.Vec3f(s, s, s))
    if ground_offset != 0.0:
        mesh_xform.AddTranslateOp().Set(Gf.Vec3d(0, -ground_offset, 0))

    mesh_prim = UsdGeom.Mesh.Define(stage, "/World/Mesh/Geometry")

    points = Vt.Vec3fArray([Gf.Vec3f(*v) for v in mesh.vertices])
    mesh_prim.GetPointsAttr().Set(points)
    mesh_prim.GetFaceVertexCountsAttr().Set(Vt.IntArray([3] * len(mesh.faces)))
    mesh_prim.GetFaceVertexIndicesAttr().Set(
        Vt.IntArray(mesh.faces.flatten().tolist())
    )

    if mesh.vertex_normals is not None and len(mesh.vertex_normals) > 0:
        normals = Vt.Vec3fArray([Gf.Vec3f(*n) for n in mesh.vertex_normals])
        mesh_prim.GetNormalsAttr().Set(normals)
        mesh_prim.SetNormalsInterpolation(UsdGeom.Tokens.vertex)

    has_colors = (
        mesh.visual
        and hasattr(mesh.visual, "vertex_colors")
        and mesh.visual.vertex_colors is not None
        and len(mesh.visual.vertex_colors) == len(mesh.vertices)
    )
    if has_colors:
        colors_f = mesh.visual.vertex_colors[:, :3].astype(np.float64) / 255.0
        primvars_api = UsdGeom.PrimvarsAPI(mesh_prim)
        primvar = primvars_api.CreatePrimvar(
            "displayColor", Sdf.ValueTypeNames.Color3fArray, UsdGeom.Tokens.vertex
        )
        primvar.Set(Vt.Vec3fArray([Gf.Vec3f(*c) for c in colors_f]))

    _add_basic_material(stage, mesh_prim, has_colors)

    stage.GetRootLayer().Save()
    log.info("  Saved USD: %s", usd_path)
    return usd_path


def _add_basic_material(stage, mesh_prim, has_vertex_colors: bool):
    mat_path = "/World/Mesh/Material"
    mat = UsdShade.Material.Define(stage, mat_path)
    shader = UsdShade.Shader.Define(stage, f"{mat_path}/Shader")
    shader.CreateIdAttr("UsdPreviewSurface")

    if has_vertex_colors:
        reader = UsdShade.Shader.Define(stage, f"{mat_path}/VertexColorReader")
        reader.CreateIdAttr("UsdPrimvarReader_float3")
        reader.CreateInput("varname", Sdf.ValueTypeNames.Token).Set("displayColor")
        reader.CreateOutput("result", Sdf.ValueTypeNames.Float3)

        shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).ConnectToSource(
            reader.ConnectableAPI(), "result"
        )
        shader.CreateInput("emissiveColor", Sdf.ValueTypeNames.Color3f).ConnectToSource(
            reader.ConnectableAPI(), "result"
        )
        shader.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(1.0)
        shader.CreateInput("metallic", Sdf.ValueTypeNames.Float).Set(0.0)
    else:
        shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(
            Gf.Vec3f(0.7, 0.7, 0.72)
        )
        shader.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(0.7)
        shader.CreateInput("metallic", Sdf.ValueTypeNames.Float).Set(0.0)

    mat.CreateSurfaceOutput().ConnectToSource(shader.ConnectableAPI(), "surface")
    UsdShade.MaterialBindingAPI(mesh_prim).Bind(mat)


# ── Isaac Sim headless rendering ─────────────────────────────────────────────


def build_lit_stage(stage, scene_center, scene_extent):
    """Add lights and ground plane to existing stage."""
    dome = UsdLux.DomeLight.Define(stage, "/World/Lights/DomeLight")
    dome.GetIntensityAttr().Set(500.0)

    sun = UsdLux.DistantLight.Define(stage, "/World/Lights/Sun")
    sun.GetIntensityAttr().Set(5000.0)
    sun.GetAngleAttr().Set(1.0)
    xf = UsdGeom.Xformable(sun)
    xf.AddRotateXYZOp().Set(Gf.Vec3f(-50.0, 35.0, 0.0))

    fill = UsdLux.DistantLight.Define(stage, "/World/Lights/Fill")
    fill.GetIntensityAttr().Set(1500.0)
    xf2 = UsdGeom.Xformable(fill)
    xf2.AddRotateXYZOp().Set(Gf.Vec3f(-20.0, -120.0, 0.0))

    gnd_size = max(scene_extent) * 4.0
    ground = UsdGeom.Mesh.Define(stage, "/World/GroundPlane")
    hs = gnd_size / 2.0
    ground.GetPointsAttr().Set(
        Vt.Vec3fArray(
            [
                Gf.Vec3f(-hs, 0, -hs),
                Gf.Vec3f(hs, 0, -hs),
                Gf.Vec3f(hs, 0, hs),
                Gf.Vec3f(-hs, 0, hs),
            ]
        )
    )
    ground.GetFaceVertexCountsAttr().Set(Vt.IntArray([4]))
    ground.GetFaceVertexIndicesAttr().Set(Vt.IntArray([0, 1, 2, 3]))
    ground.GetNormalsAttr().Set(Vt.Vec3fArray([Gf.Vec3f(0, 1, 0)] * 4))

    gnd_mat = UsdShade.Material.Define(stage, "/World/GroundPlane/Mat")
    gnd_sh = UsdShade.Shader.Define(stage, "/World/GroundPlane/Mat/Shader")
    gnd_sh.CreateIdAttr("UsdPreviewSurface")
    gnd_sh.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(
        Gf.Vec3f(0.25, 0.25, 0.25)
    )
    gnd_sh.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(0.9)
    gnd_mat.CreateSurfaceOutput().ConnectToSource(gnd_sh.ConnectableAPI(), "surface")
    UsdShade.MaterialBindingAPI(ground).Bind(gnd_mat)


def compute_orbit_cameras(
    center: tuple, extent: tuple, num_views: int, elevation_deg: float = 25.0
):
    """Return list of (position, look_at) for orbiting cameras."""
    max_dim = max(extent)
    radius = max_dim * 2.5
    elev_rad = math.radians(elevation_deg)
    y = center[1] + radius * math.sin(elev_rad)
    horiz_r = radius * math.cos(elev_rad)

    cameras = []
    for i in range(num_views):
        theta = 2.0 * math.pi * i / num_views
        x = center[0] + horiz_r * math.sin(theta)
        z = center[2] + horiz_r * math.cos(theta)
        cameras.append(((x, y, z), tuple(center)))
    return cameras


def render_in_isaacsim(
    usd_path: str,
    output_dir: Path,
    num_views: int = 4,
    resolution: tuple = (1920, 1080),
):
    os.environ.setdefault("OMNI_KIT_ACCEPT_EULA", "yes")

    log.info("Starting Isaac Sim (headless)...")
    from isaacsim import SimulationApp

    config = {
        "headless": True,
        "width": resolution[0],
        "height": resolution[1],
        "renderer": "RayTracedLighting",
        "anti_aliasing": 3,
        "sync_loads": True,
        "fast_shutdown": True,
    }
    app = SimulationApp(launch_config=config)

    try:
        import omni.usd

        context = omni.usd.get_context()

        log.info("Opening stage: %s", usd_path)
        result, err = context.open_stage(usd_path)
        if not result:
            raise RuntimeError(f"Failed to open USD stage: {err}")

        for _ in range(30):
            app.update()

        stage = context.get_stage()

        bbox_cache = UsdGeom.BBoxCache(
            Usd.TimeCode.Default(), [UsdGeom.Tokens.default_]
        )
        root = stage.GetPrimAtPath("/World")
        if not root.IsValid():
            root = stage.GetPseudoRoot()
        bbox = bbox_cache.ComputeWorldBound(root).ComputeAlignedBox()
        bmin = np.array([bbox.GetMin()[i] for i in range(3)])
        bmax = np.array([bbox.GetMax()[i] for i in range(3)])
        center = ((bmin + bmax) / 2.0).tolist()
        extent = (bmax - bmin).tolist()
        log.info("  Scene center=%s  extent=%s", center, extent)

        build_lit_stage(stage, center, extent)
        stage.Save()

        camera_poses = compute_orbit_cameras(center, extent, num_views)

        try:
            import omni.replicator.core as rep

            for i, (pos, target) in enumerate(camera_poses):
                cam = rep.create.camera(
                    position=pos,
                    look_at=target,
                    focal_length=28.0,
                    name=f"OrbitCam_{i}",
                )
                rp = rep.create.render_product(cam, resolution)

                frame_dir = str(output_dir / f"view_{i:02d}")
                writer = rep.WriterRegistry.get("BasicWriter")
                writer.initialize(output_dir=frame_dir, rgb=True, frame_padding=4)
                writer.attach([rp])

                for _ in range(30):
                    app.update()
                rep.orchestrator.step()
                app.update()
                app.update()

                log.info(
                    "  Rendered view %d/%d – camera at (%.1f, %.1f, %.1f)",
                    i + 1,
                    num_views,
                    *pos,
                )

            log.info("Renders saved to: %s", output_dir)

        except Exception as e:
            log.warning("Replicator render failed (%s), trying viewport capture...", e)
            _fallback_viewport_render(app, stage, camera_poses, output_dir)

    finally:
        app.close()
        log.info("Isaac Sim closed.")


def _fallback_viewport_render(app, stage, camera_poses, output_dir):
    """Fallback: bake cameras into stage so user can open in GUI later."""
    for i, (pos, _target) in enumerate(camera_poses):
        cam_path = f"/World/Cameras/Cam_{i}"
        cam = UsdGeom.Camera.Define(stage, cam_path)
        xf = UsdGeom.Xformable(cam)
        xf.AddTranslateOp().Set(Gf.Vec3d(*pos))

        for _ in range(20):
            app.update()

        log.info("  Camera %d placed at %s (viewport capture may be limited)", i, pos)

    stage.Save()
    enhanced_usd = output_dir / "scene_with_cameras.usd"
    stage.GetRootLayer().Export(str(enhanced_usd))
    log.info("  Saved enhanced stage: %s", enhanced_usd)


# ── Main pipeline with caching ───────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser(
        description="World Labs Marble → Isaac Sim: image to 3D render pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("image", nargs="?", default=None, help="Input image (jpg/png/webp)")
    parser.add_argument("--api-key", default=os.environ.get("WLT_API_KEY"))
    parser.add_argument("--model", default="marble-1.1", choices=VALID_MODELS)
    parser.add_argument("--text-prompt", default=None, help="Optional text guidance")
    parser.add_argument("--output-dir", default="output/marble_pipeline")
    parser.add_argument("--skip-render", action="store_true", help="Stop after USD conversion")
    parser.add_argument("--resolution", default="1920x1080")
    parser.add_argument("--num-views", type=int, default=4)
    parser.add_argument("--timeout", type=int, default=600, help="API generation timeout (sec)")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--world-id", default=None, help="Resume from existing world (skip generation)")
    parser.add_argument("--no-cache", action="store_true", help="Ignore all cached results")
    args = parser.parse_args()

    if not args.api_key:
        log.error("API key required: set WLT_API_KEY or pass --api-key")
        sys.exit(1)

    image_path = Path(args.image) if args.image and args.image != "__text_only__" else None
    if image_path and not image_path.exists():
        log.error("Image not found: %s", image_path)
        sys.exit(1)
    if not image_path and not args.text_prompt:
        log.error("Either an image or --text-prompt is required")
        sys.exit(1)

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    w, h = map(int, args.resolution.split("x"))

    cache = {} if args.no_cache else load_cache(out)
    if image_path:
        img_hash = _image_hash(image_path)
    else:
        import hashlib as _hl
        img_hash = _hl.md5(args.text_prompt.encode()).hexdigest()[:16]
    cache_key = f"{img_hash}_{args.model}_{args.seed}"

    client = MarbleClient(args.api_key)

    # ── Stage 1: Generate world (or load from cache) ─────────────────────
    log.info("=" * 64)
    log.info("STAGE 1 — Generate 3D world from image (Marble API)")
    log.info("=" * 64)

    world_id = args.world_id
    meta_file = out / "world_metadata.json"

    if world_id:
        log.info("Using provided --world-id: %s", world_id)
    elif cache.get("cache_key") == cache_key and cache.get("world_id"):
        world_id = cache["world_id"]
        log.info("CACHED  world_id=%s  (image hash %s matches)", world_id, img_hash)
    else:
        log.info("Image:  %s  (hash=%s)", image_path, img_hash)
        log.info("Model:  %s", args.model)
        op = client.generate_from_image(
            image_path=str(image_path) if image_path else None,
            text_prompt=args.text_prompt,
            model=args.model,
            seed=args.seed,
        )
        op_id = op["operation_id"]
        log.info("Operation: %s", op_id)
        log.info("Polling (this takes ~5 min)...")

        result = client.wait_for_completion(op_id, timeout_sec=args.timeout)
        world_snapshot = result["response"]
        world_id = world_snapshot.get("world_id") or world_snapshot.get("id")
        log.info("World ready: %s", world_id)

        with open(meta_file, "w") as f:
            json.dump(result, f, indent=2)

        cache["cache_key"] = cache_key
        cache["world_id"] = world_id
        cache["operation_id"] = op_id
        save_cache(out, cache)

    # ── Stage 2: Download assets (skip files already on disk) ────────────
    log.info("=" * 64)
    log.info("STAGE 2 — Download 3D assets")
    log.info("=" * 64)

    world_full_file = out / "world_full.json"
    cached_world_ok = False
    if not args.no_cache and world_full_file.exists():
        with open(world_full_file) as f:
            world_data = json.load(f)
        stored_id = world_data.get("world_id") or world_data.get("id")
        if stored_id == world_id:
            cached_world_ok = True
            log.info("CACHED  world details for %s", world_id)

    if not cached_world_ok:
        full = client.get_world(world_id)
        world_data = full.get("world", full)
        with open(world_full_file, "w") as f:
            json.dump(world_data, f, indent=2)
        log.info("Fetched world details for %s", world_id)

    assets = world_data.get("assets") or {}
    asset_paths = download_world_assets(assets, out)

    sem = (assets.get("splats") or {}).get("semantics_metadata") or {}
    scale_factor = sem.get("metric_scale_factor", 1.0)
    ground_offset = sem.get("ground_plane_offset", 0.0)
    log.info("Semantics: scale=%.4f  ground_offset=%.4f", scale_factor, ground_offset)

    cache["assets_downloaded"] = True
    save_cache(out, cache)

    # ── Stage 3: GLB → USD (skip if USD already exists) ──────────────────
    log.info("=" * 64)
    log.info("STAGE 3 — Convert collider mesh to USD")
    log.info("=" * 64)

    glb = asset_paths.get("collider_glb")
    usd_path = out / "scene.usd"

    if not args.no_cache and usd_path.exists() and usd_path.stat().st_size > 0:
        log.info("CACHED  %s  (%s bytes)", usd_path.name, f"{usd_path.stat().st_size:,}")
    elif glb and glb.exists():
        glb_to_usd(
            glb,
            usd_path,
            scale_factor=scale_factor or 1.0,
            ground_offset=ground_offset or 0.0,
        )
        cache["usd_converted"] = True
        save_cache(out, cache)
    else:
        log.warning("No collider mesh available — cannot create USD scene")
        usd_path = None

    # ── Stage 4: Isaac Sim render (skip if renders exist) ────────────────
    render_dir = out / "renders"

    if not args.skip_render and usd_path and usd_path.exists():
        existing_renders = list(render_dir.rglob("*.png")) if render_dir.exists() else []
        if not args.no_cache and len(existing_renders) >= args.num_views:
            log.info("=" * 64)
            log.info("STAGE 4 — Isaac Sim headless render")
            log.info("=" * 64)
            log.info("CACHED  %d render(s) already in %s", len(existing_renders), render_dir)
        else:
            log.info("=" * 64)
            log.info("STAGE 4 — Isaac Sim headless render")
            log.info("=" * 64)
            render_dir.mkdir(exist_ok=True)
            try:
                render_in_isaacsim(
                    usd_path=str(usd_path),
                    output_dir=render_dir,
                    num_views=args.num_views,
                    resolution=(w, h),
                )
                cache["rendered"] = True
                save_cache(out, cache)
            except Exception as e:
                log.error("Isaac Sim render failed: %s", e, exc_info=True)
                log.info("USD scene is still available at: %s", usd_path)
    elif args.skip_render:
        log.info("Skipping Isaac Sim render (--skip-render)")
    else:
        log.warning("No USD scene to render")

    # ── Summary ──────────────────────────────────────────────────────────
    log.info("=" * 64)
    log.info("PIPELINE COMPLETE")
    log.info("=" * 64)
    log.info("World ID:    %s", world_id)
    log.info("Marble URL:  %s", world_data.get("world_marble_url", "n/a"))
    log.info("Output dir:  %s", out.resolve())
    log.info("Files:")
    for p in sorted(out.rglob("*")):
        if p.is_file() and p.name != CACHE_MANIFEST:
            log.info("  %-40s  %s bytes", p.relative_to(out), f"{p.stat().st_size:,}")


if __name__ == "__main__":
    main()
