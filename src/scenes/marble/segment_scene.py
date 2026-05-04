#!/usr/bin/env python3
"""
Multi-view 3D instance segmentation pipeline for Marble scenes.

Pipeline:
  1. Render 12 multi-view RGB + depth images using gsplat
  2. Configured VLM: extract object list from the generated-world caption
  3. Configured VLM: detect objects in each rendered view
  4. SAM3 PCS: segment each named object in each view
  5. Back-project 2D masks to 3D using depth + camera params
  6. Post-process: merge cross-view, resolve overlaps
  7. Panorama verification for coverage
  8. Output SAGE-compatible scene.json
"""

import sys, os, json, base64, math, time
import numpy as np
import torch
from pathlib import Path
from PIL import Image
import requests as http_requests

# ── Config ───────────────────────────────────────────────────────────────────

VLM_URL = os.environ.get("VLM_API_BASE", "https://example.com/v1/chat/completions")
VLM_MODEL = os.environ.get("VLM_MODEL", "Qwen3-VL-235B-A22B-Thinking")
VLM_API_KEY = os.environ.get("VLM_API_KEY", "")

HFOV_DEG = 82.0
NUM_VIEWS = 12
ELEVATION_DEG = -15.0
RENDER_W, RENDER_H = 1296, 968

# ── 1. Multi-view rendering ─────────────────────────────────────────────────

def load_ply_gaussians(ply_path: str):
    from plyfile import PlyData
    ply = PlyData.read(ply_path)
    v = ply['vertex']
    n = len(v)
    print(f"  Loaded {n:,} gaussians from {ply_path}")
    xyz = np.stack([v['x'], v['y'], v['z']], axis=-1).astype(np.float32)
    C0 = 0.28209479177387814
    colors = np.clip(np.stack([
        0.5 + C0 * v['f_dc_0'].astype(np.float32),
        0.5 + C0 * v['f_dc_1'].astype(np.float32),
        0.5 + C0 * v['f_dc_2'].astype(np.float32),
    ], axis=-1), 0, 1)
    opacity = 1.0 / (1.0 + np.exp(-v['opacity'].astype(np.float32)))
    scales = np.exp(np.stack([v['scale_0'], v['scale_1'], v['scale_2']], axis=-1).astype(np.float32))
    quats = np.stack([v['rot_0'], v['rot_1'], v['rot_2'], v['rot_3']], axis=-1).astype(np.float32)
    quats = quats / (np.linalg.norm(quats, axis=-1, keepdims=True) + 1e-8)
    return {k: torch.tensor(v, device='cuda') for k, v in
            [('means', xyz), ('colors', colors), ('opacities', opacity),
             ('scales', scales), ('quats', quats)]}


def look_at_matrix(eye, target, up=None):
    if up is None:
        up = np.array([0, 1, 0], dtype=np.float32)
    eye, target, up = [np.asarray(v, dtype=np.float32) for v in [eye, target, up]]
    fwd = target - eye
    fwd /= np.linalg.norm(fwd) + 1e-8
    right = np.cross(fwd, up)
    right /= np.linalg.norm(right) + 1e-8
    new_up = np.cross(right, fwd)
    c2w = np.eye(4, dtype=np.float32)
    c2w[:3, 0] = right
    c2w[:3, 1] = new_up
    c2w[:3, 2] = -fwd
    c2w[:3, 3] = eye
    return torch.tensor(c2w, device='cuda')


def render_view_rgbd(gaussians, W, H, fx, fy, cx, cy, c2w):
    from gsplat import rasterization
    viewmat = torch.linalg.inv(c2w).unsqueeze(0)
    K = torch.tensor([[fx, 0, cx], [0, fy, cy], [0, 0, 1]],
                     dtype=torch.float32, device='cuda').unsqueeze(0)

    renders, alphas, meta = rasterization(
        means=gaussians['means'], quats=gaussians['quats'],
        scales=gaussians['scales'], opacities=gaussians['opacities'],
        colors=gaussians['colors'], viewmats=viewmat, Ks=K,
        width=W, height=H, sh_degree=None, render_mode='RGB+ED',
    )
    rgbd = renders[0]  # [H, W, 4]
    rgb = rgbd[:, :, :3].clamp(0, 1)
    depth = rgbd[:, :, 3:4]  # expected depth
    alpha = alphas[0]
    bg = torch.ones(1, 1, 3, device='cuda')
    rgb = rgb * alpha + bg * (1 - alpha)
    return rgb.cpu().numpy(), depth.cpu().numpy().squeeze(-1), alpha.cpu().numpy()


def render_multiview(gaussians, out_dir):
    """Render NUM_VIEWS evenly-spaced views + depth maps."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    W, H = RENDER_W, RENDER_H
    cx, cy = W / 2.0, H / 2.0
    fx = fy = (W / 2.0) / math.tan(math.radians(HFOV_DEG / 2.0))
    eye = np.array([0, 0, 0], dtype=np.float32)
    up = np.array([0, 1, 0], dtype=np.float32)

    cameras = []
    for i in range(NUM_VIEWS):
        azimuth = i * (360.0 / NUM_VIEWS)
        az_rad = math.radians(azimuth)
        el_rad = math.radians(ELEVATION_DEG)
        tx = math.sin(az_rad) * math.cos(el_rad)
        ty = math.sin(el_rad)
        tz = -math.cos(az_rad) * math.cos(el_rad)
        target = np.array([tx * 10, ty * 10, tz * 10], dtype=np.float32)

        c2w = look_at_matrix(eye, target, up)
        rgb, depth, alpha = render_view_rgbd(gaussians, W, H, fx, fy, cx, cy, c2w)

        name = f"view_{i:03d}"
        img_path = out_dir / f"{name}.png"
        depth_path = out_dir / f"{name}_depth.npy"

        Image.fromarray((rgb * 255).astype(np.uint8)).save(str(img_path))
        np.save(str(depth_path), depth)

        cam = {
            "name": name, "index": i, "azimuth_deg": azimuth,
            "elevation_deg": ELEVATION_DEG,
            "position": eye.tolist(),
            "look_at": target.tolist(),
            "c2w": c2w.cpu().numpy().tolist(),
            "fx": fx, "fy": fy, "cx": cx, "cy": cy,
            "width": W, "height": H,
            "horizontal_fov_deg": HFOV_DEG,
            "rgb_path": str(img_path),
            "depth_path": str(depth_path),
        }
        cameras.append(cam)
        print(f"  [{i+1}/{NUM_VIEWS}] {name} az={azimuth:.0f}° alpha={alpha.mean():.1%}")

    json.dump(cameras, open(out_dir / "cameras.json", "w"), indent=2)
    return cameras


# ── 2. VLM: extract objects from caption ────────────────────────────────────

def vlm_chat(messages, temperature=0.3, max_tokens=1024):
    payload = {
        "model": VLM_MODEL,
        "messages": messages,
        "temperature": temperature,
        "max_completion_tokens": max_tokens,
    }
    resp = http_requests.post(VLM_URL, headers={
        "Authorization": f"Bearer {VLM_API_KEY}",
        "Content-Type": "application/json",
    }, json=payload, timeout=120)
    if resp.status_code != 200:
        print(f"  VLM API error {resp.status_code}: {resp.text[:300]}")
    resp.raise_for_status()
    return resp.json()["choices"][0]["message"]["content"]


def vlm_vision(image_path, prompt, temperature=0.3, max_tokens=1024):
    with open(image_path, "rb") as f:
        b64 = base64.b64encode(f.read()).decode()
    ext = Path(image_path).suffix.lstrip('.').lower()
    mime = {"jpg": "jpeg", "jpeg": "jpeg", "png": "png"}.get(ext, "png")
    messages = [{"role": "user", "content": [
        {"type": "text", "text": prompt},
        {"type": "image_url", "image_url": {"url": f"data:image/{mime};base64,{b64}"}},
    ]}]
    return vlm_chat(messages, temperature, max_tokens)


def extract_objects_from_caption(caption):
    prompt = f"""Extract all distinct physical objects from this scene description.
Return ONLY a JSON array of short object names (lowercase, singular form).
Include every object mentioned, no matter how small.
Do NOT include materials, colors, or room-level concepts (like "walls", "floor", "ceiling").

Scene description:
{caption}

Example output: ["sink", "towel", "mirror", "faucet", "soap dispenser"]"""

    result = vlm_chat([
        {"role": "system", "content": "You extract object names from text. Always return valid JSON."},
        {"role": "user", "content": prompt},
    ])
    try:
        start = result.index('[')
        end = result.rindex(']') + 1
        return json.loads(result[start:end])
    except (ValueError, json.JSONDecodeError):
        print(f"  WARNING: Could not parse VLM response: {result[:200]}")
        return []


# ── 3. VLM: detect objects in each view ─────────────────────────────────────

def detect_objects_in_view(image_path):
    prompt = """List every distinct physical object you can see in this image.
Return ONLY a JSON array of short object names (lowercase, singular form).
Be exhaustive - include every object no matter how small or partially visible.
Do NOT include room surfaces (walls, floor, ceiling).

Example: ["sink", "towel", "mirror", "faucet", "trash can", "soap bar"]"""

    result = vlm_vision(image_path, prompt)
    try:
        start = result.index('[')
        end = result.rindex(']') + 1
        return json.loads(result[start:end])
    except (ValueError, json.JSONDecodeError):
        print(f"  WARNING: Could not parse VLM vision response: {result[:200]}")
        return []


# ── 4. Merge object lists ───────────────────────────────────────────────────

def merge_object_lists(caption_objects, view_objects_per_view):
    """Deduplicate and normalize across all sources."""
    all_names = set()
    for name in caption_objects:
        all_names.add(name.lower().strip())
    for view_objs in view_objects_per_view:
        for name in view_objs:
            all_names.add(name.lower().strip())
    # Remove very generic or room-level terms
    skip = {"wall", "floor", "ceiling", "room", "bathroom", "scene", "space",
            "light", "lighting", "air", "shadow"}
    merged = sorted([n for n in all_names if n and n not in skip])
    return merged


# ── 5. SAM3 segmentation ────────────────────────────────────────────────────

import threading

_sam3_cache: dict = {}
_sam3_lock = threading.Lock()


def _load_sam3():
    device = torch.cuda.current_device()
    with _sam3_lock:
        if device in _sam3_cache:
            return _sam3_cache[device]
    print(f"  Loading SAM3 model on cuda:{device}...")
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.autocast("cuda", dtype=torch.bfloat16).__enter__()
    from sam3.model_builder import build_sam3_image_model
    from sam3.model.sam3_image_processor import Sam3Processor
    model = build_sam3_image_model()
    model = model.to(f"cuda:{device}")
    processor = Sam3Processor(model)
    with _sam3_lock:
        _sam3_cache[device] = (model, processor)
    print(f"  SAM3 ready on cuda:{device}.")
    return model, processor


def segment_objects_in_view(image_path, object_names):
    """Run SAM3 PCS for each object name, return list of detections."""
    model, processor = _load_sam3()
    image = Image.open(image_path).convert("RGB")
    state = processor.set_image(image)

    detections = []
    for obj_name in object_names:
        processor.reset_all_prompts(state)
        try:
            output = processor.set_text_prompt(prompt=obj_name, state=state)
        except Exception as e:
            print(f"    SAM3 failed for '{obj_name}': {e}")
            continue

        masks = output.get("masks")
        boxes = output.get("boxes")
        scores = output.get("scores")

        if masks is None or len(masks) == 0:
            continue

        for idx in range(len(masks)):
            score = float(scores[idx]) if scores is not None else 0.0
            if score < 0.3:
                continue
            mask_np = masks[idx].cpu().numpy() if torch.is_tensor(masks[idx]) else np.array(masks[idx])
            if mask_np.ndim == 3:
                mask_np = mask_np[0]
            mask_bool = mask_np > 0.5

            box = boxes[idx].cpu().numpy() if torch.is_tensor(boxes[idx]) else np.array(boxes[idx])

            detections.append({
                "label": obj_name,
                "score": score,
                "mask": mask_bool,
                "box": box.tolist(),
                "instance_idx": idx,
            })

    return detections


# ── 6. Back-project to 3D ───────────────────────────────────────────────────

def backproject_mask_to_3d(mask, depth, cam):
    """Back-project a 2D binary mask to 3D points using depth and camera params."""
    ys, xs = np.where(mask)
    if len(xs) == 0:
        return np.zeros((0, 3), dtype=np.float32)

    fx, fy = cam["fx"], cam["fy"]
    cx, cy = cam["cx"], cam["cy"]
    c2w = np.array(cam["c2w"], dtype=np.float32)

    d = depth[ys, xs]
    valid = (d > 0.01) & (d < 100.0)
    xs, ys, d = xs[valid], ys[valid], d[valid]
    if len(xs) == 0:
        return np.zeros((0, 3), dtype=np.float32)

    # Subsample if too many points for efficiency
    if len(xs) > 5000:
        idx = np.random.choice(len(xs), 5000, replace=False)
        xs, ys, d = xs[idx], ys[idx], d[idx]

    x_cam = (xs - cx) / fx * d
    y_cam = (ys - cy) / fy * d
    z_cam = d

    pts_cam = np.stack([x_cam, y_cam, z_cam, np.ones_like(x_cam)], axis=-1)  # [N, 4]
    pts_world = (c2w @ pts_cam.T).T[:, :3]
    return pts_world.astype(np.float32)


def compute_3d_bbox(pts):
    """Compute axis-aligned 3D bounding box from point cloud."""
    if len(pts) == 0:
        return None
    center = pts.mean(axis=0)
    mins = pts.min(axis=0)
    maxs = pts.max(axis=0)
    dims = maxs - mins
    return {
        "center": center.tolist(),
        "min": mins.tolist(),
        "max": maxs.tolist(),
        "width": float(dims[0]),
        "length": float(dims[2]),  # Z-axis for depth
        "height": float(dims[1]),  # Y-axis for height (Y-up)
    }


# ── 7. Cross-view merging ───────────────────────────────────────────────────

def iou_3d_aabb(box_a, box_b):
    """Compute 3D IoU between two axis-aligned bounding boxes."""
    a_min, a_max = np.array(box_a["min"]), np.array(box_a["max"])
    b_min, b_max = np.array(box_b["min"]), np.array(box_b["max"])
    inter_min = np.maximum(a_min, b_min)
    inter_max = np.minimum(a_max, b_max)
    inter_dims = np.maximum(0, inter_max - inter_min)
    inter_vol = inter_dims.prod()
    a_vol = np.maximum(0, a_max - a_min).prod()
    b_vol = np.maximum(0, b_max - b_min).prod()
    union_vol = a_vol + b_vol - inter_vol
    if union_vol < 1e-8:
        return 0.0
    return inter_vol / union_vol


def merge_cross_view_detections(all_view_detections, iou_threshold=0.2):
    """Merge detections from multiple views into unique 3D objects."""
    objects_3d = []

    for det in all_view_detections:
        if det["bbox3d"] is None:
            continue

        merged = False
        for obj in objects_3d:
            if obj["label"] != det["label"]:
                continue
            iou = iou_3d_aabb(obj["bbox3d"], det["bbox3d"])
            if iou > iou_threshold:
                obj["points"].append(det["points"])
                obj["view_count"] += 1
                obj["scores"].append(det["score"])
                merged = True
                break

        if not merged:
            objects_3d.append({
                "label": det["label"],
                "points": [det["points"]],
                "bbox3d": det["bbox3d"],
                "view_count": 1,
                "scores": [det["score"]],
            })

    # Recompute bboxes from merged points
    for obj in objects_3d:
        all_pts = np.concatenate(obj["points"], axis=0)
        obj["bbox3d"] = compute_3d_bbox(all_pts)
        obj["avg_score"] = float(np.mean(obj["scores"]))
        del obj["points"]
        del obj["scores"]

    # Sort by view_count (more views = more confident)
    objects_3d.sort(key=lambda o: (-o["view_count"], -o["avg_score"]))
    return objects_3d


# ── 8. Panorama verification ────────────────────────────────────────────────

def panorama_verify_coverage(panorama_path, detected_objects):
    """Use the configured VLM on the panorama to check for missed objects."""
    detected_labels = set(o["label"] for o in detected_objects)
    prompt = f"""I have already detected these objects in a 3D scene: {json.dumps(sorted(detected_labels))}

Look at this 360-degree panorama of the same scene. List any additional physical objects
that I MISSED. Return ONLY a JSON array of object names not in my list.
If I haven't missed anything, return an empty array [].

Be thorough - check for small items, fixtures, and partially visible objects."""

    result = vlm_vision(panorama_path, prompt)
    try:
        start = result.index('[')
        end = result.rindex(']') + 1
        missed = json.loads(result[start:end])
        return [m.lower().strip() for m in missed if m.lower().strip() not in detected_labels]
    except (ValueError, json.JSONDecodeError):
        return []


# ── 9. Output scene.json ────────────────────────────────────────────────────

def build_scene_json(objects_3d, cameras, scene_id, world_json_path=None):
    scene_objects = []
    for i, obj in enumerate(objects_3d):
        bbox = obj["bbox3d"]
        if bbox is None:
            continue
        scene_objects.append({
            "id": f"{obj['label']}_{i}",
            "label": obj["label"],
            "type": obj["label"],
            "x": bbox["center"][0],
            "y": bbox["center"][1],
            "z": bbox["center"][2] - bbox["height"] / 2,  # bottom of object
            "rotation_z": 0.0,
            "width": bbox["width"],
            "length": bbox["length"],
            "height": bbox["height"],
            "view_count": obj["view_count"],
            "confidence": obj["avg_score"],
        })

    scene_cameras = []
    for cam in cameras:
        scene_cameras.append({
            "name": cam["name"],
            "position": cam["position"],
            "look_at": cam["look_at"],
            "horizontal_fov_deg": cam["horizontal_fov_deg"],
        })

    return {
        "scene_id": scene_id,
        "source": "marble",
        "world_json": world_json_path,
        "objects": scene_objects,
        "cameras": scene_cameras,
    }


# ── Main pipeline ───────────────────────────────────────────────────────────

def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--marble-dir", default="output/marble_test",
                    help="Directory with Marble outputs (scene.ply, world_full.json, panorama.jpg)")
    ap.add_argument("--output-dir", default=None,
                    help="Output directory (default: <marble-dir>/segmentation)")
    ap.add_argument("--skip-render", action="store_true",
                    help="Skip rendering if views already exist")
    ap.add_argument("--skip-sam", action="store_true",
                    help="Skip SAM3 segmentation (for testing VLM steps only)")
    args = ap.parse_args()

    marble_dir = Path(args.marble_dir)
    out_dir = Path(args.output_dir) if args.output_dir else marble_dir / "segmentation"
    out_dir.mkdir(parents=True, exist_ok=True)
    views_dir = out_dir / "views"

    ply_path = marble_dir / "scene.ply"
    world_json_path = marble_dir / "world_full.json"
    panorama_path = marble_dir / "panorama.jpg"

    # Load world metadata
    world = json.load(open(world_json_path))
    caption = world.get("assets", {}).get("caption", "")
    scene_id = world.get("world_id", "unknown")[:12]
    print(f"Scene: {scene_id}")
    print(f"Caption: {caption[:120]}...")

    # ── Step 1+2: Render multi-view + extract caption objects (parallel) ─
    cameras_json = views_dir / "cameras.json"
    from concurrent.futures import ThreadPoolExecutor, as_completed

    caption_future = None
    if args.skip_render and cameras_json.exists():
        print("\n[1/8] Skipping render (--skip-render, cameras.json exists)")
        cameras = json.load(open(cameras_json))
    else:
        print(f"\n[1/8] Rendering {NUM_VIEWS} views + extracting caption objects (parallel)...")
        _caption_pool = ThreadPoolExecutor(max_workers=1)
        caption_future = _caption_pool.submit(extract_objects_from_caption, caption)
        gaussians = load_ply_gaussians(str(ply_path))
        cameras = render_multiview(gaussians, views_dir)
        del gaussians
        torch.cuda.empty_cache()
    print(f"  {len(cameras)} views rendered.")

    if caption_future is not None:
        caption_objects = caption_future.result()
        _caption_pool.shutdown(wait=False)
        print(f"\n[2/8] Caption objects (done in parallel): {caption_objects}")
    else:
        print("\n[2/8] Extracting objects from caption via configured VLM...")
        caption_objects = extract_objects_from_caption(caption)
        print(f"  Caption objects: {caption_objects}")

    # ── Step 3: Detect objects in each view via configured VLM (parallel)
    print(f"\n[3/8] Detecting objects in {len(cameras)} views via configured VLM (parallel)...")

    def _detect_one(cam):
        return cam["name"], detect_objects_in_view(cam["rgb_path"])

    view_objects = [None] * len(cameras)
    with ThreadPoolExecutor(max_workers=len(cameras)) as pool:
        futures = {pool.submit(_detect_one, cam): i for i, cam in enumerate(cameras)}
        for fut in as_completed(futures):
            idx = futures[fut]
            name, objs = fut.result()
            view_objects[idx] = objs
            print(f"  {name}: found {len(objs)}: {objs[:5]}{'...' if len(objs)>5 else ''}")

    # ── Step 4: Merge object lists ───────────────────────────────────────
    print("\n[4/8] Merging object lists...")
    all_object_names = merge_object_lists(caption_objects, view_objects)
    print(f"  Merged unique objects ({len(all_object_names)}): {all_object_names}")

    # Save intermediate
    json.dump({
        "caption_objects": caption_objects,
        "view_objects": view_objects,
        "merged_objects": all_object_names,
    }, open(out_dir / "object_lists.json", "w"), indent=2)

    if args.skip_sam:
        print("\n[5-8] Skipping SAM3 + 3D steps (--skip-sam)")
        return

    # ── Step 5: SAM3 segmentation ────────────────────────────────────────
    print(f"\n[5/8] Running SAM3 segmentation on {len(cameras)} views...")
    all_view_detections = []

    for cam in cameras:
        dets = segment_objects_in_view(cam["rgb_path"], all_object_names)
        depth = np.load(cam["depth_path"])
        view_dets = []
        for det in dets:
            pts = backproject_mask_to_3d(det["mask"], depth, cam)
            bbox3d = compute_3d_bbox(pts) if len(pts) > 10 else None
            n_pixels = int(det["mask"].sum())
            view_dets.append({
                "label": det["label"],
                "score": det["score"],
                "view": cam["name"],
                "bbox3d": bbox3d,
                "points": pts,
                "n_pixels": n_pixels,
            })
        print(f"  {cam['name']}: {len(view_dets)} detections")
        all_view_detections.extend(view_dets)

    # ── Step 6: Cross-view merging ───────────────────────────────────────
    print(f"\n[6/8] Merging {len(all_view_detections)} detections across views...")
    objects_3d = merge_cross_view_detections(all_view_detections)
    print(f"  Merged into {len(objects_3d)} unique 3D objects:")
    for obj in objects_3d:
        b = obj["bbox3d"]
        print(f"    {obj['label']}: seen in {obj['view_count']} views, "
              f"center=({b['center'][0]:.2f},{b['center'][1]:.2f},{b['center'][2]:.2f}), "
              f"size=({b['width']:.2f}x{b['length']:.2f}x{b['height']:.2f})")

    # ── Step 7: Panorama verification ────────────────────────────────────
    print("\n[7/8] Checking panorama for missed objects...")
    if panorama_path.exists():
        missed = panorama_verify_coverage(str(panorama_path), objects_3d)
        if missed:
            print(f"  Missed objects found: {missed}")
            print("  Running SAM3 on panorama for missed objects...")
            pano_dets = segment_objects_in_view(str(panorama_path), missed)
            # For panorama detections, we can't easily back-project
            # (equirectangular has no single depth map), so just note them
            for det in pano_dets:
                objects_3d.append({
                    "label": det["label"],
                    "bbox3d": {"center": [0, 0, 0], "min": [0, 0, 0], "max": [0, 0, 0],
                               "width": 0.5, "length": 0.5, "height": 0.5},
                    "view_count": 1,
                    "avg_score": det["score"],
                    "source": "panorama_only",
                })
                print(f"    Added from panorama: {det['label']} (score={det['score']:.2f})")
        else:
            print("  No missed objects detected.")
    else:
        print("  Panorama not found, skipping.")

    # ── Step 8: Output scene.json ────────────────────────────────────────
    print("\n[8/8] Writing scene.json...")
    scene = build_scene_json(objects_3d, cameras, scene_id, str(world_json_path))
    scene_path = out_dir / "scene.json"
    json.dump(scene, open(scene_path, "w"), indent=2)
    print(f"  Saved to {scene_path}")
    print(f"  {len(scene['objects'])} objects, {len(scene['cameras'])} cameras")

    for obj in scene["objects"]:
        print(f"    {obj['id']}: label={obj['label']}, "
              f"pos=({obj['x']:.2f},{obj['y']:.2f},{obj['z']:.2f}), "
              f"size=({obj['width']:.2f}x{obj['length']:.2f}x{obj['height']:.2f}), "
              f"views={obj['view_count']}, conf={obj['confidence']:.2f}")

    print("\nDone!")


if __name__ == "__main__":
    main()
