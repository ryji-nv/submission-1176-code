#!/usr/bin/env python3
"""
3D Gaussian Instance Labeling Pipeline.

Assigns instance-level labels to each Gaussian in the splat cloud:
  1. Run SAM3 on multi-view renders -> per-view indexed instance masks
  2. Project all Gaussians into each view's pixel space
  3. Look up mask labels, depth-test to reject occluded votes
  4. Majority-vote to assign ONE category label per Gaussian
  5. DBSCAN spatial clustering within each category -> instance IDs
  6. Render labeled Gaussians via gsplat for 3D-consistent visualization
  7. Output gaussian_labels.npy, instance_info.json, scene.json, viz/
"""

import sys, os, json, math, hashlib
sys.path.insert(0, os.path.dirname(__file__))
import numpy as np
import torch
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont
from collections import Counter, defaultdict

from segment_scene import (
    load_ply_gaussians, look_at_matrix, render_multiview,
    vlm_chat, vlm_vision, extract_objects_from_caption,
    detect_objects_in_view, merge_object_lists,
    _load_sam3, segment_objects_in_view,
    RENDER_W, RENDER_H, HFOV_DEG, NUM_VIEWS,
)

DEPTH_TOLERANCE = 0.15  # relative depth tolerance for occlusion test
MIN_SCORE = 0.35
DBSCAN_EPS = 0.3
DBSCAN_MIN_SAMPLES = 50
MIN_INSTANCE_GAUSSIANS = 100
NUM_SAM_GPUS = 4  # GPUs to use for parallel SAM3
FILTER_CACHE_VERSION = 3
AMBIGUOUS_OVERWRITE_LABELS = {
    "speaker",
    "phone",
    "remote",
    "calculator",
    "computer mouse",
    "mouse",
    "usb drive",
    "security camera",
    "power adapter",
    "adapter",
    "charger",
}
AMBIGUOUS_OVERWRITE_PENALTY = 0.15
WEAK_SUPPORT_DEVICE_LABELS = {
    "speaker",
    "remote",
    "usb drive",
    "security camera",
    "power adapter",
    "adapter",
    "charger",
}


# ── VLM object list filtering ───────────────────────────────────────────────

def vlm_filter_objects(raw_objects, caption_objects=None, view_objects_per_view=None):
    """Ask the configured VLM to filter ambiguous, non-meaningful, or duplicate objects.

    Returns a cleaned list of object names suitable for spatial QA.
    """
    caption_objects = [str(n).lower().strip() for n in (caption_objects or []) if str(n).strip()]
    view_counts = Counter()
    for view_objs in (view_objects_per_view or []):
        seen = {str(n).lower().strip() for n in view_objs if str(n).strip()}
        for name in seen:
            view_counts[name] += 1

    prompt = f"""I have detected these object categories in a 3D indoor scene:
merged_candidates = {json.dumps(raw_objects)}
caption_objects = {json.dumps(sorted(set(caption_objects)))}
per_view_support = {json.dumps(dict(sorted(view_counts.items())))}

For spatial reasoning QA (e.g. "How far is the sink from the towel rack?"),
filter this list by REMOVING:
1. Ambiguous or redundant categories that overlap with a better name
   (e.g. keep "door" but remove "door frame", "door handle" can stay if distinct)
2. Non-object surface elements (e.g. "tile", "linoleum", "frame")
3. Too-generic labels (e.g. "container", "fixture", "object")
4. Duplicates where one is a sub-part of another and not useful alone
   (e.g. "drain" is part of "sink", "spout" is part of "faucet" -- remove the sub-part)
5. Over-specific small-object or device labels that are weakly supported
   across views and not explicitly grounded by the caption

Keep objects that are:
- Clearly distinct physical items you could point at
- Useful for spatial reasoning (distance, position, relative placement)
- Meaningful in the context of the scene

For small electronics / gadgets:
- Keep a specific label such as "speaker", "phone", "calculator", "remote",
  "adapter", "router", "hub", "charger", or similar only when it is clearly
  identifiable and consistently supported by the evidence.
- If a fine-grained device label appears in only 1-2 views and is not mentioned
  in the caption, prefer dropping it rather than guessing.

Also MERGE near-synonyms into a single canonical name:
- e.g. "waste bin" and "trash can" -> keep only "trash can"
- e.g. "soap bar" and "soap" -> keep only "soap"

Return ONLY a JSON array of the filtered object names. No explanation."""

    result = vlm_chat([
        {"role": "system", "content": "You filter object lists for 3D scene understanding. Return valid JSON only."},
        {"role": "user", "content": prompt},
    ])
    try:
        start = result.index('[')
        end = result.rindex(']') + 1
        filtered = json.loads(result[start:end])
        filtered = [n.lower().strip() for n in filtered]
        caption_set = set(caption_objects)
        filtered = [
            name
            for name in filtered
            if not (
                name in WEAK_SUPPORT_DEVICE_LABELS
                and name not in caption_set
                and view_counts.get(name, 0) <= 2
            )
        ]
        return filtered
    except (ValueError, json.JSONDecodeError):
        print(f"  WARNING: Could not parse VLM filter response: {result[:200]}")
        caption_set = set(caption_objects)
        return [
            name
            for name in raw_objects
            if not (
                name in WEAK_SUPPORT_DEVICE_LABELS
                and name not in caption_set
                and view_counts.get(name, 0) <= 2
            )
        ]


# ── Step 1: SAM3 -> per-view indexed instance masks (multi-GPU) ──────────────

def run_sam3_multi_gpu(cameras, object_names_path, masks_dir, num_gpus=NUM_SAM_GPUS):
    """Launch SAM3 workers across multiple GPUs in parallel."""
    import subprocess, time

    n_views = len(cameras)
    num_gpus = min(num_gpus, n_views)
    views_per_gpu = [[] for _ in range(num_gpus)]
    for i in range(n_views):
        views_per_gpu[i % num_gpus].append(i)

    script = os.path.join(os.path.dirname(os.path.abspath(__file__)), "sam3_worker.py")
    python = sys.executable
    views_dir = os.path.dirname(cameras[0]["rgb_path"])

    procs = []
    for gpu_id, view_list in enumerate(views_per_gpu):
        if not view_list:
            continue
        cmd = [
            python, script,
            "--gpu", str(gpu_id),
            "--views", ",".join(str(v) for v in view_list),
            "--objects-json", str(object_names_path),
            "--views-dir", str(views_dir),
            "--out-dir", str(masks_dir),
            "--min-score", str(MIN_SCORE),
        ]
        print(f"  GPU {gpu_id}: views {view_list}")
        p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        procs.append((gpu_id, p))

    for gpu_id, p in procs:
        stdout, _ = p.communicate()
        for line in stdout.strip().split("\n"):
            if line.strip():
                print(f"    {line}")
        if p.returncode != 0:
            print(f"  WARNING: GPU {gpu_id} worker exited with rc={p.returncode}")


def build_indexed_masks_for_view(image_path, object_names):
    """Run SAM3 on one view (single-GPU fallback). Return (H,W) int mask + label map."""
    model, processor = _load_sam3()
    image = Image.open(image_path).convert("RGB")
    W, H = image.size
    state = processor.set_image(image)

    instance_mask = np.zeros((H, W), dtype=np.int32)
    score_mask = np.zeros((H, W), dtype=np.float32)
    label_map = {}
    next_id = 1

    for obj_name in object_names:
        processor.reset_all_prompts(state)
        try:
            output = processor.set_text_prompt(prompt=obj_name, state=state)
        except Exception:
            continue

        masks = output.get("masks")
        scores = output.get("scores")
        if masks is None or len(masks) == 0:
            continue

        for idx in range(len(masks)):
            score = float(scores[idx]) if scores is not None else 0.0
            if score < MIN_SCORE:
                continue
            m = masks[idx].cpu().numpy() if torch.is_tensor(masks[idx]) else np.array(masks[idx])
            if m.ndim == 3:
                m = m[0]
            m_bool = m > 0.5
            effective_score = score - (
                AMBIGUOUS_OVERWRITE_PENALTY
                if obj_name in AMBIGUOUS_OVERWRITE_LABELS
                else 0.0
            )
            overwrite = m_bool & (effective_score > score_mask)
            instance_mask[overwrite] = next_id
            score_mask[overwrite] = effective_score
            label_map[next_id] = obj_name
            next_id += 1

    return instance_mask, label_map


# ── Step 2: Project Gaussians into views ────────────────────────────────────

def project_gaussians_to_views(means_gpu, cameras, depth_maps):
    """Project all N Gaussians into each view. Return per-view arrays.

    Returns list of dicts per view:
      {valid_mask: [N] bool, pixel_u: [N] int, pixel_v: [N] int, cam_depth: [N] float}
    """
    N = means_gpu.shape[0]

    projections = []
    with torch.amp.autocast('cuda', enabled=False):
        pts_h = torch.cat([means_gpu.float(),
                           torch.ones(N, 1, device='cuda', dtype=torch.float32)], dim=1)

        for cam in cameras:
            c2w = torch.tensor(cam["c2w"], dtype=torch.float32, device='cuda')
            w2c = torch.linalg.inv(c2w)
            fx, fy, cx, cy = cam["fx"], cam["fy"], cam["cx"], cam["cy"]
            W, H = cam["width"], cam["height"]

            p_cam = (w2c @ pts_h.T).T  # [N, 4]
            pz = p_cam[:, 2]
            px = p_cam[:, 0]
            py = p_cam[:, 1]

            u = (fx * px / pz + cx).long()
            v = (fy * py / pz + cy).long()

            valid = (pz > 0.01) & (u >= 0) & (u < W) & (v >= 0) & (v < H)

            projections.append({
                "valid": valid.cpu().numpy(),
                "u": u.cpu().numpy(),
                "v": v.cpu().numpy(),
                "cam_depth": pz.float().cpu().numpy(),
            })

    return projections


# ── Step 3: Vote and assign ─────────────────────────────────────────────────

def assign_labels_by_voting(N, projections, instance_masks, label_maps, depth_maps):
    """For each Gaussian, collect votes from all views, majority-vote a label.

    Returns: labels [N] str array ('' = background)
    """
    vote_counts = [Counter() for _ in range(N)]

    for view_idx, (proj, inst_mask, lmap, depth) in enumerate(
            zip(projections, instance_masks, label_maps, depth_maps)):
        valid = proj["valid"]
        u = proj["u"]
        v = proj["v"]
        cam_d = proj["cam_depth"]

        valid_indices = np.where(valid)[0]
        if len(valid_indices) == 0:
            continue

        vu = v[valid_indices]
        uu = u[valid_indices]
        gd = cam_d[valid_indices]

        rendered_d = depth[vu, uu]

        # Depth test: only accept if Gaussian is near the rendered surface
        rel_diff = np.abs(gd - rendered_d) / (rendered_d + 1e-6)
        depth_ok = rel_diff < DEPTH_TOLERANCE

        passing = valid_indices[depth_ok]
        vu_pass = vu[depth_ok]
        uu_pass = uu[depth_ok]

        inst_ids = inst_mask[vu_pass, uu_pass]

        for gi, iid in zip(passing, inst_ids):
            if iid > 0 and iid in lmap:
                vote_counts[gi][lmap[iid]] += 1

    labels = np.empty(N, dtype=object)
    for i in range(N):
        if vote_counts[i]:
            labels[i] = vote_counts[i].most_common(1)[0][0]
        else:
            labels[i] = ""

    return labels


# ── Step 4: Instance clustering ─────────────────────────────────────────────

def _voxel_connected_components(pts: np.ndarray, eps: float) -> np.ndarray:
    """Find connected components via voxel hashing — O(N), GPU-free.

    Points within *eps* of each other (via shared/adjacent voxels) are
    in the same component.  Returns per-point component labels (int).
    """
    from scipy.sparse import lil_matrix
    from scipy.sparse.csgraph import connected_components

    # Quantize to voxel grid
    voxel_coords = np.floor(pts / eps).astype(np.int64)
    # Hash voxels to unique IDs
    offsets = voxel_coords - voxel_coords.min(axis=0)
    dims = offsets.max(axis=0) + 1
    voxel_ids = offsets[:, 0] * dims[1] * dims[2] + offsets[:, 1] * dims[2] + offsets[:, 2]

    # Map voxel_id → list of point indices
    from collections import defaultdict
    voxel_map: dict[int, list[int]] = defaultdict(list)
    for i, vid in enumerate(voxel_ids):
        voxel_map[vid].append(i)

    # Build voxel adjacency (26-connected neighborhood)
    unique_voxels = list(voxel_map.keys())
    voxel_set = set(unique_voxels)
    n_voxels = len(unique_voxels)
    voxel_to_idx = {v: i for i, v in enumerate(unique_voxels)}

    adj = lil_matrix((n_voxels, n_voxels), dtype=np.int8)
    for vid in unique_voxels:
        vi = voxel_to_idx[vid]
        oc = offsets[voxel_map[vid][0]]
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                for dz in (-1, 0, 1):
                    nc = (oc[0] + dx) * dims[1] * dims[2] + (oc[1] + dy) * dims[2] + (oc[2] + dz)
                    if nc in voxel_set:
                        adj[vi, voxel_to_idx[nc]] = 1

    n_comp, voxel_labels = connected_components(adj, directed=False)

    # Map back to points
    point_labels = np.full(len(pts), -1, dtype=np.int32)
    for vid, pidxs in voxel_map.items():
        label = voxel_labels[voxel_to_idx[vid]]
        for pi in pidxs:
            point_labels[pi] = label

    return point_labels


def cluster_instances(means_np, category_labels):
    """Aggregate all Gaussians with the same category label into one instance,
    then merge spatially overlapping instances (same object, inconsistent labels).

    Returns: instance_ids [N] int (-1 = background), instance_info dict
    """
    instance_ids = np.full(len(category_labels), -1, dtype=np.int32)
    instance_info = {}
    next_instance = 1  # 0 is reserved for background in seg masks

    categories = sorted(set(l for l in category_labels if l != ""))
    print(f"  Aggregating {len(categories)} categories (1 instance per label)...")

    for cat in categories:
        cat_mask = category_labels == cat
        cat_indices = np.where(cat_mask)[0]
        cat_pts = means_np[cat_indices]

        if len(cat_indices) < MIN_INSTANCE_GAUSSIANS:
            continue

        iid = next_instance
        instance_ids[cat_indices] = iid
        center = cat_pts.mean(axis=0)
        mins, maxs = cat_pts.min(0), cat_pts.max(0)
        instance_info[iid] = {
            "instance_id": iid,
            "label": f"{cat}_0",
            "category": cat,
            "n_gaussians": int(len(cat_indices)),
            "center": center.tolist(),
            "bbox_min": mins.tolist(),
            "bbox_max": maxs.tolist(),
        }
        next_instance += 1

    return instance_ids, instance_info


# ── Step 5: Render visualization ────────────────────────────────────────────

def get_distinct_colors(n):
    colors = []
    for i in range(n):
        hue = (i * 0.618033988749895) % 1.0
        sat, val = 0.85, 0.95
        h = hue * 6.0
        c = val * sat
        x = c * (1 - abs(h % 2 - 1))
        m = val - c
        if h < 1:   r, g, b = c, x, 0
        elif h < 2: r, g, b = x, c, 0
        elif h < 3: r, g, b = 0, c, x
        elif h < 4: r, g, b = 0, x, c
        elif h < 5: r, g, b = x, 0, c
        else:       r, g, b = c, 0, x
        colors.append(((r+m), (g+m), (b+m)))
    return colors


def render_labeled_view(gaussians, instance_ids, instance_colors, cam):
    """Render Gaussians colored by instance ID using gsplat."""
    from gsplat import rasterization

    N = gaussians['means'].shape[0]
    colors_rgb = torch.zeros(N, 3, device='cuda', dtype=torch.float32)
    bg_color = torch.tensor([0.85, 0.85, 0.85], device='cuda')

    for iid, color in instance_colors.items():
        mask = torch.tensor(instance_ids == iid, device='cuda')
        colors_rgb[mask] = torch.tensor(color, device='cuda', dtype=torch.float32)

    # Background Gaussians get original colors blended with gray
    bg_mask = torch.tensor(instance_ids == -1, device='cuda')
    orig = gaussians['colors']
    gray = orig.mean(dim=-1, keepdim=True).expand_as(orig) * 0.4
    colors_rgb[bg_mask] = gray[bg_mask]

    c2w = torch.tensor(cam["c2w"], dtype=torch.float32, device='cuda')
    viewmat = torch.linalg.inv(c2w).unsqueeze(0)
    fx, fy, cx, cy = cam["fx"], cam["fy"], cam["cx"], cam["cy"]
    W, H = cam["width"], cam["height"]
    K = torch.tensor([[fx, 0, cx], [0, fy, cy], [0, 0, 1]],
                     dtype=torch.float32, device='cuda').unsqueeze(0)

    renders, alphas, _ = rasterization(
        means=gaussians['means'], quats=gaussians['quats'],
        scales=gaussians['scales'], opacities=gaussians['opacities'],
        colors=colors_rgb, viewmats=viewmat, Ks=K,
        width=W, height=H, sh_degree=None,
    )
    img = renders[0].clamp(0, 1)
    alpha = alphas[0]
    img = img * alpha + bg_color.view(1, 1, 3) * (1 - alpha)
    return img.cpu().numpy()


def add_labels_to_image(img_np, instance_ids, instance_info, instance_colors, cam):
    """Project instance centroids into 2D and draw labels."""
    img = Image.fromarray((img_np * 255).astype(np.uint8))
    draw = ImageDraw.Draw(img)
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 15)
    except OSError:
        font = ImageFont.load_default()

    c2w = np.array(cam["c2w"], dtype=np.float32)
    w2c = np.linalg.inv(c2w)
    fx, fy, cx, cy = cam["fx"], cam["fy"], cam["cx"], cam["cy"]
    W, H = cam["width"], cam["height"]

    drawn = []
    for iid, info in sorted(instance_info.items(), key=lambda x: -x[1]["n_gaussians"]):
        center = np.array(info["center"] + [1.0], dtype=np.float32)
        p_cam = w2c @ center
        if p_cam[2] <= 0.1:
            continue
        u = int(fx * p_cam[0] / p_cam[2] + cx)
        v = int(fy * p_cam[1] / p_cam[2] + cy)
        if u < 10 or u > W - 10 or v < 10 or v > H - 10:
            continue

        too_close = any(abs(u - du) < 100 and abs(v - dv) < 22 for du, dv in drawn)
        if too_close:
            continue

        label_text = info["label"]
        color = instance_colors.get(iid, (1, 1, 1))
        rgb = tuple(int(c * 255) for c in color)

        bbox = draw.textbbox((u, v), label_text, font=font)
        tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
        tx = max(2, min(u - tw // 2, W - tw - 4))
        ty = max(2, min(v - th // 2, H - th - 4))
        draw.rectangle([tx - 2, ty - 2, tx + tw + 2, ty + th + 2], fill=(0, 0, 0, 200))
        draw.text((tx, ty), label_text, fill=rgb, font=font)
        drawn.append((u, v))

    return img


# ── Step 6: Build scene.json ────────────────────────────────────────────────

def build_scene_json_from_instances(instance_info, cameras, scene_id):
    objects = []
    for iid, info in sorted(instance_info.items()):
        mins = np.array(info["bbox_min"])
        maxs = np.array(info["bbox_max"])
        dims = maxs - mins
        objects.append({
            "id": info["label"],
            "label": info["category"],
            "type": info["category"],
            "x": info["center"][0],
            "y": info["center"][1],
            "z": float(mins[1]),
            "rotation_z": 0.0,
            "width": float(dims[0]),
            "length": float(dims[2]),
            "height": float(dims[1]),
            "n_gaussians": info["n_gaussians"],
        })

    cam_list = [{"name": c["name"], "position": c["position"],
                 "look_at": c["look_at"], "horizontal_fov_deg": c["horizontal_fov_deg"]}
                for c in cameras]

    return {"scene_id": scene_id, "source": "marble_3d_labeling",
            "objects": objects, "cameras": cam_list}


# ── Main ────────────────────────────────────────────────────────────────────

def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--marble-dir", default="output/marble_test")
    ap.add_argument("--seg-dir", default=None,
                    help="Segmentation dir (default: <marble-dir>/segmentation)")
    ap.add_argument("--skip-sam", action="store_true",
                    help="Reuse saved instance masks from previous run")
    ap.add_argument("--num-gpus", type=int, default=NUM_SAM_GPUS,
                    help=f"Number of GPUs for parallel SAM3 (default: {NUM_SAM_GPUS})")
    ap.add_argument("--single-gpu", action="store_true",
                    help="Run SAM3 sequentially on one GPU (no subprocess)")
    args = ap.parse_args()

    marble_dir = Path(args.marble_dir)
    seg_dir = Path(args.seg_dir) if args.seg_dir else marble_dir / "segmentation"
    views_dir = seg_dir / "views"
    viz_dir = seg_dir / "viz_3d"
    viz_dir.mkdir(parents=True, exist_ok=True)

    ply_path = marble_dir / "scene.ply"
    world_json_path = marble_dir / "world_full.json"

    world = json.load(open(world_json_path))
    scene_id = world.get("world_id", "unknown")[:12]
    cameras = json.load(open(views_dir / "cameras.json"))
    obj_lists = json.load(open(seg_dir / "object_lists.json"))
    raw_object_names = obj_lists["merged_objects"]
    print(f"Scene: {scene_id}")
    print(f"Raw object categories: {len(raw_object_names)}")
    print(f"Views: {len(cameras)}")

    # ── Step 0: VLM filter ambiguous objects ──────────────────────────────
    filtered_cache = seg_dir / f"filtered_objects_v{FILTER_CACHE_VERSION}.json"
    if filtered_cache.exists():
        object_names = json.load(open(filtered_cache))
        print(f"\n[0/6] Loaded filtered object list ({len(object_names)}): {object_names}")
    else:
        print(f"\n[0/6] Filtering {len(raw_object_names)} objects via configured VLM...")
        object_names = vlm_filter_objects(
            raw_object_names,
            caption_objects=obj_lists.get("caption_objects", []),
            view_objects_per_view=obj_lists.get("view_objects", []),
        )
        json.dump(object_names, open(filtered_cache, "w"), indent=2)
        print(f"  Filtered to {len(object_names)}: {object_names}")
        removed = set(raw_object_names) - set(object_names)
        if removed:
            print(f"  Removed: {sorted(removed)}")

    # ── Step 1: SAM3 -> indexed instance masks per view ──────────────────
    masks_dir = seg_dir / "instance_masks"
    masks_dir.mkdir(parents=True, exist_ok=True)
    masks_manifest = masks_dir / "manifest.json"

    instance_masks = []
    label_maps = []
    depth_maps = []

    current_manifest = {
        "filter_cache_version": FILTER_CACHE_VERSION,
        "min_score": MIN_SCORE,
        "object_names_sha1": hashlib.sha1(
            json.dumps(sorted(object_names), separators=(",", ":")).encode()
        ).hexdigest(),
    }
    cached_manifest = None
    if masks_manifest.exists():
        try:
            cached_manifest = json.load(open(masks_manifest))
        except Exception:
            cached_manifest = None
    all_masks_exist = (
        cached_manifest == current_manifest
        and all((masks_dir / f"{c['name']}_mask.npy").exists() for c in cameras)
    )

    if args.skip_sam and all_masks_exist:
        print("\n[1/6] Loading saved instance masks...")
    elif all_masks_exist:
        print("\n[1/6] Instance masks already exist, loading...")
    else:
        import time as _time
        t0 = _time.time()
        if args.single_gpu:
            print(f"\n[1/6] Running SAM3 sequentially (single GPU)...")
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
            torch.autocast("cuda", dtype=torch.bfloat16).__enter__()
            for cam in cameras:
                name = cam["name"]
                print(f"  {name}...", end=" ", flush=True)
                inst_mask, lmap = build_indexed_masks_for_view(cam["rgb_path"], object_names)
                np.save(masks_dir / f"{name}_mask.npy", inst_mask)
                json.dump({str(k): v for k, v in lmap.items()},
                          open(masks_dir / f"{name}_labels.json", "w"))
                n_labeled = (inst_mask > 0).sum()
                print(f"{len(lmap)} dets, {n_labeled} px ({n_labeled/inst_mask.size:.0%})")
        else:
            n_gpus = args.num_gpus
            print(f"\n[1/6] Running SAM3 on {len(cameras)} views across {n_gpus} GPUs...")
            run_sam3_multi_gpu(cameras, filtered_cache, masks_dir, num_gpus=n_gpus)
        print(f"  SAM3 total: {_time.time() - t0:.0f}s")
        json.dump(current_manifest, open(masks_manifest, "w"), indent=2)

    merged_dir = seg_dir / "instance_masks_merged"
    for cam in cameras:
        name = cam["name"]
        _mm = merged_dir / f"{name}_mask.npy"
        _ml = merged_dir / f"{name}_labels.json"
        mask_path = _mm if _mm.exists() else masks_dir / f"{name}_mask.npy"
        lbl_path = _ml if _ml.exists() else masks_dir / f"{name}_labels.json"
        m = np.load(mask_path)
        lm = json.load(open(lbl_path))
        lm = {int(k): v for k, v in lm.items()}
        d = np.load(cam["depth_path"])
        instance_masks.append(m)
        label_maps.append(lm)
        depth_maps.append(d)
    if merged_dir.exists():
        print(f"  Using merged masks from {merged_dir.name}/")

    # ── Step 2: Load Gaussians and project into views ────────────────────
    print("\n[2/6] Projecting 1.92M Gaussians into views...")
    gaussians = load_ply_gaussians(str(ply_path))
    means_gpu = gaussians['means']
    N = means_gpu.shape[0]

    projections = project_gaussians_to_views(means_gpu, cameras, depth_maps)
    for i, proj in enumerate(projections):
        n_vis = proj["valid"].sum()
        print(f"  {cameras[i]['name']}: {n_vis:,} visible ({n_vis/N:.0%})")

    # ── Step 3: Multi-view voting ────────────────────────────────────────
    print("\n[3/6] Voting across views to assign category labels...")
    category_labels = assign_labels_by_voting(
        N, projections, instance_masks, label_maps, depth_maps)

    labeled = np.array([l != "" for l in category_labels])
    label_counts = Counter(l for l in category_labels if l != "")
    print(f"  Labeled: {labeled.sum():,} / {N:,} ({labeled.mean():.1%})")
    print(f"  Categories: {len(label_counts)}")
    for cat, cnt in label_counts.most_common(15):
        print(f"    {cat}: {cnt:,}")
    if len(label_counts) > 15:
        print(f"    ... and {len(label_counts) - 15} more")

    # ── Step 4: Instance clustering ──────────────────────────────────────
    print("\n[4/6] Clustering into instances (DBSCAN)...")
    means_np = means_gpu.cpu().numpy()
    instance_ids, instance_info = cluster_instances(means_np, category_labels)

    assigned = (instance_ids >= 0).sum()
    print(f"  Instances: {len(instance_info)}")
    print(f"  Assigned Gaussians: {assigned:,} / {N:,} ({assigned/N:.1%})")
    for iid, info in sorted(instance_info.items()):
        print(f"    {info['label']}: {info['n_gaussians']:,} Gaussians, "
              f"center=({info['center'][0]:.2f},{info['center'][1]:.2f},{info['center'][2]:.2f})")

    # ── Step 5: Render visualization ─────────────────────────────────────
    print(f"\n[5/6] Rendering {len(cameras)} labeled views...")
    n_instances = len(instance_info)
    colors = get_distinct_colors(n_instances)
    instance_colors = {iid: colors[i] for i, iid in enumerate(sorted(instance_info.keys()))}

    for cam in cameras:
        name = cam["name"]
        print(f"  {name}...", end=" ", flush=True)
        img_np = render_labeled_view(gaussians, instance_ids, instance_colors, cam)
        img_pil = add_labels_to_image(img_np, instance_ids, instance_info, instance_colors, cam)
        img_pil.save(str(viz_dir / f"{name}_seg3d.png"))
        print("done")

    # Grid
    print("  Creating grid...")
    view_imgs = [Image.open(viz_dir / f"{c['name']}_seg3d.png") for c in cameras]
    cols = 4
    rows = math.ceil(len(view_imgs) / cols)
    tw, th = 648, 484
    grid = Image.new("RGB", (cols * tw, rows * th), (30, 30, 30))
    draw_g = ImageDraw.Draw(grid)
    try:
        font_g = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 18)
    except OSError:
        font_g = ImageFont.load_default()
    for i, img in enumerate(view_imgs):
        r, c = divmod(i, cols)
        grid.paste(img.resize((tw, th), Image.LANCZOS), (c * tw, r * th))
        draw_g.text((c * tw + 8, r * th + 4), cameras[i]["name"], fill=(255, 255, 0), font=font_g)
    grid.save(str(viz_dir / "grid_3d.png"))

    # Legend
    row_h = 24
    legend = Image.new("RGB", (300, 40 + n_instances * row_h), (30, 30, 30))
    draw_l = ImageDraw.Draw(legend)
    try:
        font_l = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 14)
    except OSError:
        font_l = ImageFont.load_default()
    draw_l.text((10, 5), "Instance Legend", fill=(255, 255, 255), font=font_l)
    for idx, (iid, info) in enumerate(sorted(instance_info.items())):
        y = 30 + idx * row_h
        color = tuple(int(c * 255) for c in instance_colors[iid])
        draw_l.rectangle([10, y, 30, y + row_h - 4], fill=color)
        draw_l.text((36, y + 2), f"{info['label']} ({info['n_gaussians']:,})",
                    fill=(255, 255, 255), font=font_l)
    legend.save(str(viz_dir / "legend_3d.png"))

    # ── Step 6: Save outputs ─────────────────────────────────────────────
    print("\n[6/6] Saving outputs...")
    np.save(seg_dir / "gaussian_labels.npy", instance_ids)
    print(f"  gaussian_labels.npy: {instance_ids.shape}")

    json.dump({str(k): v for k, v in instance_info.items()},
              open(seg_dir / "instance_info.json", "w"), indent=2)
    print(f"  instance_info.json: {len(instance_info)} instances")

    scene = build_scene_json_from_instances(instance_info, cameras, scene_id)
    json.dump(scene, open(seg_dir / "scene.json", "w"), indent=2)
    print(f"  scene.json: {len(scene['objects'])} objects")

    for obj in scene["objects"]:
        print(f"    {obj['id']}: pos=({obj['x']:.2f},{obj['y']:.2f},{obj['z']:.2f}), "
              f"size=({obj['width']:.2f}x{obj['length']:.2f}x{obj['height']:.2f}), "
              f"gaussians={obj['n_gaussians']}")

    print(f"\n  Visualizations: {viz_dir}/")
    print("Done!")


if __name__ == "__main__":
    main()
