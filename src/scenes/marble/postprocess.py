#!/usr/bin/env python3
"""
Post-process 3D instance labels: split disconnected components, then
VLM-merge components that belong to the same physical object.

Parallelizes gsplat rendering across multiple GPUs.

Usage:
  python postprocess_instances.py --marble-dir output/spar_kitchen [--eps 0.1] [--num-gpus 8]
"""

import sys, os, json, math, time
import numpy as np
from pathlib import Path
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
sys.path.insert(0, os.path.dirname(__file__))
from segment_scene import load_ply_gaussians
from label_gaussians import (
    render_labeled_view, add_labels_to_image,
    get_distinct_colors, build_scene_json_from_instances,
    project_gaussians_to_views, assign_labels_by_voting, cluster_instances,
    _voxel_connected_components,
)
from PIL import Image, ImageDraw, ImageFont


def restore_labels(ply_path, seg_dir, cameras):
    """Re-run voting + clustering from saved SAM masks (steps 2-4)."""
    t = time.time()
    gaussians = load_ply_gaussians(str(ply_path))
    means_gpu = gaussians['means']
    N = means_gpu.shape[0]

    instance_masks, label_maps, depth_maps = [], [], []
    masks_dir = seg_dir / "instance_masks"
    merged_dir = seg_dir / "instance_masks_merged"
    for cam in cameras:
        name = cam["name"]
        _m = merged_dir / f"{name}_mask.npy"
        _l = merged_dir / f"{name}_labels.json"
        mask_path = _m if _m.exists() else masks_dir / f"{name}_mask.npy"
        lbl_path = _l if _l.exists() else masks_dir / f"{name}_labels.json"
        instance_masks.append(np.load(mask_path))
        lm = json.load(open(lbl_path))
        label_maps.append({int(k): v for k, v in lm.items()})
        depth_maps.append(np.load(cam["depth_path"]))
    print(f"  load: {time.time()-t:.1f}s")

    t = time.time()
    print("[1/5] Projecting Gaussians...")
    projections = project_gaussians_to_views(means_gpu, cameras, depth_maps)
    print(f"  project: {time.time()-t:.1f}s")

    t = time.time()
    print("[2/5] Voting...")
    category_labels = assign_labels_by_voting(
        N, projections, instance_masks, label_maps, depth_maps)
    labeled = sum(1 for l in category_labels if l != "")
    print(f"  {labeled:,}/{N:,} labeled ({time.time()-t:.1f}s)")

    t = time.time()
    print("[3/5] Clustering...")
    means_np = means_gpu.cpu().numpy()
    instance_ids, instance_info = cluster_instances(means_np, category_labels)
    print(f"  {len(instance_info)} instances ({time.time()-t:.1f}s)")

    return gaussians, means_np, instance_ids, instance_info


def split_components(means, labels, instance_info, eps=0.10):
    """Split disconnected components (>eps apart) into separate instances."""
    next_id = max(instance_info.keys()) + 1
    splits = 0

    for iid in list(instance_info.keys()):
        info = instance_info[iid]
        mask = labels == iid
        n = mask.sum()
        if n < 30:
            continue
        pts = means[mask]
        idx = np.where(mask)[0]
        full_labels = _voxel_connected_components(pts, eps)

        cluster_ids = sorted(set(full_labels) - {-1})
        if len(cluster_ids) <= 1:
            labels[idx[full_labels == -1]] = -1
            continue

        cluster_sizes = [(c, (full_labels == c).sum()) for c in cluster_ids]
        cluster_sizes.sort(key=lambda x: -x[1])
        cat = info['category']

        for rank, (cid, csz) in enumerate(cluster_sizes):
            cmask = full_labels == cid
            cpts = pts[cmask]
            cidx = idx[cmask]
            if rank == 0:
                labels[idx[~cmask & (full_labels >= 0)]] = -999
                labels[idx[full_labels == -1]] = -1
                info['center'] = cpts.mean(axis=0).tolist()
                info['bbox_min'] = cpts.min(axis=0).tolist()
                info['bbox_max'] = cpts.max(axis=0).tolist()
                info['n_gaussians'] = int(csz)
            else:
                new_id = next_id; next_id += 1
                labels[cidx] = new_id
                existing = [v['label'] for v in instance_info.values() if v['category'] == cat]
                new_idx = max(int(l.split('_')[-1]) for l in existing) + 1
                instance_info[new_id] = {
                    'instance_id': new_id, 'label': f'{cat}_{new_idx}',
                    'category': cat, 'n_gaussians': int(csz),
                    'center': cpts.mean(axis=0).tolist(),
                    'bbox_min': cpts.min(axis=0).tolist(),
                    'bbox_max': cpts.max(axis=0).tolist(),
                }
        labels[labels == -999] = -1
        splits += len(cluster_sizes) - 1

    # Remove tiny
    tiny = [iid for iid, info in instance_info.items() if info['n_gaussians'] < 10]
    for iid in tiny:
        labels[labels == iid] = -1
        del instance_info[iid]

    print(f"  {splits} splits, {len(tiny)} tiny removed, {len(instance_info)} instances")
    return labels, instance_info


def vlm_merge_components(means, labels, instance_info, seg_dir, num_views=12):
    """Ask VLM across 3 views to identify cross-category instances that
    are actually the same physical object (e.g. blind+window, door+cabinet).

    Queries 3 views spread ~120° apart so every angle is covered.
    A merge is applied if ANY view confirms it.
    """
    import base64, requests
    from concurrent.futures import ThreadPoolExecutor

    api_key = os.environ.get("VLM_API_KEY", "")

    # Find nearby pairs of DIFFERENT categories
    items = [(iid, info) for iid, info in instance_info.items() if info['n_gaussians'] >= 50]
    pairs = []
    for i, (id_a, a) in enumerate(items):
        for id_b, b in items[i+1:]:
            if a['category'] == b['category']:
                continue
            ca, cb = np.array(a['center']), np.array(b['center'])
            dist = np.linalg.norm(ca - cb)
            if dist < 1.0:
                pairs.append((id_a, a, id_b, b, round(dist, 2)))

    if not pairs:
        print("  No nearby cross-category pairs found")
        return labels, instance_info

    n_rounds = 5
    print(f"  {len(pairs)} candidate pairs, checking {num_views} views × {n_rounds} rounds in parallel...")

    pair_desc = []
    for id_a, a, id_b, b, dist in pairs:
        pair_desc.append({
            "a": a['label'], "b": b['label'], "dist_m": dist,
            "a_size": [round(a['bbox_max'][i] - a['bbox_min'][i], 2) for i in range(3)],
            "b_size": [round(b['bbox_max'][i] - b['bbox_min'][i], 2) for i in range(3)],
        })

    prompt_template = """Image 1: RGB rendering of a room from one viewpoint.
Image 2: 3D segmentation of the same view — each instance colored and labeled.

These object pairs are spatially very close (<1m apart):
{pairs}

For EACH pair, answer: are they physically COMPONENTS of the same object?

YES (merge) — they are physically attached/integrated parts:
- blind + window → same wall opening
- door + cabinet → door is the cabinet's door panel
- chair back + chair seat → parts of same chair
- drawer + cabinet → drawer is built into the cabinet

NO (keep separate) — merely near each other:
- laptop ON desk, picture ON wall, plant IN flowerpot
- keyboard ON desk, phone ON desk, book ON shelf

Be STRICT: only merge if physically attached. Objects sitting on, hanging from,
or placed next to each other are SEPARATE.

Return ONLY a JSON array: [{{"keep": "window_0", "remove": "blind_0"}}]
Return [] if no merges."""

    def _query_view(args):
        vi, round_idx = args
        rgb_path = seg_dir / "views" / f"view_{vi:03d}.png"
        seg_path = seg_dir / "viz_3d" / f"view_{vi:03d}_seg3d.png"
        content = []
        for p in [rgb_path, seg_path]:
            if not p.exists():
                continue
            with open(p, "rb") as f:
                b64 = base64.b64encode(f.read()).decode()
            content.append({"type": "image_url",
                            "image_url": {"url": f"data:image/png;base64,{b64}", "detail": "high"}})
        if len(content) < 2:
            return vi, round_idx, []
        content.append({"type": "text",
                        "text": prompt_template.format(pairs=json.dumps(pair_desc, indent=1))})
        try:
            resp = requests.post(
                os.environ.get("VLM_API_BASE", "https://example.com/v1/chat/completions"),
                headers={"Authorization": f"Bearer {api_key}",
                         "Content-Type": "application/json"},
                json={"model": os.environ.get("VLM_MODEL", "Qwen3-VL-235B-A22B-Thinking"),
                      "messages": [{"role": "user", "content": content}],
                      "temperature": 0.2 * round_idx, "max_completion_tokens": 512},
                timeout=90,
            )
            resp.raise_for_status()
            text = resp.json()["choices"][0]["message"]["content"].strip()
            start, end = text.find("["), text.rfind("]") + 1
            return vi, round_idx, json.loads(text[start:end]) if start >= 0 and end > start else []
        except Exception as e:
            return vi, round_idx, []

    # All views × n_rounds, fully parallel
    tasks = [(vi, r) for vi in range(num_views) for r in range(n_rounds)]
    all_merges: list[dict] = []
    with ThreadPoolExecutor(max_workers=100) as pool:
        results = list(pool.map(_query_view, tasks))
    for vi, r, merges in results:
        if merges:
            print(f"    view_{vi:03d} r{r}: {len(merges)} merge(s)")
            all_merges.extend(merges)

    # Deduplicate: keep each (keep, remove) pair once
    seen = set()
    unique_merges = []
    for m in all_merges:
        key = (m.get("keep", ""), m.get("remove", ""))
        if key not in seen:
            seen.add(key)
            unique_merges.append(m)

    # Apply merges — chase chains if a target was already merged
    label_to_id = {info['label']: iid for iid, info in instance_info.items()}
    # Track where each id ended up after chained merges
    merged_into: dict[int, int] = {}

    def _resolve(iid: int) -> int:
        while iid in merged_into:
            iid = merged_into[iid]
        return iid

    for m in unique_merges:
        keep_label, remove_label = m.get("keep"), m.get("remove")
        keep_id = label_to_id.get(keep_label)
        remove_id = label_to_id.get(remove_label)
        if keep_id is None or remove_id is None:
            continue
        keep_id = _resolve(keep_id)
        remove_id = _resolve(remove_id)
        if keep_id == remove_id or remove_id not in instance_info:
            continue
        if keep_id not in instance_info:
            keep_id, remove_id = remove_id, keep_id  # swap if keep was consumed
        labels[labels == remove_id] = keep_id
        merged_into[remove_id] = keep_id
        instance_info.pop(remove_id, None)
        ki = instance_info[keep_id]
        mask = labels == keep_id
        pts = means[mask]
        ki['center'] = pts.mean(axis=0).tolist()
        ki['bbox_min'] = pts.min(axis=0).tolist()
        ki['bbox_max'] = pts.max(axis=0).tolist()
        ki['n_gaussians'] = int(mask.sum())
        print(f"  Merged {remove_label} → {ki['label']}")

    if not unique_merges:
        print("  VLM: no merges needed")
    return labels, instance_info


def vlm_merge_3d(means_np, instance_ids, instance_info, seg_dir,
                 ply_path, cameras, *, n_iters=3, max_dist=1.5):
    """Ask VLM whether nearby 3D instance pairs should be merged.

    For each pair within max_dist, find the best camera view where both
    are visible, send RGB + 3D seg overlay + panorama to VLM, and merge
    if confirmed. Runs n_iters iterations to allow cascading merges.
    """
    import base64, requests
    from concurrent.futures import ThreadPoolExecutor

    api_key = os.environ.get("VLM_API_KEY", "")
    if not api_key:
        print("  [vlm-3d] No VLM_API_KEY, skipping")
        return instance_ids, instance_info

    VLM_URL = os.environ.get("VLM_API_BASE", "https://example.com/v1/chat/completions")
    VLM_MODEL = os.environ.get("VLM_MODEL", "Qwen3-VL-235B-A22B-Thinking")

    views_dir = seg_dir / "views"
    viz_dir = seg_dir / "viz_3d"
    panorama_path = seg_dir.parent / "panorama.jpg"

    def _encode(path, detail="low"):
        if not os.path.isfile(str(path)):
            return None
        with open(str(path), "rb") as f:
            b64 = base64.b64encode(f.read()).decode()
        ext = str(path).rsplit(".", 1)[-1].lower()
        mime = {"jpg": "jpeg", "jpeg": "jpeg"}.get(ext, "png")
        return {"type": "image_url",
                "image_url": {"url": f"data:image/{mime};base64,{b64}",
                              "detail": detail}}

    def _find_best_view(id_a, id_b):
        """Find the view whose look-at direction best faces both instances."""
        ca = np.array(instance_info[id_a]["center"])
        cb = np.array(instance_info[id_b]["center"])
        mid = (ca + cb) / 2
        best_vi, best_score = 0, -999.0
        for vi, cam in enumerate(cameras):
            cam_pos = np.array(cam.get("position", [0, 0, 0]))
            look_at = np.array(cam.get("look_at", [0, 0, -1]))
            fwd = look_at - cam_pos
            fwd_norm = np.linalg.norm(fwd)
            if fwd_norm < 1e-6:
                continue
            fwd = fwd / fwd_norm
            to_mid = mid - cam_pos
            to_mid_norm = np.linalg.norm(to_mid)
            if to_mid_norm < 1e-6:
                continue
            cos_angle = float(np.dot(fwd, to_mid / to_mid_norm))
            if cos_angle > best_score:
                best_score = cos_angle
                best_vi = vi
        return best_vi

    def _vlm_check_pair(task):
        id_a, id_b, vi = task
        a_info = instance_info[id_a]
        b_info = instance_info[id_b]
        cam = cameras[vi]
        cam_name = cam["name"]

        content = []
        seg_img = _encode(viz_dir / f"{cam_name}_seg3d.png", "high")
        if seg_img:
            content.append(seg_img)
        rgb_img = _encode(views_dir / f"{cam_name}.png", "high")
        if rgb_img:
            content.append(rgb_img)
        pano = _encode(panorama_path, "low")
        if pano:
            content.append(pano)

        if len(content) < 2:
            return id_a, id_b, False

        content.append({"type": "text", "text": (
            "Image 1: 3D instance segmentation with labeled colors.\n"
            "Image 2: RGB photo of the same view.\n"
            "Image 3: panorama of the full scene.\n\n"
            f"Should \"{a_info['label']}\" and \"{b_info['label']}\" "
            f"be MERGED into one instance?\n\n"
            "MERGE if:\n"
            "- They are adjacent parts of the SAME physical object "
            "(e.g. two sections of a long cabinet, parts of a kitchen "
            "island, connected countertop sections)\n"
            "- One is a sub-component of the other "
            "(e.g. handle on a door, knob on an appliance, "
            "drawer built into a cabinet)\n"
            "- They are the same object labeled with different names "
            "across views (e.g. 'tongs' and 'scissors' for one tool)\n\n"
            "DO NOT MERGE if:\n"
            "- They are separate objects that happen to be nearby "
            "(e.g. plate on counter, bottle next to sink)\n"
            "- They are the same category but physically disconnected "
            "(e.g. two separate chairs, two separate windows)\n"
            "- One is sitting on, leaning against, or placed next to "
            "the other without being attached\n\n"
            "Answer ONLY: {\"merge\": true} or {\"merge\": false}"
        )})

        try:
            resp = requests.post(VLM_URL, headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            }, json={
                "model": VLM_MODEL,
                "messages": [{"role": "user", "content": content}],
                "temperature": 0.1,
                "max_completion_tokens": 64,
            }, timeout=30)
            resp.raise_for_status()
            text = resp.json()["choices"][0]["message"]["content"].strip()
            return id_a, id_b, '"merge": true' in text.lower() or '"merge":true' in text.lower()
        except Exception as e:
            print(f"    [vlm-3d] error for {a_info['label']}+{b_info['label']}: {e}")
            return id_a, id_b, False

    total_merged = 0
    for iteration in range(n_iters):
        iids = sorted(instance_info.keys())
        pairs = []
        for i, id_a in enumerate(iids):
            for id_b in iids[i + 1:]:
                ca = np.array(instance_info[id_a]["center"])
                cb = np.array(instance_info[id_b]["center"])
                dist = float(np.linalg.norm(ca - cb))
                if dist < max_dist:
                    vi = _find_best_view(id_a, id_b)
                    if vi >= 0:
                        pairs.append((id_a, id_b, vi))

        if not pairs:
            print(f"  iter {iteration+1}: no nearby pairs, stopping")
            break

        print(f"  iter {iteration+1}: {len(pairs)} nearby pairs, querying VLM...")
        with ThreadPoolExecutor(max_workers=50) as pool:
            results = list(pool.map(_vlm_check_pair, pairs))

        merge_map: dict[int, int] = {}
        for id_a, id_b, should_merge in results:
            if not should_merge:
                continue
            if id_a in merge_map or id_b in merge_map:
                continue
            smaller, larger = (id_a, id_b) if instance_info[id_a]["n_gaussians"] <= instance_info[id_b]["n_gaussians"] else (id_b, id_a)
            merge_map[smaller] = larger

        if not merge_map:
            print(f"  iter {iteration+1}: no merges confirmed, stopping")
            break

        for old in list(merge_map.keys()):
            new = merge_map[old]
            while new in merge_map:
                new = merge_map[new]
            merge_map[old] = new

        for old, new in merge_map.items():
            instance_ids[instance_ids == old] = new
            old_info = instance_info.pop(old)
            new_info = instance_info[new]
            all_pts = means_np[instance_ids == new]
            new_info["n_gaussians"] = int(len(all_pts))
            new_info["center"] = all_pts.mean(axis=0).tolist()
            new_info["bbox_min"] = all_pts.min(axis=0).tolist()
            new_info["bbox_max"] = all_pts.max(axis=0).tolist()
            print(f"    {old_info['label']} -> {new_info['label']}")

        total_merged += len(merge_map)
        print(f"  iter {iteration+1}: {len(merge_map)} merges applied, "
              f"{len(instance_info)} instances remaining")

        # Re-save for next iteration's viz render
        np.save(seg_dir / "gaussian_labels.npy", instance_ids)
        json.dump({str(k): v for k, v in instance_info.items()},
                  open(seg_dir / "instance_info.json", "w"), indent=2)

    if total_merged:
        print(f"  VLM 3D merge: {total_merged} total merges across {n_iters} iters")
    else:
        print(f"  VLM 3D merge: no merges needed")

    return instance_ids, instance_info


def vlm_filter_3d(means_np, instance_ids, instance_info, seg_dir,
                  cameras):
    """Ask VLM to verify each 3D instance: keep valid objects, remove
    structural elements, badly segmented blobs, or mislabeled items.

    For each instance, find the view where it's most prominent, send
    RGB + seg overlay + panorama to VLM. If rejected, remove all
    Gaussians of that instance.
    """
    import base64, requests
    from concurrent.futures import ThreadPoolExecutor

    api_key = os.environ.get("VLM_API_KEY", "")
    if not api_key:
        print("  [vlm-filter] No VLM_API_KEY, skipping")
        return instance_ids, instance_info

    VLM_URL = os.environ.get("VLM_API_BASE", "https://example.com/v1/chat/completions")
    VLM_MODEL = os.environ.get("VLM_MODEL", "Qwen3-VL-235B-A22B-Thinking")

    views_dir = seg_dir / "views"
    viz_dir = seg_dir / "viz_3d"
    panorama_path = seg_dir.parent / "panorama.jpg"

    def _encode(path, detail="low"):
        if not os.path.isfile(str(path)):
            return None
        with open(str(path), "rb") as f:
            b64 = base64.b64encode(f.read()).decode()
        ext = str(path).rsplit(".", 1)[-1].lower()
        mime = {"jpg": "jpeg", "jpeg": "jpeg"}.get(ext, "png")
        return {"type": "image_url",
                "image_url": {"url": f"data:image/{mime};base64,{b64}",
                              "detail": detail}}

    def _best_view_for_instance(iid):
        """Find the camera whose look-at direction best faces this instance."""
        center = np.array(instance_info[iid]["center"])
        best_vi, best_score = 0, -999.0
        for vi, cam in enumerate(cameras):
            cam_pos = np.array(cam.get("position", [0, 0, 0]))
            look_at = np.array(cam.get("look_at", [0, 0, -1]))
            fwd = look_at - cam_pos
            fn = np.linalg.norm(fwd)
            if fn < 1e-6:
                continue
            fwd = fwd / fn
            to_c = center - cam_pos
            tn = np.linalg.norm(to_c)
            if tn < 1e-6:
                continue
            score = float(np.dot(fwd, to_c / tn))
            if score > best_score:
                best_score = score
                best_vi = vi
        return best_vi

    tasks = []
    for iid, info in instance_info.items():
        vi = _best_view_for_instance(iid)
        tasks.append((iid, info, vi))

    def _vlm_verify(task):
        iid, info, vi = task
        cam = cameras[vi]
        cam_name = cam["name"]

        content = []
        seg_img = _encode(viz_dir / f"{cam_name}_seg3d.png", "high")
        if seg_img:
            content.append(seg_img)
        rgb_img = _encode(views_dir / f"{cam_name}.png", "high")
        if rgb_img:
            content.append(rgb_img)
        pano = _encode(panorama_path, "low")
        if pano:
            content.append(pano)

        if len(content) < 2:
            return iid, True

        content.append({"type": "text", "text": (
            "Image 1: 3D instance segmentation with labeled colors.\n"
            "Image 2: RGB photo of the same view.\n"
            "Image 3: panorama of the full scene.\n\n"
            f"Evaluate the instance labeled \"{info['label']}\" "
            f"({info['n_gaussians']:,} points).\n\n"
            "KEEP this instance UNLESS it is clearly noise.\n"
            "Keep all real objects: furniture, appliances, fixtures, "
            "doors, windows, cabinets, chairs, decorations, etc.\n"
            "Keep even if the segmentation is imperfect.\n\n"
            "REMOVE ONLY if:\n"
            "- The colored region is clearly noise, random scattered "
            "points, or an unidentifiable blob\n"
            "- The label is completely wrong and the colored region "
            "does not correspond to ANY real object\n"
            "- It is a tiny fragment with no recognizable shape\n\n"
            "- It is a clearly wrong label that does not correspond to the colored region\n\n"
            "When in doubt, KEEP it.\n\n"
            "Answer ONLY: {\"keep\": true} or {\"keep\": false}"
        )})

        try:
            resp = requests.post(VLM_URL, headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            }, json={
                "model": VLM_MODEL,
                "messages": [{"role": "user", "content": content}],
                "temperature": 0.1,
                "max_completion_tokens": 64,
            }, timeout=30)
            resp.raise_for_status()
            text = resp.json()["choices"][0]["message"]["content"].strip()
            keep = '"keep": true' in text.lower() or '"keep":true' in text.lower()
            return iid, keep
        except Exception as e:
            print(f"    [vlm-filter] error for {info['label']}: {e}")
            return iid, True

    print(f"  {len(tasks)} instances to verify...")
    with ThreadPoolExecutor(max_workers=50) as pool:
        results = list(pool.map(_vlm_verify, tasks))

    removed = []
    for iid, keep in results:
        if not keep:
            info = instance_info.pop(iid)
            instance_ids[instance_ids == iid] = -1
            removed.append(info["label"])

    if removed:
        print(f"  Removed {len(removed)}: {removed}")
    else:
        print(f"  All instances verified, none removed")

    return instance_ids, instance_info


def _render_one_view(args):
    """Render a single view on a specific GPU (for parallel execution)."""
    gpu_id, cam, ply_path, labels_path, inst_info_path, out_path = args
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    import torch
    sys.path.insert(0, os.path.dirname(__file__))
    from segment_scene import load_ply_gaussians
    from label_gaussians import render_labeled_view, add_labels_to_image, get_distinct_colors

    gaussians = load_ply_gaussians(ply_path)
    instance_ids = np.load(labels_path)
    with open(inst_info_path) as f:
        instance_info = {int(k): v for k, v in json.load(f).items()}
    n_inst = len(instance_info)
    colors = get_distinct_colors(n_inst)
    instance_colors = {iid: colors[i] for i, iid in enumerate(sorted(instance_info.keys()))}

    img_np = render_labeled_view(gaussians, instance_ids, instance_colors, cam)
    img_pil = add_labels_to_image(img_np, instance_ids, instance_info, instance_colors, cam)
    img_pil.save(out_path)
    return cam['name']


def render_parallel(cameras, ply_path, labels_path, inst_info_path, viz_dir, num_gpus=8):
    """Render all views in parallel across GPUs using subprocess."""
    import subprocess as _sp
    procs = []
    for i, cam in enumerate(cameras):
        gpu_id = i % num_gpus
        out_path = str(viz_dir / f"{cam['name']}_seg3d.png")
        script = f"""\
import os, sys, json, numpy as np
os.environ["CUDA_VISIBLE_DEVICES"] = "{gpu_id}"
sys.path.insert(0, {os.path.dirname(__file__)!r})
from segment_scene import load_ply_gaussians
from label_gaussians import render_labeled_view, add_labels_to_image, get_distinct_colors

gaussians = load_ply_gaussians({str(ply_path)!r})
labels = np.load({str(labels_path)!r})
with open({str(inst_info_path)!r}) as f:
    info = {{int(k): v for k, v in json.load(f).items()}}
colors = get_distinct_colors(len(info))
ic = {{iid: colors[i] for i, iid in enumerate(sorted(info.keys()))}}
cam = {json.dumps(cam)}
img = render_labeled_view(gaussians, labels, ic, cam)
pil = add_labels_to_image(img, labels, info, ic, cam)
pil.save({out_path!r})
"""
        p = _sp.Popen([sys.executable, "-c", script],
                      stdout=_sp.DEVNULL, stderr=_sp.DEVNULL)
        procs.append((cam['name'], p))
        # Batch: wait when all GPUs busy
        if len(procs) >= num_gpus:
            name, p = procs.pop(0)
            p.wait()
            print(f"  {name}")
    for name, p in procs:
        p.wait()
        print(f"  {name}")


def make_grid_legend(cameras, instance_info, instance_colors, viz_dir):
    """Generate grid + legend images."""
    view_imgs = [Image.open(viz_dir / f"{c['name']}_seg3d.png") for c in cameras]
    cols, tw, th = 4, 648, 484
    rows = math.ceil(len(view_imgs) / cols)
    grid = Image.new("RGB", (cols * tw, rows * th), (30, 30, 30))
    draw_g = ImageDraw.Draw(grid)
    font_g = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 18)
    for i, img in enumerate(view_imgs):
        r, c = divmod(i, cols)
        grid.paste(img.resize((tw, th), Image.LANCZOS), (c * tw, r * th))
        draw_g.text((c * tw + 8, r * th + 4), cameras[i]["name"], fill=(255, 255, 0), font=font_g)
    grid.save(str(viz_dir / "grid_3d.png"))

    n_inst = len(instance_info)
    row_h = 24
    legend = Image.new("RGB", (300, 40 + n_inst * row_h), (30, 30, 30))
    draw_l = ImageDraw.Draw(legend)
    font_l = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 14)
    draw_l.text((10, 5), "Instance Legend", fill=(255, 255, 255), font=font_l)
    for idx, (iid, info) in enumerate(sorted(instance_info.items())):
        y = 30 + idx * row_h
        color = tuple(int(c * 255) for c in instance_colors[iid])
        draw_l.rectangle([10, y, 30, y + row_h - 4], fill=color)
        draw_l.text((36, y + 2), f"{info['label']} ({info['n_gaussians']:,})",
                    fill=(255, 255, 255), font=font_l)
    legend.save(str(viz_dir / "legend_3d.png"))


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--marble-dir", default="output/spar_kitchen")
    ap.add_argument("--eps", type=float, default=0.10, help="Connected component threshold (m)")
    ap.add_argument("--num-gpus", type=int, default=8)
    ap.add_argument("--skip-restore", action="store_true",
                    help="Skip re-running voting+clustering, use existing labels")
    ap.add_argument("--skip-viz-render", action="store_true",
                    help="Skip final visualization render (saves ~14s)")
    ap.add_argument("--skip-vlm-merge", action="store_true",
                    help="Skip VLM merge (handled by step 4c instead)")
    args = ap.parse_args()

    marble_dir = Path(args.marble_dir)
    seg_dir = marble_dir / "segmentation"
    viz_dir = seg_dir / "viz_3d"
    viz_dir.mkdir(parents=True, exist_ok=True)
    ply_path = marble_dir / "scene.ply"
    cameras = json.load(open(seg_dir / "views" / "cameras.json"))

    t0 = time.time()

    if args.skip_restore:
        print("[1-3/5] Loading existing labels...")
        from segment_scene import load_ply_gaussians as _load
        gaussians = _load(str(ply_path))
        means_np = gaussians['means'].cpu().numpy()
        instance_ids = np.load(seg_dir / "gaussian_labels.npy")
        with open(seg_dir / "instance_info.json") as f:
            instance_info = {int(k): v for k, v in json.load(f).items()}
    else:
        gaussians, means_np, instance_ids, instance_info = restore_labels(
            ply_path, seg_dir, cameras)
    print(f"  {time.time()-t0:.0f}s")

    # Split
    t1 = time.time()
    print(f"\n[4/5] Splitting disconnected components (eps={args.eps}m)...")
    instance_ids, instance_info = split_components(
        means_np, instance_ids, instance_info, eps=args.eps)
    print(f"  {time.time()-t1:.0f}s")

    # Spatial merge: merge instances with different labels but co-located
    # point clouds (same physical object labeled differently across views).
    # Runs AFTER split so each instance is a coherent spatial cluster.
    t_sm = time.time()
    from scipy.spatial import cKDTree
    iids = sorted(instance_info.keys())
    merge_map: dict[int, int] = {}
    trees: dict[int, cKDTree] = {}
    for iid in iids:
        pts = means_np[instance_ids == iid]
        if len(pts) > 0:
            trees[iid] = cKDTree(pts)

    for i, a in enumerate(iids):
        if a in merge_map or a not in trees:
            continue
        for b in iids[i + 1:]:
            if b in merge_map or b not in trees:
                continue
            if instance_info[a]["category"] == instance_info[b]["category"]:
                continue
            smaller, larger = (a, b) if instance_info[a]["n_gaussians"] <= instance_info[b]["n_gaussians"] else (b, a)
            s_pts = means_np[instance_ids == smaller]
            n_sample = min(500, len(s_pts))
            idx = np.random.choice(len(s_pts), n_sample, replace=False)
            dists, _ = trees[larger].query(s_pts[idx])
            frac_close = float((dists < 0.05).mean())
            if frac_close > 0.7:
                merge_map[smaller] = larger

    if merge_map:
        for old in list(merge_map.keys()):
            new = merge_map[old]
            while new in merge_map:
                new = merge_map[new]
            merge_map[old] = new

        for old, new in merge_map.items():
            instance_ids[instance_ids == old] = new
            old_info = instance_info.pop(old)
            new_info = instance_info[new]
            all_pts = means_np[instance_ids == new]
            new_info["n_gaussians"] = int(len(all_pts))
            new_info["center"] = all_pts.mean(axis=0).tolist()
            new_info["bbox_min"] = all_pts.min(axis=0).tolist()
            new_info["bbox_max"] = all_pts.max(axis=0).tolist()
            print(f"  Merged {old_info['label']} into "
                  f"{new_info['label']} (point cloud overlap)")

    n_merged = len(merge_map)
    if n_merged:
        print(f"  {n_merged} spatial merges ({time.time()-t_sm:.1f}s)")
    else:
        print(f"  No spatial merges needed ({time.time()-t_sm:.1f}s)")

    # VLM 3D merge: ask VLM whether nearby instance pairs should merge,
    # using RGB + seg view + panorama as context. Run 3 iterations to
    # allow cascading merges.
    if not args.skip_vlm_merge:
        t_gm = time.time()
        print("\n[4b/5] VLM 3D merge (3 iterations)...")
        instance_ids, instance_info = vlm_merge_3d(
            means_np, instance_ids, instance_info, seg_dir,
            ply_path, cameras, n_iters=3)
        print(f"  VLM 3D merge total: {time.time()-t_gm:.1f}s")
    else:
        print("\n[4b/5] VLM merge: skipped (--skip-vlm-merge)")

    # VLM 3D filter: verify each instance, remove structural/noise/mislabeled
    if not args.skip_vlm_merge:
        t_gf = time.time()
        print("\n[4c/5] VLM 3D filter (verify instances)...")
        instance_ids, instance_info = vlm_filter_3d(
            means_np, instance_ids, instance_info, seg_dir, cameras)
        print(f"  VLM 3D filter: {time.time()-t_gf:.1f}s")

    # Legacy VLM merge (disabled)
    if False:
        t2 = time.time()
        print("\n[4b/5] VLM merge check for cross-category components...")
        instance_ids, instance_info = vlm_merge_components(
            means_np, instance_ids, instance_info, seg_dir)
        print(f"  {time.time()-t2:.0f}s")
    else:
        print("\n[4b/5] VLM merge: skipped (--skip-vlm-merge)")

    # Save labels first (needed by parallel renderers)
    np.save(seg_dir / "gaussian_labels.npy", instance_ids)
    json.dump({str(k): v for k, v in instance_info.items()},
              open(seg_dir / "instance_info.json", "w"), indent=2)

    # Parallel render (optional — only for visualization, not needed by pipeline)
    if not args.skip_viz_render:
        t3 = time.time()
        print(f"\n[5/5] Rendering {len(cameras)} views on {args.num_gpus} GPUs...")
        render_parallel(cameras, ply_path,
                        seg_dir / "gaussian_labels.npy",
                        seg_dir / "instance_info.json",
                        viz_dir, num_gpus=args.num_gpus)

        n_inst = len(instance_info)
        colors = get_distinct_colors(n_inst)
        instance_colors = {iid: colors[i] for i, iid in enumerate(sorted(instance_info.keys()))}
        make_grid_legend(cameras, instance_info, instance_colors, viz_dir)
        print(f"  {time.time()-t3:.0f}s")
    else:
        print("\n[5/5] Viz render: skipped (--skip-viz-render)")

    # Save scene.json
    world = json.load(open(marble_dir / "world_full.json"))
    scene_id = world.get("world_id", "unknown")[:12]
    scene = build_scene_json_from_instances(instance_info, cameras, scene_id)
    json.dump(scene, open(seg_dir / "scene.json", "w"), indent=2)

    print(f"\nTotal: {time.time()-t0:.0f}s — {len(instance_info)} instances")
    print(f"Grid: {viz_dir / 'grid_3d.png'}")


if __name__ == "__main__":
    main()
