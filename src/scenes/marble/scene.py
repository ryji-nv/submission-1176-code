"""
Marble scene builder: camera generation, gsplat rendering, and scene.json assembly.

Rendering uses the same functions as segment_scene.py
(look_at_matrix, render_view_rgbd, load_ply_gaussians) so that the rendered images
are pixel-identical to what the segmentation pipeline produces.

Coordinate convention
---------------------
PLY/Gaussian space is Y-up.  QA space is Z-up.  The per-scene vertical offset
``C`` is derived by matching seg-view camera look_at vectors between PLY and QA
frames.

All points (objects + cameras):  qa = (-gauss_x, -gauss_z, -gauss_y + C)
PLY Y points downward in Marble reconstructions, so negating it gives z-up.
"""

from __future__ import annotations

import json
import math
import os
import random
import sys
from pathlib import Path

import numpy as np


def _ground_offset(seg_cameras_json: str, qa_cameras: list[dict] | None = None) -> float:
    """Derive the vertical offset C from matching seg-to-QA camera look_at vectors."""
    with open(seg_cameras_json) as f:
        seg_cams = json.load(f)

    if qa_cameras:
        qa_by_name = {c["name"]: c for c in qa_cameras}
        offsets = []
        for sc in seg_cams:
            if sc["name"] in qa_by_name:
                qc = qa_by_name[sc["name"]]
                offsets.append(qc["look_at"][2] - sc["look_at"][1])
        if offsets:
            return float(np.median(offsets))

    return 0.0


def _obj_ply_to_qa(center, bbox_min, bbox_max, C: float) -> dict:
    """Convert one instance's PLY-space geometry to QA-space object dict fields.

    Same convention as cameras: qa = (-gauss_x, -gauss_z, -gauss_y + C).
    """
    return {
        "x": float(-center[0]),
        "y": float(-center[2]),
        "z": float(C - bbox_max[1]),
        "width": float(bbox_max[0] - bbox_min[0]),
        "length": float(bbox_max[2] - bbox_min[2]),
        "height": float(bbox_max[1] - bbox_min[1]),
    }


def _cam_ply_to_qa(pos, look_at, C: float):
    """Convert a PLY-space camera to QA-space (position, look_at)."""
    return (
        [-pos[0], -pos[2], -pos[1] + C],
        [-look_at[0], -look_at[2], -look_at[1] + C],
    )


def _qa_to_ply_point(p, C: float):
    """Reverse QA-space point to PLY-space.  Inverse of _cam_ply_to_qa."""
    return np.array([-p[0], C - p[2], -p[1]], dtype=np.float32)


def _sample_uniform_in_hull(points_2d: np.ndarray, scale: float, rng: random.Random) -> tuple[float, float]:
    """Uniformly sample a point inside the convex hull of *points_2d*, scaled by *scale*.

    Uses triangle-fan triangulation weighted by area for true uniform
    distribution (no center bias).
    """
    from scipy.spatial import ConvexHull

    centroid = points_2d.mean(axis=0)
    hull = ConvexHull(centroid + (points_2d - centroid) * scale)
    verts = hull.points[hull.vertices]
    fan_center = verts.mean(axis=0)
    n = len(verts)
    areas = []
    for i in range(n):
        a, b = verts[i], verts[(i + 1) % n]
        areas.append(0.5 * abs(np.cross(a - fan_center, b - fan_center)))
    total = sum(areas)
    r = rng.random() * total
    cum = 0.0
    for i, area in enumerate(areas):
        cum += area
        if cum >= r:
            a, b = verts[i], verts[(i + 1) % n]
            u, v = rng.random(), rng.random()
            if u + v > 1:
                u, v = 1 - u, 1 - v
            pt = fan_center + u * (a - fan_center) + v * (b - fan_center)
    return float(pt[0]), float(pt[1])
    return float(fan_center[0]), float(fan_center[1])


def _best_look_at(px, py, pz, objects, fov_deg, rng):
    """Pick a look-at target from near/mid/far objects that maximises FOV coverage.

    Objects are binned by XY distance into near (<1m), mid (1-2m), far (>2m).
    One bin is chosen randomly, then the best target within that bin is the one
    putting the most total objects inside the FOV cone.  The target Z is clamped
    to a plausible height range so the camera doesn't stare at the floor/ceiling.
    """
    obj_centers = np.array(
        [[o["x"], o["y"], o["z"] + o.get("height", 0) / 2] for o in objects]
    )
    cam = np.array([px, py, pz])
    vecs = obj_centers - cam
    dists_xy = np.sqrt((obj_centers[:, 0] - px) ** 2 + (obj_centers[:, 1] - py) ** 2)
    dists_3d = np.linalg.norm(vecs, axis=1)
    valid = dists_3d > 0.1
    vecs_n = vecs[valid] / dists_3d[valid, None]
    half_cos = math.cos(math.radians(fov_deg / 2))

    near_idx = [i for i in range(len(objects)) if dists_3d[i] > 0.1 and dists_xy[i] < 1.0]
    mid_idx = [i for i in range(len(objects)) if dists_3d[i] > 0.1 and 1.0 <= dists_xy[i] < 2.0]
    far_idx = [i for i in range(len(objects)) if dists_3d[i] > 0.1 and dists_xy[i] >= 2.0]

    bins = [(b, label) for b, label in [(near_idx, "near"), (mid_idx, "mid"), (far_idx, "far")] if b]
    if not bins:
        look = [2 * px - obj_centers[0, 0], 2 * py - obj_centers[0, 1], 2 * pz - obj_centers[0, 2]]
        return look, 0

    chosen_bin, _ = rng.choice(bins)

    best_dir, best_count = None, -1
    for idx in chosen_bin:
        fwd = vecs[idx] / dists_3d[idx]
        dots = vecs_n @ fwd
        in_fov = int((dots >= half_cos).sum())
        if in_fov > best_count:
            best_count = in_fov
            best_dir = fwd
            best_idx = idx

    target_z = obj_centers[best_idx, 2]
    target_z = max(pz - 1.0, min(target_z, pz - 0.1))

    target = np.array([
        obj_centers[best_idx, 0] + rng.gauss(0, 0.05),
        obj_centers[best_idx, 1] + rng.gauss(0, 0.05),
        target_z + rng.gauss(0, 0.02),
    ])
    look = [2 * px - target[0], 2 * py - target[1], 2 * pz - target[2]]
    return look, best_count


def sample_random_cameras(
    objects: list[dict],
    count: int,
    *,
    start_index: int = 0,
    seed: int = 42,
    cam_height_range: tuple[float, float] = (0.7, 1.75),
    hull_scale: float = 0.7,
    existing_cameras: list[dict] | None = None,
    min_pos_dist: float = 0.3,
    min_dir_cos: float = 0.85,
) -> list[dict]:
    """Sample cameras inside 70% convex hull, oriented to see the most objects.

    Rejects candidates too close in position or too similar in viewing
    direction to any already-accepted or *existing_cameras*.
    """
    rng = random.Random(seed)
    obj_xy = np.array([[o["x"], o["y"]] for o in objects], dtype=np.float64)
    floor_z = min(o["z"] for o in objects)

    accepted_pos: list[np.ndarray] = []
    accepted_dir: list[np.ndarray] = []
    if existing_cameras:
        for c in existing_cameras:
            p = np.array(c["position"])
            accepted_pos.append(p)
            la = np.array(c["look_at"])
            d = la - p
            n = np.linalg.norm(d)
            accepted_dir.append(d / n if n > 1e-9 else np.array([1, 0, 0]))

    def _is_distinct(pos, fwd):
        for ep, ed in zip(accepted_pos, accepted_dir):
            if np.linalg.norm(pos - ep) < min_pos_dist:
                cos = float(np.dot(fwd, ed))
                if cos > min_dir_cos:
                    return False
        return True

    cams: list[dict] = []
    idx = 0
    for _ in range(count * 5):
        if idx >= count:
            break
        px, py = _sample_uniform_in_hull(obj_xy, hull_scale, rng)
        pz = floor_z + rng.uniform(cam_height_range[0], cam_height_range[1])
        fov = 82.0
        look, _ = _best_look_at(px, py, pz, objects, fov, rng)

        pos = np.array([px, py, pz])
        fwd = np.array(look) - pos
        n = np.linalg.norm(fwd)
        fwd = fwd / n if n > 1e-9 else np.array([1, 0, 0])

        if not _is_distinct(pos, fwd):
            continue

        accepted_pos.append(pos)
        accepted_dir.append(fwd)
        cams.append({
            "name": f"rand_{start_index + idx:03d}",
            "position": [px, py, pz],
            "look_at": look,
            "horizontal_fov_deg": fov,
        })
        idx += 1
    return cams


def generate_cameras(
    seg_cameras: list[dict],
    C: float,
    objects: list[dict] | None = None,
    *,
    num_random: int = 64,
    seed: int = 42,
    cam_height_range: tuple[float, float] = (0.7, 1.75),
    hull_scale: float = 0.7,
) -> list[dict]:
    """Generate QA cameras.

    Gold cameras = seg views transformed to QA space.
    Random cameras = placed inside the 70 % convex hull of object
    centroids (XY), at 1.0–2.0 m height, looking at the nearest object.
    Falls back to PLY-origin cameras if no objects are provided.
    """
    rng = random.Random(seed)
    cameras: list[dict] = []

    for sc in seg_cameras:
        qa_pos, qa_look = _cam_ply_to_qa(sc["position"], sc["look_at"], C)
        cameras.append({
            "name": sc["name"],
            "position": qa_pos,
            "look_at": qa_look,
            "horizontal_fov_deg": sc.get("horizontal_fov_deg", 82.0),
        })

    if objects and len(objects) >= 3:
        cameras.extend(sample_random_cameras(
            objects, num_random, seed=seed,
            cam_height_range=cam_height_range, hull_scale=hull_scale,
        ))
    else:
        # Fallback: origin-based cameras (same as segment_scene.py)
        ply_eye = [0.0, 0.0, 0.0]
        for i in range(num_random):
            azimuth = rng.uniform(0, 360.0)
            elevation = rng.uniform(-25.0, 5.0)
            fov = rng.uniform(70.0, 100.0)
            az_rad = math.radians(azimuth)
            el_rad = math.radians(elevation)
            tx = math.sin(az_rad) * math.cos(el_rad) * 10.0
            ty = math.sin(el_rad) * 10.0
            tz = -math.cos(az_rad) * math.cos(el_rad) * 10.0
            qa_pos, qa_look = _cam_ply_to_qa(ply_eye, [tx, ty, tz], C)
            cameras.append({
                "name": f"rand_{i:03d}",
                "position": qa_pos,
                "look_at": qa_look,
                "horizontal_fov_deg": fov,
            })

    # Edge cameras: shifted along right/up vectors from each seg view camera
    base_cams = [c for c in cameras if c["name"].startswith("view_")]
    edge_dist = 1.50
    step_dist = 1.00
    for bc in base_cams:
        pos = bc["position"]
        la = bc["look_at"]
        fx = la[0] - pos[0]
        fy = la[1] - pos[1]
        fz = la[2] - pos[2]
        flen = math.sqrt(fx * fx + fy * fy + fz * fz)
        if flen < 1e-9:
            continue
        fx, fy, fz = fx / flen, fy / flen, fz / flen
        rx, ry = fy, -fx
        for direction, dx, dy, dz in [
            ("left", -rx * edge_dist, -ry * edge_dist, 0),
            ("right", rx * edge_dist, ry * edge_dist, 0),
        ]:
            new_pos = [pos[0] + dx, pos[1] + dy, pos[2] + dz]
            new_la = [la[0] + dx, la[1] + dy, la[2] + dz]
            cameras.append({
                "name": f"{bc['name']}_edge_{direction}",
                "position": new_pos,
                "look_at": new_la,
                "horizontal_fov_deg": bc["horizontal_fov_deg"],
                "edge_direction": direction,
            })

    # Stepped cameras: move-only, rotate-only, and move+rotate variants
    rot_angle = math.radians(20)
    cos_r, sin_r = math.cos(rot_angle), math.sin(rot_angle)
    for bc in base_cams:
        pos = bc["position"]
        la = bc["look_at"]
        fx = la[0] - pos[0]
        fy = la[1] - pos[1]
        flen = math.sqrt(fx * fx + fy * fy)
        if flen < 1e-9:
            continue
        fx, fy = fx / flen, fy / flen
        rx, ry = fy, -fx

        # Move-only: translate position + look_at by same offset
        height_step = 0.50
        for direction, dx, dy, dz in [
            ("forward", fx * step_dist, fy * step_dist, 0),
            ("left", -rx * step_dist, -ry * step_dist, 0),
            ("right", rx * step_dist, ry * step_dist, 0),
            ("up", 0, 0, height_step),
            ("down", 0, 0, -height_step),
        ]:
            new_pos = [pos[0] + dx, pos[1] + dy, pos[2] + dz]
            new_la = [la[0] + dx, la[1] + dy, la[2] + dz]
            cameras.append({
                "name": f"{bc['name']}_step_{direction}",
                "position": new_pos,
                "look_at": new_la,
                "horizontal_fov_deg": bc["horizontal_fov_deg"],
                "step_from": bc["name"],
                "step_direction": direction,
            })

        # Rotate-only: rotate look_at around position
        dfx, dfy = la[0] - pos[0], la[1] - pos[1]
        pitch_offset = 3.0
        for direction, c, s, dz_la in [
            ("yaw_left", cos_r, -sin_r, 0),
            ("yaw_right", cos_r, sin_r, 0),
            ("pitch_up", 1, 0, pitch_offset),
            ("pitch_down", 1, 0, -pitch_offset),
        ]:
            new_la = [pos[0] + dfx * c - dfy * s,
                      pos[1] + dfx * s + dfy * c, la[2] + dz_la]
            cameras.append({
                "name": f"{bc['name']}_step_{direction}",
                "position": list(pos),
                "look_at": new_la,
                "horizontal_fov_deg": bc["horizontal_fov_deg"],
                "step_from": bc["name"],
                "step_direction": direction,
            })

        # Move + rotate: translate forward + yaw
        fwd_pos = [pos[0] + fx * step_dist, pos[1] + fy * step_dist, pos[2]]
        for direction, c, s in [
            ("forward_yaw_left", cos_r, -sin_r),
            ("forward_yaw_right", cos_r, sin_r),
        ]:
            new_la = [fwd_pos[0] + dfx * c - dfy * s,
                      fwd_pos[1] + dfx * s + dfy * c, la[2]]
            cameras.append({
                "name": f"{bc['name']}_step_{direction}",
                "position": list(fwd_pos),
                "look_at": new_la,
                "horizontal_fov_deg": bc["horizontal_fov_deg"],
                "step_from": bc["name"],
                "step_direction": direction,
            })

    return cameras


def render_cameras(
    ply_path: str,
    cameras: list[dict],
    images_dir: str,
    C: float,
    labels_path: str | None = None,
    instance_info: dict | None = None,
    *,
    width: int = 1296,
    height: int = 968,
    seg_alpha_threshold: float = 0.02,
    batch_size: int = 16,
) -> None:
    """Render RGB + instance seg masks + seg visualizations via batched GPU passes.

    Uses gsplat's native camera batching to render multiple cameras in parallel
    on GPU, giving large speedups over sequential per-camera rendering.
    """
    os.makedirs(images_dir, exist_ok=True)

    have_labels = labels_path is not None and os.path.isfile(str(labels_path))
    jobs: list[dict] = []
    for cam in cameras:
        rgb_p = os.path.join(images_dir, f"{cam['name']}.jpg")
        seg_p = os.path.join(images_dir, f"{cam['name']}_seg.npy")
        rgb_ok = os.path.isfile(rgb_p)
        seg_ok = os.path.isfile(seg_p)
        if have_labels and (not rgb_ok or not seg_ok):
            do_rgb = do_seg = True
        else:
            do_rgb = not rgb_ok
            do_seg = have_labels and not seg_ok
        is_view = (
            cam["name"].startswith("view_")
            and "_edge_" not in cam["name"]
            and "_step_" not in cam["name"]
        )
        vis_p = os.path.join(images_dir, f"{cam['name']}_seg_vis.png")
        do_vis = is_view and instance_info is not None and not os.path.isfile(vis_p)
        jobs.append({"cam": cam, "rgb_path": rgb_p, "seg_path": seg_p,
                     "vis_path": vis_p, "do_rgb": do_rgb, "do_seg": do_seg,
                     "do_vis": do_vis})

    rgb_jobs = [j for j in jobs if j["do_rgb"]]
    seg_jobs = [j for j in jobs if j["do_seg"]]
    vis_jobs = [j for j in jobs if j["do_vis"]]

    if not rgb_jobs and not seg_jobs and not vis_jobs:
        print(f"  [render] all {len(cameras)} images + masks cached")
        return

    from src.scenes.marble.segment_scene import load_ply_gaussians, look_at_matrix

    import torch
    from gsplat import rasterization
    from PIL import Image as PILImage

    gaussians = load_ply_gaussians(ply_path)

    vis_colors = iid_to_color = iid_lut_gpu = labels_clamped = None
    labels = None
    if (seg_jobs or vis_jobs) and have_labels:
        labels = np.load(labels_path).astype(np.int32)

    if (vis_jobs or seg_jobs) and instance_info and labels is not None:
        sorted_iids = sorted(int(k) for k in instance_info)
        dist_colors = _get_distinct_colors(len(sorted_iids))
        iid_to_color = {iid: c for iid, c in zip(sorted_iids, dist_colors)}

        max_iid = max(sorted_iids) + 1
        color_lut = np.zeros((max_iid + 1, 3), dtype=np.float32)
        for iid, color in iid_to_color.items():
            color_lut[iid] = color
        color_lut_t = torch.tensor(color_lut, device="cuda")
        labels_clamped = torch.tensor(
            np.clip(labels, 0, max_iid), device="cuda", dtype=torch.long
        )
        vis_colors = color_lut_t[labels_clamped]
        bg_mask = torch.tensor(labels <= 0, device="cuda")
        gray = gaussians["colors"].mean(dim=-1, keepdim=True).expand_as(gaussians["colors"]) * 0.4
        vis_colors[bg_mask] = gray[bg_mask]

        iid_lut_gpu = torch.tensor(sorted_iids, device="cuda", dtype=torch.int32)

    from PIL import ImageDraw, ImageFont
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 15)
    except OSError:
        font = ImageFont.load_default()

    up = np.array([0, 1, 0], dtype=np.float32)
    cx_cam, cy_cam = width / 2.0, height / 2.0

    def _cam_mats(cam):
        ply_eye = _qa_to_ply_point(cam["position"], C)
        ply_target = _qa_to_ply_point(cam["look_at"], C)
        fov = cam["horizontal_fov_deg"]
        fx = fy = (width / 2.0) / math.tan(math.radians(fov / 2.0))
        c2w = look_at_matrix(ply_eye, ply_target, up)
        viewmat = torch.linalg.inv(c2w)
        K = torch.tensor([[fx, 0, cx_cam], [0, fy, cy_cam], [0, 0, 1]],
                         dtype=torch.float32, device="cuda")
        return viewmat, K, c2w, fx, fy

    import time as _time
    _t_total = _time.perf_counter()

    # ── Phase 1: batched RGB ────────────────────────────────────────────
    if rgb_jobs:
        _t0 = _time.perf_counter()
        print(f"  [render] RGB: {len(rgb_jobs)} cameras (batch={batch_size})")
        bg = torch.ones(3, device="cuda")
        for s in range(0, len(rgb_jobs), batch_size):
            chunk = rgb_jobs[s : s + batch_size]
            vms, ks = zip(*[_cam_mats(j["cam"])[:2] for j in chunk])
            with torch.no_grad():
                renders, alphas, _ = rasterization(
                    means=gaussians["means"], quats=gaussians["quats"],
                    scales=gaussians["scales"], opacities=gaussians["opacities"],
                    colors=gaussians["colors"],
                    viewmats=torch.stack(vms), Ks=torch.stack(ks),
                    width=width, height=height, sh_degree=None,
                )
            for i, j in enumerate(chunk):
                img = renders[i].clamp(0, 1)
                a = alphas[i]
                j["cam"]["mean_alpha"] = float(a[:, :, 0].mean().item())
                out = (img * a + bg * (1 - a)).cpu().numpy()
                PILImage.fromarray((out * 255).astype(np.uint8)).save(
                    j["rgb_path"], quality=95)
            del renders, alphas
        torch.cuda.empty_cache()
        print(f"  [render] RGB done in {_time.perf_counter() - _t0:.1f}s")

    # ── Phase 2: batched seg (per-instance, all cameras) ────────────────
    if seg_jobs and iid_lut_gpu is not None:
        _t0 = _time.perf_counter()
        n_seg = len(seg_jobs)
        print(f"  [render] Seg: {n_seg} cameras × {len(iid_lut_gpu)} instances "
              f"(batch={batch_size})")
        seg_vms, seg_ks = zip(*[_cam_mats(j["cam"])[:2] for j in seg_jobs])
        seg_vms, seg_ks = list(seg_vms), list(seg_ks)
        best_vis = torch.zeros(n_seg, height, width, device="cuda")
        seg_masks_t = torch.zeros(n_seg, height, width, device="cuda", dtype=torch.int32)
        black = torch.zeros_like(gaussians["colors"])

        for idx in range(len(iid_lut_gpu)):
            iid = int(iid_lut_gpu[idx].item())
            inst_mask = labels_clamped == iid
            if inst_mask.sum() == 0:
                continue
            inst_colors = black.clone()
            inst_colors[inst_mask] = 1.0
            for cs in range(0, n_seg, batch_size):
                ce = min(cs + batch_size, n_seg)
                with torch.no_grad():
                    ri, _, _ = rasterization(
                means=gaussians["means"], quats=gaussians["quats"],
                        scales=gaussians["scales"], opacities=gaussians["opacities"],
                        colors=inst_colors,
                        viewmats=torch.stack(seg_vms[cs:ce]),
                        Ks=torch.stack(seg_ks[cs:ce]),
                width=width, height=height, sh_degree=None,
            )
                v = ri[:, :, :, 0]
                better = (v > best_vis[cs:ce]) & (v >= seg_alpha_threshold)
                seg_masks_t[cs:ce][better] = iid
                best_vis[cs:ce][better] = v[better]
                del ri

        masks_np = seg_masks_t.cpu().numpy()
        for i, j in enumerate(seg_jobs):
            np.save(j["seg_path"], masks_np[i])
        del best_vis, seg_masks_t, masks_np, black
        torch.cuda.empty_cache()
        print(f"  [render] Seg done in {_time.perf_counter() - _t0:.1f}s")

    # ── Phase 3: batched vis ────────────────────────────────────────────
    seg_view_imgs: list[tuple[str, PILImage.Image]] = []
    if vis_jobs and vis_colors is not None:
        _t0 = _time.perf_counter()
        print(f"  [render] Vis: {len(vis_jobs)} cameras")
        bg_gray = torch.tensor([0.85, 0.85, 0.85], device="cuda")
        for s in range(0, len(vis_jobs), batch_size):
            chunk = vis_jobs[s : s + batch_size]
            cam_data = [_cam_mats(j["cam"]) for j in chunk]
            vms = [d[0] for d in cam_data]
            ks = [d[1] for d in cam_data]
            with torch.no_grad():
                renders, alphas, _ = rasterization(
                    means=gaussians["means"], quats=gaussians["quats"],
                    scales=gaussians["scales"], opacities=gaussians["opacities"],
                    colors=vis_colors,
                    viewmats=torch.stack(vms), Ks=torch.stack(ks),
                    width=width, height=height, sh_degree=None,
                )
            for i, j in enumerate(chunk):
                img = renders[i].clamp(0, 1)
                out = (img * alphas[i] + bg_gray * (1 - alphas[i])).cpu().numpy()
                pil = PILImage.fromarray((out * 255).astype(np.uint8))
                if instance_info:
                    draw = ImageDraw.Draw(pil)
                    c2w_i = cam_data[i][2]
                    fx_i, fy_i = cam_data[i][3], cam_data[i][4]
                    w2c = torch.linalg.inv(c2w_i).cpu().numpy()
                    drawn: list[tuple[int, int]] = []
                    for iid_str, info in sorted(instance_info.items(),
                                                key=lambda kv: -kv[1]["n_gaussians"]):
                        center = np.array(info["center"] + [1.0], dtype=np.float32)
                        p = w2c @ center
                        if p[2] <= 0.1:
                            continue
                        u = int(fx_i * p[0] / p[2] + cx_cam)
                        v_px = int(fy_i * p[1] / p[2] + cy_cam)
                        if u < 10 or u > width - 10 or v_px < 10 or v_px > height - 10:
                            continue
                        if any(abs(u - du) < 100 and abs(v_px - dv) < 22
                               for du, dv in drawn):
                            continue
                        iid_int = int(iid_str)
                        clr = tuple(int(c * 255)
                                    for c in iid_to_color.get(iid_int, (1, 1, 1)))
                        bbox = draw.textbbox((u, v_px), info["label"], font=font)
                        tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
                        tx = max(2, min(u - tw // 2, width - tw - 4))
                        ty = max(2, min(v_px - th // 2, height - th - 4))
                        draw.rectangle([tx - 2, ty - 2, tx + tw + 2, ty + th + 2],
                                       fill=(0, 0, 0, 200))
                        draw.text((tx, ty), info["label"], fill=clr, font=font)
                        drawn.append((u, v_px))
                pil.save(j["vis_path"])
                seg_view_imgs.append((j["cam"]["name"], pil))
            del renders, alphas
        torch.cuda.empty_cache()
        print(f"  [render] Vis done in {_time.perf_counter() - _t0:.1f}s")

    # Seg vis grid + legend (only if we rendered any)
    grid_path = os.path.join(images_dir, "seg_vis_grid.png")
    if seg_view_imgs and not os.path.isfile(grid_path):
        cols = 4
        rows = math.ceil(len(seg_view_imgs) / cols)
        tw_g, th_g = 648, 484
        grid = PILImage.new("RGB", (cols * tw_g, rows * th_g), (30, 30, 30))
        draw_g = ImageDraw.Draw(grid)
        try:
            font_g = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 18)
        except OSError:
            font_g = ImageFont.load_default()
        for i, (name, img) in enumerate(seg_view_imgs):
            r, c = divmod(i, cols)
            grid.paste(img.resize((tw_g, th_g), PILImage.LANCZOS), (c * tw_g, r * th_g))
            draw_g.text((c * tw_g + 8, r * th_g + 4), name, fill=(255, 255, 0), font=font_g)
        grid.save(grid_path)

        if instance_info:
            n_inst = len(instance_info)
            row_h = 24
            legend = PILImage.new("RGB", (300, 40 + n_inst * row_h), (30, 30, 30))
            draw_l = ImageDraw.Draw(legend)
            try:
                font_l = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 14)
            except OSError:
                font_l = ImageFont.load_default()
            draw_l.text((10, 5), "Instance Legend", fill=(255, 255, 255), font=font_l)
            for idx, (iid_str, info) in enumerate(sorted(instance_info.items(), key=lambda kv: int(kv[0]))):
                y = 30 + idx * row_h
                color_rgb = tuple(int(c * 255) for c in iid_to_color.get(int(iid_str), (1, 1, 1)))
                draw_l.rectangle([10, y, 30, y + row_h - 4], fill=color_rgb)
                draw_l.text((36, y + 2), f"{info['label']} ({info['n_gaussians']:,})",
                            fill=(255, 255, 255), font=font_l)
            legend.save(os.path.join(images_dir, "seg_vis_legend.png"))

    del gaussians
    if vis_colors is not None:
        del vis_colors
    torch.cuda.empty_cache()
    print(f"  [render] {len(cameras)} cameras done "
          f"({len(rgb_jobs)} rgb, {len(seg_jobs)} seg, {len(vis_jobs)} vis) "
          f"in {_time.perf_counter() - _t_total:.1f}s")


def _get_distinct_colors(n: int) -> list[tuple[float, float, float]]:
    """Generate n visually distinct colors using golden-ratio hue stepping."""
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
        colors.append((r + m, g + m, b + m))
    return colors


def build_scene_json(
    instance_info: dict,
    cameras: list[dict],
    scene_id: str,
    C: float,
    gaussian_xyz: np.ndarray | None = None,
    gaussian_labels: np.ndarray | None = None,
    bbox_percentile: float = 2.0,
) -> dict:
    """Build a QA-compatible scene.json from upstream instance_info + QA cameras.

    When *gaussian_xyz* and *gaussian_labels* are provided, recomputes tight
    per-instance bboxes using percentile trimming to exclude outlier Gaussians.
    """
    objects = []
    for _iid, info in sorted(instance_info.items(), key=lambda kv: int(kv[0])):
        bmin, bmax = info["bbox_min"], info["bbox_max"]
        if gaussian_xyz is not None and gaussian_labels is not None:
            iid = int(_iid)
            pts = gaussian_xyz[gaussian_labels == iid]
            if len(pts) > 20:
                bmin = [float(np.percentile(pts[:, a], bbox_percentile)) for a in range(3)]
                bmax = [float(np.percentile(pts[:, a], 100 - bbox_percentile)) for a in range(3)]
        fields = _obj_ply_to_qa(info["center"], bmin, bmax, C)
        objects.append({
            "id": info["label"],
            "label": info["category"],
            "type": info["category"],
            **fields,
            "ply_center": info["center"],
            "rotation_z": 0.0,
            "n_gaussians": info["n_gaussians"],
        })

    room_bounds = None
    if objects:
        xs = [o["x"] for o in objects]
        ys = [o["y"] for o in objects]
        ws = [o.get("width", 0) / 2 for o in objects]
        ls = [o.get("length", 0) / 2 for o in objects]
        zs = [o.get("z", 0) + o.get("height", 0) for o in objects]
        room_bounds = [
            min(x - w for x, w in zip(xs, ws)),
            min(y - l for y, l in zip(ys, ls)),
            max(x + w for x, w in zip(xs, ws)),
            max(y + l for y, l in zip(ys, ls)),
            max(zs),
        ]

    from src.scenes.marble.segment_scene import look_at_matrix
    up = np.array([0, 1, 0], dtype=np.float32)
    W, H = 1296, 968
    cam_list = []
    for c in cameras:
        ply_eye = _qa_to_ply_point(c["position"], C)
        ply_target = _qa_to_ply_point(c["look_at"], C)
        fov = c["horizontal_fov_deg"]
        fx = fy = (W / 2.0) / math.tan(math.radians(fov / 2.0))
        c2w = look_at_matrix(ply_eye, ply_target, up).cpu().numpy().tolist()
        # look_at_matrix uses OpenGL convention (c2w[:3,2] = -fwd), but
        # gsplat renders positive camera-space Z as "in front". This makes
        # the rendering look_at point OPPOSITE to the actual viewing direction.
        # Reflect through position so _compute_depth gives positive depth
        # for visible objects, matching the SAGE/SceneSmith convention.
        pos_qa = c["position"]
        la_qa = c["look_at"]
        view_la = [2 * pos_qa[i] - la_qa[i] for i in range(3)]
        cam_dict = {
            "name": c["name"],
            "position": pos_qa,
            "look_at": view_la,
            "horizontal_fov_deg": fov,
            "c2w": c2w,
            "fx": fx, "fy": fy,
            "cx": W / 2.0, "cy": H / 2.0,
            "width": W, "height": H,
            **{k: c[k] for k in ("edge_direction", "step_from", "step_direction") if k in c},
        }
        cam_list.append(cam_dict)

    return {
        "scene_id": scene_id,
        "scene_type": "marble",
        "objects": objects,
        "cameras": cam_list,
        "room_bounds": room_bounds,
    }


def validate_object_dimensions(
    objects: list[dict], *, model: str = os.environ.get("VLM_MODEL", "Qwen3-VL-235B-A22B-Thinking"), batch_size: int = 20,
) -> set[str]:
    """Ask VLM whether each object's longest dimension is physically plausible.

    Returns set of object IDs flagged as implausible.
    """
    import requests

    api_key = os.environ.get("VLM_API_KEY", "")
    if not api_key:
        print("  [dim-filter] VLM_API_KEY not set, skipping")
        return set()

    implausible: set[str] = set()
    for start in range(0, len(objects), batch_size):
        batch = objects[start : start + batch_size]
        lines = []
        for i, o in enumerate(batch, 1):
            longest_cm = round(
                max(o.get("width", 0), o.get("length", 0), o.get("height", 0)) * 100, 1
            )
            lines.append(f"{i}. {o['id']}: longest_dim={longest_cm}cm")

        prompt = (
            "For each object below, answer YES if its longest dimension is "
            "physically plausible for a real-world instance of that object type, "
            "or NO if it seems wrong (too large or too small). "
            "Reply with exactly one line per object: \"id: YES\" or \"id: NO\".\n\n"
            + "\n".join(lines)
        )

        try:
            resp = requests.post(
                os.environ.get("VLM_API_BASE", "https://example.com/v1/chat/completions"),
                headers={
                    "Authorization": f"Bearer {api_key}",
                    "Content-Type": "application/json",
                },
                json={
                    "model": model,
                    "messages": [{"role": "user", "content": prompt}],
                    "temperature": 0.0,
                    "max_completion_tokens": 512,
                },
                timeout=30,
            )
            resp.raise_for_status()
            text = resp.json()["choices"][0]["message"]["content"]
        except Exception as e:
            print(f"  [dim-filter] VLM call failed: {e}")
            continue

        for line in text.strip().splitlines():
            line = line.strip()
            if ": NO" in line.upper():
                obj_id = line.split(":")[0].strip().lstrip("0123456789. ")
                if any(o["id"] == obj_id for o in batch):
                    implausible.add(obj_id)

    if implausible:
        print(f"  [dim-filter] excluded {len(implausible)}: {sorted(implausible)}")
    else:
        print("  [dim-filter] all objects plausible")
    return implausible


def _render_2d_numbered_seg(
    seg_dir: str,
    view_indices: list[int],
    label_maps: list[dict] | None = None,
) -> list[tuple[str, str]]:
    """Overlay numbered 2D SAM3 instance masks on RGB images.

    Works with raw SAM3 output (before 3D labeling). Each mask ID gets a
    unique color and its number drawn at the mask centroid.
    Returns list of (rgb_path, seg_vis_path) pairs.
    """
    import os
    import json as _json
    import numpy as np
    from PIL import Image, ImageDraw, ImageFont
    import colorsys

    views_dir = os.path.join(seg_dir, "views")
    masks_dir = os.path.join(seg_dir, "instance_masks")
    merged_masks_dir = os.path.join(seg_dir, "instance_masks_merged")
    cameras = _json.load(open(os.path.join(views_dir, "cameras.json")))
    out_dir = os.path.join(seg_dir, "viz_numbered")
    os.makedirs(out_dir, exist_ok=True)

    try:
        font = ImageFont.truetype(
            "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 18)
    except OSError:
        font = ImageFont.load_default()

    # Collect all unique mask IDs across requested views for consistent coloring
    all_ids: set[int] = set()
    view_data = []
    for vi in view_indices:
        if vi >= len(cameras):
            continue
        cam = cameras[vi]
        # Prefer merged masks if available
        _merged_p = os.path.join(merged_masks_dir, f"view_{vi:03d}_mask.npy")
        mask_path = _merged_p if os.path.isfile(_merged_p) else \
            os.path.join(masks_dir, f"view_{vi:03d}_mask.npy")
        lbl_path = os.path.join(masks_dir, f"view_{vi:03d}_labels.json")
        if not os.path.isfile(mask_path):
            continue
        mask = np.load(mask_path)
        lbl = _json.load(open(lbl_path)) if os.path.isfile(lbl_path) else {}
        if label_maps is not None and vi < len(label_maps):
            lbl = label_maps[vi]
        ids_in_view = set(int(x) for x in np.unique(mask) if x > 0)
        all_ids |= ids_in_view
        rgb_path = cam.get("rgb_path", os.path.join(views_dir, f"view_{vi:03d}.png"))
        view_data.append((vi, cam, mask, lbl, rgb_path, ids_in_view))

    # Assign colors
    colors = {}
    for i, mid in enumerate(sorted(all_ids)):
        h = (i * 0.618033988749895) % 1.0
        r, g, b = colorsys.hsv_to_rgb(h, 0.85, 0.95)
        colors[mid] = (int(r * 255), int(g * 255), int(b * 255))

    pairs = []
    for vi, cam, mask, lbl, rgb_path, ids_in_view in view_data:
        rgb = Image.open(rgb_path).convert("RGB")
        overlay = np.array(rgb, dtype=np.float32)
        seg_layer = np.zeros_like(overlay)

        for mid in ids_in_view:
            px = mask == mid
            if px.sum() < 50:
                continue
            c = colors.get(mid, (128, 128, 128))
            seg_layer[px] = c

        blended = (overlay * 0.5 + seg_layer * 0.5).astype(np.uint8)
        # Darken areas with no mask
        no_mask = mask == 0
        blended[no_mask] = (overlay[no_mask] * 0.3).astype(np.uint8)

        img = Image.fromarray(blended)
        draw = ImageDraw.Draw(img)

        # Draw mask ID numbers at centroids
        drawn = []
        for mid in sorted(ids_in_view, key=lambda m: -int((mask == m).sum())):
            ys, xs = np.where(mask == mid)
            if len(xs) < 50:
                continue
            cx, cy = int(xs.mean()), int(ys.mean())
            too_close = any(abs(cx - dx) < 60 and abs(cy - dy) < 22
                           for dx, dy in drawn)
            if too_close:
                continue
            label_text = str(mid)
            c = colors.get(mid, (255, 255, 255))
            bbox = draw.textbbox((cx, cy), label_text, font=font)
            tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
            tx = max(2, min(cx - tw // 2, img.width - tw - 4))
            ty = max(2, min(cy - th // 2, img.height - th - 4))
            draw.rectangle([tx - 2, ty - 2, tx + tw + 2, ty + th + 2],
                           fill=(0, 0, 0, 200))
            draw.text((tx, ty), label_text, fill=c, font=font)
            drawn.append((cx, cy))

        seg_vis_path = os.path.join(out_dir, f"view_{vi:03d}_numbered.png")
        img.save(seg_vis_path)
        pairs.append((rgb_path, seg_vis_path))

    return pairs


def vlm_merge_and_label_2d(
    seg_dir: str,
    *,
    model: str = os.environ.get("VLM_MODEL", "Qwen3-VL-235B-A22B-Thinking"),
):
    """2-stage VLM using raw 2D SAM3 masks (before 3D labeling).

    For each of 12 views in parallel:
      - Primary: that view's numbered seg overlay + RGB (high detail)
      - Context: panorama image (low detail)
    Stage 1: merge/remove per view, aggregate votes.
    Stage 2: label per view, aggregate majority labels.

    Saves: instance_masks_merged/, viz_numbered/ (3 stages), merged_labels.json.
    """
    import base64
    import json as _json
    import os
    import colorsys
    from collections import Counter

    import numpy as np
    import requests as _req
    from concurrent.futures import ThreadPoolExecutor, as_completed
    from PIL import Image, ImageDraw, ImageFont

    api_key = os.environ.get("VLM_API_KEY", "")
    if not api_key:
        print("  [merge2d] VLM_API_KEY not set, skipping")
        return

    masks_dir = os.path.join(seg_dir, "instance_masks")
    views_dir = os.path.join(seg_dir, "views")
    viz_dir = os.path.join(seg_dir, "viz_numbered")
    merged_dir = os.path.join(seg_dir, "instance_masks_merged")
    os.makedirs(viz_dir, exist_ok=True)
    os.makedirs(merged_dir, exist_ok=True)

    cameras = _json.load(open(os.path.join(views_dir, "cameras.json")))
    n_views = len(cameras)
    panorama_path = os.path.join(os.path.dirname(seg_dir), "panorama.jpg")

    def _encode(path, detail="low"):
        if not os.path.isfile(path):
            return None
        with open(path, "rb") as f:
            b64 = base64.b64encode(f.read()).decode()
        ext = "png" if path.endswith(".png") else "jpeg"
        return {"type": "image_url",
                "image_url": {"url": f"data:image/{ext};base64,{b64}", "detail": detail}}

    def _vlm(content, max_tokens=1024):
        resp = _req.post(
            os.environ.get("VLM_API_BASE", "https://example.com/v1/chat/completions"),
            headers={"Authorization": f"Bearer {api_key}",
                     "Content-Type": "application/json"},
            json={"model": model,
                  "messages": [{"role": "user", "content": content}],
                  "temperature": 0.0, "max_completion_tokens": max_tokens},
            timeout=120,
        )
        resp.raise_for_status()
        return resp.json()["choices"][0]["message"]["content"].strip()

    def _json_parse(text):
        s, e = text.find("{"), text.rfind("}") + 1
        if s < 0 or e <= s:
            return None
        try:
            return _json.loads(text[s:e])
        except _json.JSONDecodeError:
            return None

    # ── Helper: render numbered overlay from masks ────────────────────
    try:
        font = ImageFont.truetype(
            "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 16)
    except OSError:
        font = ImageFont.load_default()

    def _overlay(mask, labels_dict, rgb_path, use_labels=False):
        rgb = Image.open(rgb_path).convert("RGB")
        ov = np.array(rgb, dtype=np.float32)
        seg = np.zeros_like(ov)
        ids = sorted(set(int(x) for x in np.unique(mask) if x > 0))
        colors = {}
        for i, mid in enumerate(ids):
            h = (i * 0.618) % 1.0
            r, g, b = colorsys.hsv_to_rgb(h, 0.85, 0.95)
            colors[mid] = (int(r * 255), int(g * 255), int(b * 255))
            seg[mask == mid] = colors[mid]
        bl = (ov * 0.5 + seg * 0.5).astype(np.uint8)
        bl[mask == 0] = (ov[mask == 0] * 0.3).astype(np.uint8)
        img = Image.fromarray(bl)
        draw = ImageDraw.Draw(img)
        drawn = []
        for mid in sorted(ids, key=lambda m: -int((mask == m).sum())):
            ys, xs = np.where(mask == mid)
            if len(xs) < 50:
                continue
            cx, cy = int(xs.mean()), int(ys.mean())
            if any(abs(cx - dx) < 60 and abs(cy - dy) < 22 for dx, dy in drawn):
                continue
            txt = labels_dict.get(str(mid), labels_dict.get(mid, str(mid))) \
                if use_labels else str(mid)
            c = colors.get(mid, (255, 255, 255))
            bbox = draw.textbbox((cx, cy), txt, font=font)
            tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
            tx = max(2, min(cx - tw // 2, img.width - tw - 4))
            ty = max(2, min(cy - th // 2, img.height - th - 4))
            draw.rectangle([tx - 2, ty - 2, tx + tw + 2, ty + th + 2],
                           fill=(0, 0, 0, 200))
            draw.text((tx, ty), txt, fill=c, font=font)
            drawn.append((cx, cy))
        return img

    # ── Stage 1 viz + pre-filter (parallel) ──────────────────────────
    # Combined: save SAM3 originals, copy masks to merged_dir with
    # tiny fragments (<500px) removed, generate filtered overlays.
    # Originals in masks_dir are untouched.
    import io
    MIN_PX = 500
    seg_paths: dict[int, str] = {}
    _auto_counts = [0]

    def _viz_and_filter(vi):
        mask_p = os.path.join(masks_dir, f"view_{vi:03d}_mask.npy")
        lbl_p = os.path.join(masks_dir, f"view_{vi:03d}_labels.json")
        if not os.path.isfile(mask_p):
            return vi, None, 0
        rgb_p = cameras[vi].get("rgb_path",
                                os.path.join(views_dir, f"view_{vi:03d}.png"))
        orig_mask = np.load(mask_p)
        orig_img = _overlay(orig_mask, {}, rgb_p, use_labels=False)
        orig_img.save(os.path.join(viz_dir, f"view_{vi:03d}_numbered.png"))

        mask = orig_mask.copy()
        lbl = {int(k): v for k, v in _json.load(open(lbl_p)).items()} \
            if os.path.isfile(lbl_p) else {}
        n_removed = 0
        for mid in sorted(set(int(x) for x in np.unique(mask) if x > 0)):
            if int((mask == mid).sum()) < MIN_PX:
                mask[mask == mid] = 0
                lbl.pop(mid, None)
                n_removed += 1
        np.save(os.path.join(merged_dir, f"view_{vi:03d}_mask.npy"), mask)
        _json.dump({str(k): v for k, v in lbl.items()},
                   open(os.path.join(merged_dir,
                                     f"view_{vi:03d}_labels.json"), "w"),
                   indent=2)
        if n_removed:
            filt_img = _overlay(mask, {}, rgb_p, use_labels=False)
            p = os.path.join(viz_dir, f"view_{vi:03d}_numbered.png")
            filt_img.save(p)
            return vi, p, n_removed
        return vi, os.path.join(viz_dir, f"view_{vi:03d}_numbered.png"), 0

    print("  [merge2d] Stage 1 viz + pre-filter (parallel)...")
    with ThreadPoolExecutor(max_workers=n_views) as pool:
        for f in as_completed(
                [pool.submit(_viz_and_filter, vi) for vi in range(n_views)]):
            vi, path, cnt = f.result()
            if path:
                seg_paths[vi] = path
            _auto_counts[0] += cnt
    if _auto_counts[0]:
        print(f"  [merge2d] Pre-filter: auto-removed {_auto_counts[0]} "
              f"fragments <{MIN_PX}px")

    # ── Stage 1a: Merge+Remove proposals (per-view) ─────────────────

    def _isolated_b64(mask, mid, rgb_path):
        """Render isolated overlay for one mask ID (rest darkened)."""
        rgb = Image.open(rgb_path).convert("RGB")
        arr = np.array(rgb, dtype=np.float32)
        h = (mid * 0.618033988) % 1.0
        r, g, b = colorsys.hsv_to_rgb(h, 0.85, 0.95)
        color = np.array([r * 255, g * 255, b * 255])
        out = arr.copy()
        out[mask != mid] *= 0.25
        m = mask == mid
        out[m] = out[m] * 0.5 + color * 0.5
        buf = io.BytesIO()
        Image.fromarray(out.astype(np.uint8)).save(buf, format="PNG")
        return base64.b64encode(buf.getvalue()).decode()

    def _merge_one(vi):
        if vi not in seg_paths:
            return vi, None
        rgb_p = cameras[vi].get("rgb_path",
                                os.path.join(views_dir, f"view_{vi:03d}.png"))
        content = []
        img = _encode(seg_paths[vi], "high")
        if img:
            content.append(img)
        img = _encode(rgb_p, "high")
        if img:
            content.append(img)
        if os.path.isfile(panorama_path):
            img = _encode(panorama_path, "low")
            if img:
                content.append(img)

        mask = np.load(os.path.join(merged_dir, f"view_{vi:03d}_mask.npy"))
        view_ids = sorted(set(int(x) for x in np.unique(mask) if x > 0))
        if not view_ids:
            return vi, None

        content.append({"type": "text", "text": (
            "Image 1: instance segmentation with numbered IDs. "
            "Image 2: RGB of same view. "
            "Image 3: panorama of the full scene.\n\n"
            f"Instance IDs in this view: {view_ids}\n\n"
            "Which IDs should be MERGED?\n"
            "- Adjacent segments of the same object type "
            "(e.g. two cabinet doors next to each other, "
            "switches on one plate, parts of one appliance)\n"
            "- Sub-components attached to a parent object "
            "(e.g. a handle/knob on a door or appliance, "
            "a button on a machine — merge with the parent)\n\n"
            "Which IDs should be REMOVED?\n"
            "- Room structure: walls, floors, ceilings, "
            "countertops, baseboards, or any fixed surface\n"
            "- Very thin, narrow, or tiny objects that are "
            "hard to identify (wires, skewers, thin rods)\n"
            "- Noise, strips, or unidentifiable fragments\n"
            "- KEEP all recognizable objects: things you can "
            "pick up, move, open, or use — including fixtures "
            "like sinks, faucets, switches, outlets, appliances\n\n"
            '{"merge": [[1,2,3]], "remove": [5,8]}\n'
            '{"merge": [], "remove": []} if no changes.'
        )})
        try:
            return vi, _json_parse(_vlm(content))
        except Exception:
            return vi, None

    print(f"  [merge2d] Stage 1a: {n_views} merge proposals (per-view)...")
    per_view_merge: dict[int, dict] = {}

    with ThreadPoolExecutor(max_workers=12) as pool:
        futs = {pool.submit(_merge_one, vi): vi for vi in range(n_views)}
        for f in as_completed(futs):
            vi, data = f.result()
            if data:
                per_view_merge[vi] = data

    # ── Stage 1b: Verify each merge pair ──────────────────────────
    # For each proposed merge, show VLM isolated overlays of both
    # objects + full seg + RGB + panorama to confirm/reject.
    verify_tasks: list[tuple[int, int, int]] = []
    for vi, data in per_view_merge.items():
        for group in data.get("merge", []):
            ids = sorted(int(x) for x in group)
            for i in range(len(ids)):
                for j in range(i + 1, len(ids)):
                    verify_tasks.append((vi, ids[i], ids[j]))

    def _verify_merge(vi, id_a, id_b):
        mask = np.load(os.path.join(merged_dir, f"view_{vi:03d}_mask.npy"))
        rgb_p = cameras[vi].get(
            "rgb_path", os.path.join(views_dir, f"view_{vi:03d}.png"))
        iso_a = _isolated_b64(mask, id_a, rgb_p)
        iso_b = _isolated_b64(mask, id_b, rgb_p)
        content = [
            {"type": "image_url",
             "image_url": {"url": f"data:image/png;base64,{iso_a}",
                           "detail": "high"}},
            {"type": "image_url",
             "image_url": {"url": f"data:image/png;base64,{iso_b}",
                           "detail": "high"}},
        ]
        if vi in seg_paths:
            ctx = _encode(seg_paths[vi], "low")
            if ctx:
                content.append(ctx)
        rgb_img = _encode(rgb_p, "high")
        if rgb_img:
            content.append(rgb_img)
        if os.path.isfile(panorama_path):
            ctx = _encode(panorama_path, "low")
            if ctx:
                content.append(ctx)
        content.append({"type": "text", "text": (
            "Image 1: highlighted segment A.\n"
            "Image 2: highlighted segment B.\n"
            "Image 3: all numbered segments in this view.\n"
            "Image 4: RGB photo of this view.\n"
            "Image 5: panorama of the full scene.\n\n"
            "Should segments A and B be merged into one instance?\n"
            "MERGE if:\n"
            "- They are adjacent parts of the same object type "
            "(e.g. two cabinet doors side by side, parts of one "
            "appliance, switches on one plate)\n"
            "- One is a sub-component of the other (e.g. a "
            "handle/knob attached to a door or appliance)\n"
            "- They are the same object split by segmentation\n"
            "SEPARATE if:\n"
            "- They are clearly different, independent objects\n\n"
            "Reply with one word: MERGE or SEPARATE"
        )})
        try:
            answer = _vlm(content, max_tokens=16)
            return vi, id_a, id_b, "merge" in answer.lower()
        except Exception:
            return vi, id_a, id_b, True

    if verify_tasks:
        print(f"  [merge2d] Stage 1b: {len(verify_tasks)} merge "
              f"verifications...")
        confirmed: dict[int, set[tuple[int, int]]] = {}
        with ThreadPoolExecutor(max_workers=50) as pool:
            futs = [pool.submit(_verify_merge, vi, a, b)
                    for vi, a, b in verify_tasks]
            for f in as_completed(futs):
                vi, id_a, id_b, ok = f.result()
                if ok:
                    confirmed.setdefault(vi, set()).add((id_a, id_b))
        n_confirmed = sum(len(v) for v in confirmed.values())
        n_rejected = len(verify_tasks) - n_confirmed
        print(f"  [merge2d] Stage 1b: {n_confirmed} confirmed, "
              f"{n_rejected} rejected")
    else:
        confirmed = {}

    # ── Stage 1c: Check rejected-merge IDs for removal ──────────
    # For each ID that was part of a rejected merge, ask VLM if it's
    # a real object (keep) or noise/structure (remove).
    confirmed_ids: set[tuple[int, int]] = set()
    for vi, pairs in confirmed.items():
        for a, b in pairs:
            confirmed_ids.add((vi, a))
            confirmed_ids.add((vi, b))

    reject_check: set[tuple[int, int]] = set()
    for vi, id_a, id_b in verify_tasks:
        if (id_a, id_b) not in confirmed.get(vi, set()):
            reject_check.add((vi, id_a))
            reject_check.add((vi, id_b))
    reject_check -= confirmed_ids

    def _check_remove(vi, mid):
        mask = np.load(os.path.join(merged_dir, f"view_{vi:03d}_mask.npy"))
        rgb_p = cameras[vi].get(
            "rgb_path", os.path.join(views_dir, f"view_{vi:03d}.png"))
        iso = _isolated_b64(mask, mid, rgb_p)
        content = [
            {"type": "image_url",
             "image_url": {"url": f"data:image/png;base64,{iso}",
                           "detail": "high"}},
        ]
        if vi in seg_paths:
            ctx = _encode(seg_paths[vi], "low")
            if ctx:
                content.append(ctx)
        rgb_img = _encode(rgb_p, "high")
        if rgb_img:
            content.append(rgb_img)
        if os.path.isfile(panorama_path):
            ctx = _encode(panorama_path, "low")
            if ctx:
                content.append(ctx)
        content.append({"type": "text", "text": (
            "Image 1: a SINGLE highlighted instance (colored overlay "
            "on the RGB image).\n"
            "Image 2: all numbered segments in this view.\n"
            "Image 3: RGB photo of this view.\n"
            "Image 4: panorama of the full scene.\n\n"
            "Is this highlighted region a clearly recognizable, "
            "substantive OBJECT?\n"
            "- KEEP: objects you can pick up, move, open, or use "
            "(bowls, cups, appliances, switches, faucets, etc.)\n"
            "- REMOVE: room structure (walls, floors, ceilings, "
            "countertops, baseboards, any fixed surface), "
            "very thin/narrow objects (wires, skewers, rods), "
            "noise, or unidentifiable fragments\n\n"
            "Reply with one word: KEEP or REMOVE"
        )})
        try:
            answer = _vlm(content, max_tokens=16)
            return vi, mid, "remove" in answer.lower()
        except Exception:
            return vi, mid, False

    extra_removes: dict[int, set[int]] = {}
    if reject_check:
        print(f"  [merge2d] Stage 1c: {len(reject_check)} rejected-merge "
              f"IDs checked for removal...")
        with ThreadPoolExecutor(max_workers=50) as pool:
            futs = [pool.submit(_check_remove, vi, mid)
                    for vi, mid in reject_check]
            for f in as_completed(futs):
                vi, mid, should_remove = f.result()
                if should_remove:
                    extra_removes.setdefault(vi, set()).add(mid)
        n_extra = sum(len(v) for v in extra_removes.values())
        print(f"  [merge2d] Stage 1c: {n_extra} additional removes")

    # ── Apply verified merges + removes + 1c removes ─────────────
    total_merges = 0
    total_removes = 0
    for vi in range(n_views):
        mp = os.path.join(merged_dir, f"view_{vi:03d}_mask.npy")
        lp = os.path.join(merged_dir, f"view_{vi:03d}_labels.json")
        if not os.path.isfile(mp):
            continue
        mask = np.load(mp).copy()
        lbl = {int(k): v for k, v in _json.load(open(lp)).items()} \
            if os.path.isfile(lp) else {}

        data = per_view_merge.get(vi, {})
        view_confirmed = confirmed.get(vi, set())

        view_remap: dict[int, int] = {}
        for a, b in view_confirmed:
            view_remap[b] = a
        for mid in list(view_remap.keys()):
            target = view_remap[mid]
            while target in view_remap:
                target = view_remap[target]
            view_remap[mid] = target

        view_removes = set(int(r) for r in data.get("remove", []))
        view_removes |= extra_removes.get(vi, set())

        for old, new in view_remap.items():
            mask[mask == old] = new
            if old in lbl:
                if new not in lbl:
                    lbl[new] = lbl[old]
                lbl.pop(old, None)
        for r in view_removes:
            mask[mask == r] = 0
            lbl.pop(r, None)

        total_merges += len(view_remap)
        total_removes += len(view_removes)
        np.save(os.path.join(merged_dir, f"view_{vi:03d}_mask.npy"), mask)
        _json.dump({str(k): v for k, v in lbl.items()},
                   open(os.path.join(merged_dir,
                                     f"view_{vi:03d}_labels.json"), "w"),
                   indent=2)

    print(f"  [merge2d] Stage 1: {total_merges} verified merges, "
          f"{total_removes} removes (per-view)")

    # ── Save Stage 2 viz: post-merge numbered (parallel) ──────────────
    print("  [merge2d] Saving Stage 2 viz (post-merge numbered)...")
    merged_seg_paths: dict[int, str] = {}

    def _viz2(vi):
        mp = os.path.join(merged_dir, f"view_{vi:03d}_mask.npy")
        if not os.path.isfile(mp):
            return vi, None
        mask = np.load(mp)
        rgb_p = cameras[vi].get("rgb_path",
                                os.path.join(views_dir, f"view_{vi:03d}.png"))
        num_img = _overlay(mask, {}, rgb_p, use_labels=False)
        p = os.path.join(viz_dir, f"view_{vi:03d}_merged_numbered.png")
        num_img.save(p)
        return vi, p

    with ThreadPoolExecutor(max_workers=n_views) as pool:
        for f in as_completed(
                [pool.submit(_viz2, vi) for vi in range(n_views)]):
            vi, p = f.result()
            if p:
                merged_seg_paths[vi] = p

    # ── Stage 2: Label (per-view, all views in parallel) ─────────────
    # IDs are per-view, so each view gets its own labels.
    def _label_one(vi):
        if vi not in merged_seg_paths:
            return vi, None
        rgb_p = cameras[vi].get("rgb_path",
                                os.path.join(views_dir, f"view_{vi:03d}.png"))
        content = []
        img = _encode(merged_seg_paths[vi], "high")
        if img:
            content.append(img)
        img = _encode(rgb_p, "high")
        if img:
            content.append(img)
        if os.path.isfile(panorama_path):
            img = _encode(panorama_path, "low")
            if img:
                content.append(img)

        mask = np.load(os.path.join(merged_dir, f"view_{vi:03d}_mask.npy"))
        view_ids = sorted(set(int(x) for x in np.unique(mask) if x > 0))
        if not view_ids:
            return vi, None

        content.append({"type": "text", "text": (
            "Image 1: instance segmentation with numbered IDs. "
            "Image 2: RGB of same view. "
            "Image 3: panorama of the full scene.\n\n"
            f"Instance IDs in this view: {view_ids}\n\n"
            "For each ID, provide its object CATEGORY label.\n"
            "Rules:\n"
            "- Use simple category names (e.g. \"cabinet door\", "
            "\"microwave\", \"bowl\") — NO positional qualifiers "
            "like left/right/upper/lower/front/back\n"
            "- If a segment is a sub-component (handle, knob, "
            "button), label it as the parent object category\n"
            "- If the segment is room structure (wall, floor, "
            "ceiling, countertop, baseboard, any fixed surface) "
            'or too thin/narrow to be useful, label it "STRUCTURE"\n'
            "- Only give category names to clearly recognizable, "
            "substantive objects\n\n"
            '{"1": "STRUCTURE", "3": "microwave", "5": "bowl"}\n'
            "Include all IDs."
        )})
        try:
            return vi, _json_parse(_vlm(content))
        except Exception:
            return vi, None

    print(f"  [merge2d] Stage 2: {n_views} views in parallel (per-view)...")
    per_view_labels: dict[int, dict] = {}

    with ThreadPoolExecutor(max_workers=12) as pool:
        futs = {pool.submit(_label_one, vi): vi for vi in range(n_views)}
        for f in as_completed(futs):
            vi, data = f.result()
            if data:
                per_view_labels[vi] = data

    total_labeled = 0
    n_struct_removed = 0
    n_strip_removed = 0
    all_labels_union: dict[str, str] = {}
    for vi in range(n_views):
        lp = os.path.join(merged_dir, f"view_{vi:03d}_labels.json")
        mp = os.path.join(merged_dir, f"view_{vi:03d}_mask.npy")
        lbl = _json.load(open(lp)) if os.path.isfile(lp) else {}
        vlm_lbl = per_view_labels.get(vi, {})
        for mid_s, cat in vlm_lbl.items():
            lbl[str(int(mid_s))] = cat.lower().strip()
        mask = np.load(mp) if os.path.isfile(mp) else None
        to_remove = []
        for k, v in list(lbl.items()):
            if v == "structure":
                to_remove.append((int(k), "STRUCTURE"))
                continue
            if mask is not None:
                mid = int(k)
                ys, xs = np.where(mask == mid)
                if len(xs) > 0:
                    w = int(xs.max()) - int(xs.min()) + 1
                    h = int(ys.max()) - int(ys.min()) + 1
                    aspect = max(w, h) / max(min(w, h), 1)
                    fill = len(xs) / max(w * h, 1)
                    if aspect > 6 and fill < 0.3:
                        to_remove.append((mid, "strip"))
        for mid, reason in to_remove:
            lbl.pop(str(mid), None)
            if mask is not None:
                mask[mask == mid] = 0
            if reason == "STRUCTURE":
                n_struct_removed += 1
            else:
                n_strip_removed += 1
        if to_remove and mask is not None:
            np.save(mp, mask)
        _json.dump(lbl, open(lp, "w"), indent=2)
        total_labeled += len(lbl)
        for k, v in lbl.items():
            all_labels_union[f"v{vi:03d}_{k}"] = v

    _json.dump(all_labels_union,
               open(os.path.join(seg_dir, "merged_labels.json"), "w"), indent=2)
    n_total_removed = n_struct_removed + n_strip_removed
    if n_total_removed:
        parts = []
        if n_struct_removed:
            parts.append(f"{n_struct_removed} STRUCTURE")
        if n_strip_removed:
            parts.append(f"{n_strip_removed} thin strips")
        print(f"  [merge2d] Post-label filter: removed {' + '.join(parts)}")
    print(f"  [merge2d] Stage 2: {total_labeled} labels across {n_views} views")

    # ── Save Stage 3 viz: post-merge with labels (parallel) ───────────
    print("  [merge2d] Saving Stage 3 viz (merged + labeled)...")

    def _viz3(vi):
        mp = os.path.join(merged_dir, f"view_{vi:03d}_mask.npy")
        if not os.path.isfile(mp):
            return
        mask = np.load(mp)
        rgb_p = cameras[vi].get("rgb_path",
                                os.path.join(views_dir, f"view_{vi:03d}.png"))
        lbl = _json.load(open(os.path.join(merged_dir,
                                            f"view_{vi:03d}_labels.json")))
        labeled_img = _overlay(mask, lbl, rgb_p, use_labels=True)
        labeled_img.save(os.path.join(viz_dir,
                                       f"view_{vi:03d}_merged_labeled.png"))

    with ThreadPoolExecutor(max_workers=n_views) as pool:
        list(pool.map(_viz3, range(n_views)))


def _render_numbered_seg_vis(
    instance_info: dict,
    gaussian_labels,
    ply_path: str,
    seg_dir: str,
    view_indices: list[int],
) -> list[tuple[str, str]]:
    """Render seg_vis overlays with instance ID numbers instead of category labels.

    Returns list of (rgb_path, seg_vis_path) pairs.
    """
    import os
    import json as _json
    import torch
    import numpy as np
    from PIL import Image

    from src.scenes.marble.label_gaussians import (
        load_ply_gaussians, render_labeled_view,
        get_distinct_colors,
    )

    views_dir = os.path.join(seg_dir, "views")
    cameras = _json.load(open(os.path.join(views_dir, "cameras.json")))
    gaussians = load_ply_gaussians(ply_path)

    n_inst = len(instance_info)
    colors_list = get_distinct_colors(max(n_inst, 1))
    instance_colors = {
        iid: colors_list[i]
        for i, iid in enumerate(sorted(instance_info.keys()))
    }

    out_dir = os.path.join(seg_dir, "viz_numbered")
    os.makedirs(out_dir, exist_ok=True)

    pairs = []
    for vi in view_indices:
        if vi >= len(cameras):
            continue
        cam = cameras[vi]
        rgb_path = cam.get("rgb_path", os.path.join(views_dir, f"view_{vi:03d}.png"))
        seg_img_np = render_labeled_view(gaussians, gaussian_labels, instance_colors, cam)

        # Draw instance ID numbers (not category labels)
        from PIL import ImageDraw, ImageFont
        img = Image.fromarray((seg_img_np * 255).astype(np.uint8))
        draw = ImageDraw.Draw(img)
        try:
            font = ImageFont.truetype(
                "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 18)
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
            too_close = any(abs(u - du) < 80 and abs(v - dv) < 22 for du, dv in drawn)
            if too_close:
                continue
            label_text = str(iid)
            color = instance_colors.get(iid, (1, 1, 1))
            rgb_c = tuple(int(c * 255) for c in color)
            bbox = draw.textbbox((u, v), label_text, font=font)
            tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
            tx = max(2, min(u - tw // 2, W - tw - 4))
            ty = max(2, min(v - th // 2, H - th - 4))
            draw.rectangle([tx - 2, ty - 2, tx + tw + 2, ty + th + 2],
                           fill=(0, 0, 0, 200))
            draw.text((tx, ty), label_text, fill=rgb_c, font=font)
            drawn.append((u, v))

        seg_vis_path = os.path.join(out_dir, f"view_{vi:03d}_numbered.png")
        img.save(seg_vis_path)
        pairs.append((rgb_path, seg_vis_path))

    del gaussians
    torch.cuda.empty_cache()
    return pairs


def vlm_merge_and_label(
    instance_info: dict,
    gaussian_labels,
    means_np,
    seg_dir: str,
    ply_path: str,
    *,
    model: str = os.environ.get("VLM_MODEL", "Qwen3-VL-235B-A22B-Thinking"),
):
    """2-stage VLM: Stage 1 merges numbered instances, Stage 2 labels them.

    Modifies gaussian_labels and instance_info in-place.
    Returns (gaussian_labels, instance_info).
    """
    import base64
    import json as _json
    import os

    import numpy as np
    import requests as _req

    api_key = os.environ.get("VLM_API_KEY", "")
    if not api_key:
        print("  [merge+label] VLM_API_KEY not set, skipping")
        return gaussian_labels, instance_info

    view_indices = [0, 2, 4, 6, 8, 10]

    def _build_content(pairs, prompt_text):
        content = []
        for rgb_p, seg_p in pairs:
            for p in [rgb_p, seg_p]:
                if not os.path.isfile(p):
                    continue
                with open(p, "rb") as f:
                    b64 = base64.b64encode(f.read()).decode()
                ext = "png" if p.endswith(".png") else "jpeg"
                content.append({
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/{ext};base64,{b64}",
                        "detail": "high",
                    },
                })
        content.append({"type": "text", "text": prompt_text})
        return content

    def _vlm_call(content, max_tokens=1024):
        resp = _req.post(
            os.environ.get("VLM_API_BASE", "https://example.com/v1/chat/completions"),
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            },
            json={
                "model": model,
                "messages": [{"role": "user", "content": content}],
                "temperature": 0.0,
                "max_completion_tokens": max_tokens,
            },
            timeout=120,
        )
        resp.raise_for_status()
        return resp.json()["choices"][0]["message"]["content"].strip()

    def _parse_json(text):
        start = text.find("{")
        end = text.rfind("}") + 1
        if start < 0 or end <= start:
            return None
        try:
            return _json.loads(text[start:end])
        except _json.JSONDecodeError:
            return None

    # ── Stage 1: Merge ────────────────────────────────────────────────────
    print("  [merge+label] Stage 1: rendering numbered seg_vis...")
    pairs = _render_numbered_seg_vis(
        instance_info, gaussian_labels, ply_path, seg_dir, view_indices)

    iid_list = sorted(instance_info.keys())
    merge_prompt = (
        "These are 6 views of an indoor scene. For each pair, "
        "image 1 is the RGB render and image 2 is the instance "
        "segmentation with numbered IDs.\n\n"
        f"Instance IDs present: {iid_list}\n\n"
        "Identify instances that are the SAME physical object seen "
        "from different views, or adjacent segments that are parts "
        "of the same object.\n\n"
        'Reply as JSON: {"merge": [[1, 5], [3, 7, 12]], "remove": [99]}\n'
        '- "merge": groups of instance IDs to merge into one\n'
        '- "remove": instance IDs that should be excluded:\n'
        '  * noise or rendering artifacts (floating blobs)\n'
        '  * partial surface fragments (e.g. a thin strip of countertop edge, '
        'a small backsplash piece, a baseboard sliver)\n'
        '  * structural elements not useful as objects (bare wall patches, '
        'ceiling fragments, floor edges)\n'
        '  * any segment too small or ambiguous to be a recognizable object\n'
        'Use {"merge": [], "remove": []} if no changes needed.'
    )

    print("  [merge+label] Stage 1: VLM merge call...")
    try:
        content = _build_content(pairs, merge_prompt)
        text = _vlm_call(content)
        merge_data = _parse_json(text)
    except Exception as e:
        print(f"  [merge+label] Stage 1 failed: {e}")
        merge_data = None

    if merge_data:
        merge_groups = merge_data.get("merge", [])
        remove_ids = merge_data.get("remove", [])

        # Apply merges
        merged_count = 0
        for group in merge_groups:
            group = [int(g) for g in group if int(g) in instance_info]
            if len(group) < 2:
                continue
            sizes = [(iid, instance_info[iid]["n_gaussians"]) for iid in group]
            sizes.sort(key=lambda x: -x[1])
            keep_id = sizes[0][0]
            for iid, _ in sizes[1:]:
                gaussian_labels[gaussian_labels == iid] = keep_id
                instance_info.pop(iid, None)
                merged_count += 1
            mask = gaussian_labels == keep_id
            pts = means_np[mask]
            if len(pts) > 0:
                instance_info[keep_id]["center"] = pts.mean(axis=0).tolist()
                instance_info[keep_id]["bbox_min"] = pts.min(axis=0).tolist()
                instance_info[keep_id]["bbox_max"] = pts.max(axis=0).tolist()
                instance_info[keep_id]["n_gaussians"] = int(mask.sum())

        # Apply removes
        for iid in remove_ids:
            iid = int(iid)
            if iid in instance_info:
                gaussian_labels[gaussian_labels == iid] = -1
                instance_info.pop(iid)

        print(f"  [merge+label] Stage 1: merged {merged_count}, "
              f"removed {len(remove_ids)}, {len(instance_info)} remaining")
    else:
        print("  [merge+label] Stage 1: no merges needed")

    # ── Stage 2: Label ────────────────────────────────────────────────────
    print("  [merge+label] Stage 2: re-rendering merged seg_vis...")
    pairs = _render_numbered_seg_vis(
        instance_info, gaussian_labels, ply_path, seg_dir, view_indices)

    iid_list = sorted(instance_info.keys())
    label_prompt = (
        "These are 6 views of an indoor scene after instance merging. "
        "For each pair, image 1 is RGB and image 2 is the segmentation "
        "with numbered instance IDs.\n\n"
        f"Instance IDs present: {iid_list}\n\n"
        "For each instance ID, provide its object category label.\n"
        'Reply as JSON: {"1": "refrigerator", "2": "door", ...}\n'
        "Use simple, common object names (e.g. wall, floor, ceiling, "
        "door, table, chair, sofa, lamp, countertop, cabinet, sink, "
        "window, rug, shelf, refrigerator, oven, etc.).\n"
        "Include ALL instance IDs."
    )

    print("  [merge+label] Stage 2: VLM label call...")
    try:
        content = _build_content(pairs, label_prompt)
        text = _vlm_call(content, max_tokens=2048)
        label_data = _parse_json(text)
    except Exception as e:
        print(f"  [merge+label] Stage 2 failed: {e}")
        label_data = None

    if label_data:
        labeled = 0
        for iid_str, cat in label_data.items():
            iid = int(iid_str)
            if iid in instance_info:
                instance_info[iid]["category"] = cat.lower().strip()
                labeled += 1

        # Re-index labels: {category}_0, {category}_1, ...
        from collections import Counter
        cat_counter: Counter = Counter()
        for iid in sorted(instance_info.keys()):
            info = instance_info[iid]
            cat = info["category"]
            idx = cat_counter[cat]
            info["label"] = f"{cat}_{idx}"
            cat_counter[cat] += 1

        print(f"  [merge+label] Stage 2: labeled {labeled}/{len(instance_info)} instances")
    else:
        print("  [merge+label] Stage 2: labeling failed, keeping original labels")

    # Schedule seg_vis re-render as background thread (runs during step 5)
    import threading
    def _render_final_viz():
        try:
            import torch
            from src.scenes.marble.label_gaussians import (
                load_ply_gaussians, render_labeled_view, add_labels_to_image,
                get_distinct_colors,
            )
            import json as _json2
            from PIL import Image

            views_dir = os.path.join(seg_dir, "views")
            viz_dir = os.path.join(seg_dir, "viz_3d")
            os.makedirs(viz_dir, exist_ok=True)
            cameras = _json2.load(open(os.path.join(views_dir, "cameras.json")))
            gaussians = load_ply_gaussians(ply_path)

            n_inst = len(instance_info)
            colors_list = get_distinct_colors(max(n_inst, 1))
            inst_colors = {
                iid: colors_list[i]
                for i, iid in enumerate(sorted(instance_info.keys()))
            }

            seg_images = []
            for vi, cam in enumerate(cameras):
                img_np = render_labeled_view(gaussians, gaussian_labels, inst_colors, cam)
                labeled_img = add_labels_to_image(
                    img_np, gaussian_labels, instance_info, inst_colors, cam)
                out_path = os.path.join(viz_dir, f"view_{vi:03d}_seg3d.png")
                labeled_img.save(out_path)
                seg_images.append(labeled_img)

            # 3x4 grid
            if seg_images:
                ncols, nrows = 4, 3
                tw = seg_images[0].width // 3
                th = seg_images[0].height // 3
                grid = Image.new("RGB", (ncols * tw, nrows * th), (30, 30, 30))
                for i, img in enumerate(seg_images[:ncols * nrows]):
                    x, y = (i % ncols) * tw, (i // ncols) * th
                    grid.paste(img.resize((tw, th)), (x, y))
                grid.save(os.path.join(viz_dir, "grid_3d.png"))

            # Legend
            from PIL import ImageDraw as _IDraw, ImageFont as _IFont
            try:
                _lfont = _IFont.truetype(
                    "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 16)
            except OSError:
                _lfont = _IFont.load_default()
            sorted_iids = sorted(instance_info.keys())
            lh = 24
            legend = Image.new("RGB", (300, lh * len(sorted_iids) + 10), (255, 255, 255))
            ld = _IDraw.Draw(legend)
            for li, iid in enumerate(sorted_iids):
                color = inst_colors.get(iid, (0.5, 0.5, 0.5))
                rgb_c = tuple(int(c * 255) for c in color)
                y = 5 + li * lh
                ld.rectangle([5, y, 25, y + 18], fill=rgb_c)
                ld.text((30, y), instance_info[iid]["label"], fill=(0, 0, 0), font=_lfont)
            legend.save(os.path.join(viz_dir, "legend_3d.png"))

            del gaussians
            torch.cuda.empty_cache()
            print(f"  [merge+label] Updated {len(seg_images)} seg_vis + grid + legend")
        except Exception as e:
            print(f"  [merge+label] Viz update failed: {e}")

    _viz_thread = threading.Thread(target=_render_final_viz, daemon=True)
    _viz_thread.start()
    print("  [merge+label] Viz re-render started in background")

    return gaussian_labels, instance_info


def relabel_instances_vlm(
    instance_info: dict,
    seg_dir: str,
    *,
    model: str = os.environ.get("VLM_MODEL", "Qwen3-VL-235B-A22B-Thinking"),
) -> dict:
    """Send multi-view seg_vis + RGB pairs to the configured VLM.

    Updates instance_info in-place and returns it. Re-indexes labels when
    categories change so that duplicates get {category}_0, {category}_1, etc.
    """
    import base64
    import json as _json
    import os

    import requests as _req

    api_key = os.environ.get("VLM_API_KEY", "")
    if not api_key:
        print("  [relabel] VLM_API_KEY not set, skipping")
        return instance_info

    viz_dir = os.path.join(seg_dir, "viz_3d")
    views_dir = os.path.join(seg_dir, "views")

    view_indices = [0, 2, 4, 6, 8, 10]
    content: list[dict] = []
    for vi in view_indices:
        rgb_path = os.path.join(views_dir, f"view_{vi:03d}.png")
        seg_path = os.path.join(viz_dir, f"view_{vi:03d}_seg3d.png")
        for p in [rgb_path, seg_path]:
            if not os.path.isfile(p):
                continue
            with open(p, "rb") as f:
                b64 = base64.b64encode(f.read()).decode()
            content.append({
                "type": "image_url",
                "image_url": {"url": f"data:image/png;base64,{b64}", "detail": "high"},
            })

    if len(content) < 4:
        print("  [relabel] not enough view images, skipping")
        return instance_info

    current_labels = sorted(set(
        info["label"] for info in instance_info.values()
    ))
    prompt = (
        "These are 6 views of an indoor scene. For each view, the first image "
        "is the RGB render and the second is the segmentation overlay with "
        "instance labels.\n\n"
        f"Current instance labels: {current_labels}\n\n"
        "For each instance, verify if the label matches what the image shows. "
        "If wrong, provide the correct object category.\n"
        'Reply as JSON: {"old_label": "new_category", ...}\n'
        "Only include instances that need relabeling. "
        "Use empty {} if all labels are correct.\n"
        "Use simple, common object names (e.g. wall, refrigerator, floor, "
        "ceiling, rug, table, chair, lamp, etc.)."
    )
    content.append({"type": "text", "text": prompt})

    print(f"  [relabel] sending {len(content)-1} images + prompt to {model}...")
    try:
        resp = _req.post(
            os.environ.get("VLM_API_BASE", "https://example.com/v1/chat/completions"),
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            },
            json={
                "model": model,
                "messages": [{"role": "user", "content": content}],
                "temperature": 0.0,
                "max_completion_tokens": 1024,
            },
            timeout=120,
        )
        resp.raise_for_status()
        text = resp.json()["choices"][0]["message"]["content"].strip()
    except Exception as e:
        print(f"  [relabel] VLM call failed: {e}")
        return instance_info

    start = text.find("{")
    end = text.rfind("}") + 1
    if start < 0 or end <= start:
        print("  [relabel] no JSON in response, skipping")
        return instance_info

    try:
        remap = _json.loads(text[start:end])
    except _json.JSONDecodeError:
        print("  [relabel] invalid JSON, skipping")
        return instance_info

    if not remap:
        print("  [relabel] all labels correct")
        return instance_info

    # Apply remapping
    changed = 0
    for iid, info in instance_info.items():
        old_label = info["label"]
        if old_label in remap:
            new_cat = remap[old_label].lower().strip()
            info["category"] = new_cat
            changed += 1

    # Re-index: assign {category}_0, {category}_1, etc.
    from collections import Counter
    cat_counter: Counter = Counter()
    for iid in sorted(instance_info.keys(), key=int):
        info = instance_info[iid]
        cat = info["category"]
        idx = cat_counter[cat]
        info["label"] = f"{cat}_{idx}"
        cat_counter[cat] += 1

    print(f"  [relabel] changed {changed} labels: "
          f"{{{', '.join(f'{k} -> {v}' for k, v in remap.items())}}}")
    return instance_info


def verify_object_labels(
    objects: list[dict],
    images_dir: str,
    instance_id_map: dict[str, int],
    *,
    model: str = os.environ.get("VLM_MODEL", "Qwen3-VL-235B-A22B-Thinking"),
    max_workers: int = 20,
    min_px: int = 500,
    pad: int = 20,
) -> set[str]:
    """Verify each object's label by sending its cropped bbox to a VLM.

    For each object, finds the camera where it has the largest seg mask area,
    crops the RGB image to the object's bbox, and asks VLM vision whether the
    label matches. Returns set of object IDs that failed verification.
    """
    import base64
    import io
    import os
    from concurrent.futures import ThreadPoolExecutor, as_completed

    import requests as _req
    from PIL import Image as PILImage

    api_key = os.environ.get("VLM_API_KEY", "")
    if not api_key:
        print("  [label-verify] VLM_API_KEY not set, skipping")
        return set()

    seg_files = sorted(f for f in os.listdir(images_dir) if f.endswith("_seg.npy"))
    if not seg_files:
        return set()

    # Find best camera (largest mask area) for each object
    print(f"  [label-verify] scanning {len(seg_files)} masks for {len(objects)} objects...")
    best_cam: dict[str, tuple[str, int]] = {}
    for seg_file in seg_files:
        cam_name = seg_file.replace("_seg.npy", "")
        mask = np.load(os.path.join(images_dir, seg_file))
        for obj in objects:
            oid = obj.get("id", "")
            iid = instance_id_map.get(oid, -1)
            if iid < 0:
                continue
            px = int((mask == iid).sum())
            if px >= min_px and px > best_cam.get(oid, ("", 0))[1]:
                best_cam[oid] = (cam_name, px)

    # Build verification tasks
    tasks: list[tuple[str, str, str]] = []
    for obj in objects:
        oid = obj.get("id", "")
        label = obj.get("type", obj.get("label", ""))
        if oid not in best_cam:
            continue
        cam_name, _ = best_cam[oid]
        tasks.append((oid, label, cam_name))

    if not tasks:
        print("  [label-verify] no objects to verify")
        return set()

    def _verify_one(task):
        oid, label, cam_name = task
        iid = instance_id_map.get(oid, -1)
        try:
            mask = np.load(os.path.join(images_dir, f"{cam_name}_seg.npy"))
            ys, xs = np.where(mask == iid)
            if len(xs) == 0:
                return oid, True
            H, W = mask.shape
            x1 = max(0, int(xs.min()) - pad)
            y1 = max(0, int(ys.min()) - pad)
            x2 = min(W, int(xs.max()) + pad)
            y2 = min(H, int(ys.max()) + pad)
            img = PILImage.open(os.path.join(images_dir, f"{cam_name}.jpg")).convert("RGB")
            crop = img.crop((x1, y1, x2, y2))
            buf = io.BytesIO()
            crop.save(buf, format="JPEG", quality=85)
            b64 = base64.b64encode(buf.getvalue()).decode()

            resp = _req.post(
                os.environ.get("VLM_API_BASE", "https://example.com/v1/chat/completions"),
                headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
                json={
                    "model": model,
                    "messages": [{
                        "role": "user",
                        "content": [
                            {"type": "text", "text": f"This region is labeled '{label}'. Does the image show a {label}? Answer only YES or NO."},
                            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}", "detail": "low"}},
                        ],
                    }],
                    "temperature": 0.0,
                    "max_completion_tokens": 8,
                },
                timeout=30,
            )
            resp.raise_for_status()
            answer = resp.json()["choices"][0]["message"]["content"].strip().upper()
            return oid, "YES" in answer
        except Exception as e:
            return oid, True

    print(f"  [label-verify] verifying {len(tasks)} objects with {model} ({max_workers} parallel)...")
    excluded: set[str] = set()
    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = {pool.submit(_verify_one, t): t for t in tasks}
        for future in as_completed(futures):
            oid, passed = future.result()
            if not passed:
                excluded.add(oid)

    if excluded:
        print(f"  [label-verify] excluded {len(excluded)}: {sorted(excluded)}")
    else:
        print("  [label-verify] all labels verified")
    return excluded
