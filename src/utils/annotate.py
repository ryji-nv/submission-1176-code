"""
Annotate rendered images with projected world-space markers.

Uses world coordinates + camera pose to project to pixels, then draws
filled dots (circles) or wireframe 3D bounding boxes with PIL.
"""

from __future__ import annotations

import json
import os
import random as _random
import shutil
from pathlib import Path

import numpy as np

from src.utils.projection import project_world_to_pixel

# Circle colors (R, G, B)
BLUE = (30, 100, 255)
RED = (255, 60, 60)
YELLOW = (255, 210, 0)
GREEN = (50, 200, 80)


def draw_circles_on_image(
    image_path: str,
    blue_xyz: tuple[float, float, float],
    red_xyz: tuple[float, float, float],
    camera_position: list[float] | tuple[float, float, float],
    camera_look_at: list[float] | tuple[float, float, float],
    output_path: str | None = None,
    *,
    dot_radius_frac: float = 0.018,
    flip_y: bool = False,
    horizontal_fov_deg: float = 82.0,
    override_blue_pixel: tuple[int, int] | None = None,
    override_red_pixel: tuple[int, int] | None = None,
) -> str:
    """
    Load image, project blue and red world positions to pixels, draw solid dots, save.
    Returns the path where the image was saved (output_path or image_path).
    """
    try:
        from PIL import Image, ImageDraw
    except ImportError:
        raise ImportError(
            "PIL is required for draw_circles_on_image. Install with: pip install Pillow"
        )

    img_path = Path(image_path)
    if not img_path.is_file():
        raise FileNotFoundError(f"Image not found: {image_path}")
    out_path = output_path or str(img_path)
    img = Image.open(img_path).convert("RGB")
    w, h = img.size

    cam_pos = tuple(camera_position)
    look_at = tuple(camera_look_at)

    px_blue = override_blue_pixel or project_world_to_pixel(
        blue_xyz,
        cam_pos,
        look_at,
        w,
        h,
        flip_y=flip_y,
        horizontal_fov_deg=horizontal_fov_deg,
    )
    px_red = override_red_pixel or project_world_to_pixel(
        red_xyz,
        cam_pos,
        look_at,
        w,
        h,
        flip_y=flip_y,
        horizontal_fov_deg=horizontal_fov_deg,
    )

    radius = max(6, int(min(w, h) * dot_radius_frac))
    draw = ImageDraw.Draw(img)

    def draw_dot(center: tuple[int, int], color: tuple[int, int, int]) -> None:
        x, y = center
        draw.ellipse(
            (x - radius, y - radius, x + radius, y + radius),
            fill=color,
        )

    if px_blue is None and px_red is None:
        raise ValueError(
            "Both object positions projected behind the camera or invalid. "
            "Check that camera_position and camera_look_at match the render; try --flip-y."
        )
    if px_blue is not None:
        draw_dot(px_blue, BLUE)
    if px_red is not None:
        draw_dot(px_red, RED)

    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    img.save(out_path, quality=95)
    return out_path


def draw_single_circle(
    image_path: str,
    xyz: tuple[float, float, float],
    color: tuple[int, int, int],
    camera_position: list[float] | tuple[float, float, float],
    camera_look_at: list[float] | tuple[float, float, float],
    output_path: str | None = None,
    *,
    dot_radius_frac: float = 0.018,
    flip_y: bool = False,
    horizontal_fov_deg: float = 82.0,
    override_pixel: tuple[int, int] | None = None,
) -> str:
    """
    Load image, project one world position to a pixel, draw a solid dot of the given color, save.

    Returns the path where the image was saved.
    """
    try:
        from PIL import Image, ImageDraw
    except ImportError:
        raise ImportError(
            "PIL is required for draw_single_circle. Install with: pip install Pillow"
        )

    img_path = Path(image_path)
    if not img_path.is_file():
        raise FileNotFoundError(f"Image not found: {image_path}")
    out_path = output_path or str(img_path)
    img = Image.open(img_path).convert("RGB")
    w, h = img.size

    if override_pixel is not None:
        px = override_pixel
    else:
        px = project_world_to_pixel(
            xyz,
            tuple(camera_position),
            tuple(camera_look_at),
            w,
            h,
            flip_y=flip_y,
            horizontal_fov_deg=horizontal_fov_deg,
        )
    if px is None:
        raise ValueError(
            "Object position projected behind the camera or invalid. "
            "Check that camera_position and camera_look_at match the render; try --flip-y."
        )

    radius = max(6, int(min(w, h) * dot_radius_frac))
    draw = ImageDraw.Draw(img)
    x, y = px
    draw.ellipse(
        (x - radius, y - radius, x + radius, y + radius),
        fill=color,
    )

    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    img.save(out_path, quality=95)
    return out_path


def _project_obj_bbox_2d(
    obj_centroid: tuple,
    obj_dims: tuple,
    cam_pos: tuple,
    look_at: tuple,
    img_w: int,
    img_h: int,
    horizontal_fov_deg: float = 82.0,
) -> tuple[int, int, int, int] | None:
    """Project object's 3D extent to 2D pixel bbox (x1,y1,x2,y2)."""
    cx, cy, cz = obj_centroid
    hw, hl, hh = obj_dims[0] / 2, obj_dims[1] / 2, obj_dims[2] / 2
    xs, ys = [], []
    for sx in (-1, 1):
        for sy in (-1, 1):
            for sz in (-1, 1):
                pt = (cx + sx * hw, cy + sy * hl, cz + sz * hh)
                px = project_world_to_pixel(
                    pt,
                    cam_pos,
                    look_at,
                    img_w,
                    img_h,
                    horizontal_fov_deg=horizontal_fov_deg,
                )
                if px is None:
                    return None
                xs.append(px[0])
                ys.append(px[1])
    return (min(xs), min(ys), max(xs), max(ys))


def draw_single_bbox(
    image_path: str,
    centroid: tuple[float, float, float],
    dims: tuple[float, float, float],
    color: tuple[int, int, int],
    camera_position: tuple,
    camera_look_at: tuple,
    output_path: str | None = None,
    *,
    line_width: int = 6,
    horizontal_fov_deg: float = 82.0,
    obj_id: str | None = None,
    cam_name: str | None = None,
    seg_mask_dir: str | None = None,
    instance_id_map: dict | None = None,
) -> str:
    """Draw a 2D bounding box around a projected object."""
    from PIL import Image, ImageDraw

    img = Image.open(image_path).convert("RGB")
    w, h = img.size
    bbox = _bbox_from_seg_mask(obj_id, cam_name, seg_mask_dir, instance_id_map)
    if bbox is None:
        bbox = _project_obj_bbox_2d(
            centroid,
            dims,
            tuple(camera_position),
            tuple(camera_look_at),
            w,
            h,
            horizontal_fov_deg=horizontal_fov_deg,
        )
    if bbox is not None:
        draw = ImageDraw.Draw(img)
        draw.rectangle(bbox, outline=color, width=line_width)

    out = output_path or image_path
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    img.save(out, quality=95)
    return out


def draw_multiple_bboxes(
    image_path: str,
    objects: list[tuple[tuple, tuple, tuple[int, int, int]]],
    camera_position: tuple,
    camera_look_at: tuple,
    output_path: str | None = None,
    *,
    line_width: int = 6,
    horizontal_fov_deg: float = 82.0,
    obj_ids: list[str] | None = None,
    cam_name: str | None = None,
    seg_mask_dir: str | None = None,
    instance_id_map: dict | None = None,
) -> str:
    """Draw multiple 2D bounding boxes. objects: list of (centroid, dims, color)."""
    from PIL import Image, ImageDraw

    img = Image.open(image_path).convert("RGB")
    w, h = img.size
    draw = ImageDraw.Draw(img)
    cam_pos = tuple(camera_position)
    look_at = tuple(camera_look_at)

    for idx, (centroid, dims, color) in enumerate(objects):
        oid = obj_ids[idx] if obj_ids and idx < len(obj_ids) else None
        bbox = _bbox_from_seg_mask(oid, cam_name, seg_mask_dir, instance_id_map)
        if bbox is None and not seg_mask_dir:
            bbox = _project_obj_bbox_2d(
                centroid,
                dims,
                cam_pos,
                look_at,
                w,
                h,
                horizontal_fov_deg=horizontal_fov_deg,
            )
        if bbox is not None:
            draw.rectangle(bbox, outline=color, width=line_width)

    out = output_path or image_path
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    img.save(out, quality=95)
    return out


def draw_multiple_circles(
    image_path: str,
    objects: list[tuple[tuple[float, float, float], tuple[int, int, int]]],
    camera_position: list[float] | tuple[float, float, float],
    camera_look_at: list[float] | tuple[float, float, float],
    output_path: str | None = None,
    *,
    dot_radius_frac: float = 0.018,
    horizontal_fov_deg: float = 82.0,
) -> str:
    """
    Draw multiple colored dots on an image.
    objects: list of ((x, y, z), (R, G, B)) pairs.
    Returns the path where the image was saved.
    """
    try:
        from PIL import Image, ImageDraw
    except ImportError:
        raise ImportError("PIL is required. Install with: pip install Pillow")

    img_path = Path(image_path)
    if not img_path.is_file():
        raise FileNotFoundError(f"Image not found: {image_path}")
    out_path = output_path or str(img_path)
    img = Image.open(img_path).convert("RGB")
    w, h = img.size

    cam_pos = tuple(camera_position)
    look_at = tuple(camera_look_at)
    radius = max(6, int(min(w, h) * dot_radius_frac))
    draw = ImageDraw.Draw(img)

    for xyz, color in objects:
        px = project_world_to_pixel(
            xyz, cam_pos, look_at, w, h, horizontal_fov_deg=horizontal_fov_deg
        )
        if px is not None:
            x, y = px
            draw.ellipse((x - radius, y - radius, x + radius, y + radius), fill=color)

    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    img.save(out_path, quality=95)
    return out_path


def draw_bbox_3d(
    image_path: str,
    centroid: tuple[float, float, float],
    dims: tuple[float, float, float],
    camera_position,
    camera_look_at,
    output_path: str | None = None,
    *,
    color: tuple[int, int, int] = (0, 255, 0),
    line_width: int = 2,
    flip_y: bool = False,
    horizontal_fov_deg: float = 82.0,
) -> str:
    """
    Draw an axis-aligned 3D bounding box as 12 wireframe edges on an image.

    centroid: (cx, cy, cz) — center of the box in world coordinates
    dims: (width, length, height) — full extents along each axis
    Returns the path where the image was saved.
    """
    try:
        from PIL import Image, ImageDraw
    except ImportError:
        raise ImportError(
            "PIL is required for draw_bbox_3d. Install with: pip install Pillow"
        )

    img_path = Path(image_path)
    if not img_path.is_file():
        raise FileNotFoundError(f"Image not found: {image_path}")
    out_path = output_path or str(img_path)
    img = Image.open(img_path).convert("RGB")
    w, h = img.size

    cam_pos = tuple(camera_position)
    look_at = tuple(camera_look_at)

    cx, cy, cz = centroid
    hw, hl, hh = dims[0] / 2.0, dims[1] / 2.0, dims[2] / 2.0

    # 8 corners: signs in order (sx, sy, sz) for indices 0..7
    signs = [
        (-1, -1, -1),
        (1, -1, -1),
        (-1, 1, -1),
        (1, 1, -1),
        (-1, -1, 1),
        (1, -1, 1),
        (-1, 1, 1),
        (1, 1, 1),
    ]
    corners_3d = [(cx + sx * hw, cy + sy * hl, cz + sz * hh) for sx, sy, sz in signs]

    # 12 edges: pairs of corner indices
    edges = [
        (0, 1),
        (2, 3),
        (4, 5),
        (6, 7),  # along X
        (0, 2),
        (1, 3),
        (4, 6),
        (5, 7),  # along Y
        (0, 4),
        (1, 5),
        (2, 6),
        (3, 7),  # along Z
    ]

    # Project all 8 corners to pixels
    pixels = [
        project_world_to_pixel(
            pt,
            cam_pos,
            look_at,
            w,
            h,
            flip_y=flip_y,
            horizontal_fov_deg=horizontal_fov_deg,
        )
        for pt in corners_3d
    ]

    draw = ImageDraw.Draw(img)
    for i, j in edges:
        p1, p2 = pixels[i], pixels[j]
        if p1 is None or p2 is None:
            continue
        draw.line([p1, p2], fill=color, width=line_width)

    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    img.save(out_path, quality=95)
    return out_path


# ---------------------------------------------------------------------------
# Ground-truth mask generation
# ---------------------------------------------------------------------------


def generate_ground_truth(
    q: dict,
    out_dir: str,
    i: int,
    images_dir: str = "",
    img_w: int = 1920,
    img_h: int = 1080,
    fov: float = 82.0,
    n_points: int = 16,
    seg_mask_dir: str | None = None,
    instance_id_map: dict | None = None,
) -> None:
    """Generate mask image + multi-point ground truth for coordinate answers.
    Uses seg masks (Marble), instance_seg (SAGE), or convex hull fallback."""
    from PIL import Image as PILImage, ImageDraw

    cam_name = q.get("camera_name", "")
    cam_pos = tuple(q["camera_position"])
    look_at = tuple(q["camera_look_at"])

    obj = q.get("object") or q.get("surface")
    if not obj or "centroid" not in obj:
        return

    obj_id = obj.get("id", "")
    mask_arr = None

    # Path 1: Marble seg masks ({cam_name}_seg.npy + instance_id_map)
    if seg_mask_dir and instance_id_map:
        iid = instance_id_map.get(obj_id)
        seg_path = os.path.join(seg_mask_dir, f"{cam_name}_seg.npy")
        if iid is not None and os.path.isfile(seg_path):
            seg = np.load(seg_path)
            mask_arr = np.where(seg == iid, 255, 0).astype(np.uint8)
            img_h, img_w = mask_arr.shape
            if mask_arr.sum() == 0:
                mask_arr = None

    # Path 2: SAGE/SceneSmith instance_seg directory
    if mask_arr is None:
        seg_dir = (
            os.path.join(os.path.dirname(images_dir), "instance_seg")
            if images_dir
            else ""
        )
        seg_path = os.path.join(seg_dir, f"{cam_name}.npy") if seg_dir else ""
        info_path = os.path.join(seg_dir, f"{cam_name}_info.json") if seg_dir else ""
        if seg_path and os.path.isfile(seg_path) and os.path.isfile(info_path):
            try:
                seg = np.load(seg_path)
                with open(info_path) as f:
                    seg_info = json.load(f)
                id_to_obj = seg_info.get("idToObjectId", {})
                target_ids = [int(k) for k, v in id_to_obj.items() if v == obj_id]
                if target_ids:
                    mask_arr = np.zeros((seg.shape[0], seg.shape[1]), dtype=np.uint8)
                    for tid in target_ids:
                        mask_arr[seg == tid] = 255
                    img_h, img_w = mask_arr.shape
            except Exception:
                pass

    # Path 3: Convex hull projection fallback
    if mask_arr is None or mask_arr.sum() == 0:
        centroid = tuple(obj["centroid"])
        w, l, h = obj.get("width", 0.3), obj.get("length", 0.3), obj.get("height", 0.3)
        hw, hl, hh = w / 2, l / 2, h / 2
        corners_3d = [
            (centroid[0] + sx * hw, centroid[1] + sy * hl, centroid[2] + sz * hh)
            for sx in (-1, 1)
            for sy in (-1, 1)
            for sz in (-1, 1)
        ]
        pts_2d = []
        for pt in corners_3d:
            px = project_world_to_pixel(
                pt, cam_pos, look_at, img_w, img_h, horizontal_fov_deg=fov
            )
            if px is not None:
                pts_2d.append(px)
        if len(pts_2d) < 3:
            return
        mask_img = PILImage.new("L", (img_w, img_h), 0)
        draw = ImageDraw.Draw(mask_img)
        pts_arr = np.array(pts_2d)
        try:
            from scipy.spatial import ConvexHull

            hull = ConvexHull(pts_arr)
            draw.polygon([tuple(pts_arr[v]) for v in hull.vertices], fill=255)
        except Exception:
            draw.rectangle(
                [
                    int(pts_arr[:, 0].min()),
                    int(pts_arr[:, 1].min()),
                    int(pts_arr[:, 0].max()),
                    int(pts_arr[:, 1].max()),
                ],
                fill=255,
            )
        mask_arr = np.array(mask_img)

    if q.get("type") in ("object_placement", "free_space") and mask_arr is not None:
        surf_top = obj.get("z", 0) + obj.get("height", 0)
        scene_path = os.path.join(os.path.dirname(images_dir), "scene.json") if images_dir else ""
        if scene_path and os.path.isfile(scene_path):
            try:
                with open(scene_path) as f:
                    scene_objs = {o["id"]: o for o in json.load(f).get("objects", [])}
                seg_mask = _load_seg_mask(
                    os.path.join(seg_mask_dir, f"{cam_name}_seg.npy")
                ) if seg_mask_dir else None
                if seg_mask is not None and instance_id_map:
                    for oid, iid in instance_id_map.items():
                        if oid == obj_id:
                            continue
                        so = scene_objs.get(oid, {})
                        so_z = so.get("z", -999)
                        if abs(so_z - surf_top) < 0.15:
                            mask_arr[seg_mask == iid] = 0
                elif seg_path and os.path.isfile(seg_path):
                    seg = np.load(seg_path)
                    info_path_sage = os.path.join(
                        os.path.dirname(images_dir), "instance_seg",
                        f"{cam_name}_info.json") if images_dir else ""
                    if info_path_sage and os.path.isfile(info_path_sage):
                        with open(info_path_sage) as f2:
                            id_to_obj = json.load(f2).get("idToObjectId", {})
                        for sid_str, oid in id_to_obj.items():
                            if oid == obj_id:
                                continue
                            so = scene_objs.get(oid, {})
                            if abs(so.get("z", -999) - surf_top) < 0.15:
                                mask_arr[seg == int(sid_str)] = 0
            except Exception:
                pass

    mask_name = f"question_mask_{i}.png"
    PILImage.fromarray(mask_arr).save(os.path.join(out_dir, mask_name))
    q["ground_truth_mask"] = mask_name

    white_ys, white_xs = np.where(mask_arr > 0)
    if len(white_xs) == 0:
        return
    rng = _random.Random(hash((cam_name, i)))
    indices = list(range(len(white_xs)))
    rng.shuffle(indices)
    points = []
    for idx in indices[:n_points]:
        points.append(
            (
                round(float(white_xs[idx]) / img_w, 3),
                round(float(white_ys[idx]) / img_h, 3),
            )
        )
    q["answer"] = str(points)


from functools import lru_cache


@lru_cache(maxsize=16)
def _load_seg_mask(seg_path: str) -> np.ndarray | None:
    if not os.path.isfile(seg_path):
        return None
    return np.load(seg_path)


def _bbox_from_seg_mask(
    obj_id: str | None,
    cam_name: str,
    seg_mask_dir: str | None,
    instance_id_map: dict | None = None,
    padding: int = 4,
) -> tuple[int, int, int, int] | None:
    """Derive 2D bbox from the seg mask for an object. Returns (x1,y1,x2,y2)."""
    if not seg_mask_dir or not obj_id or not instance_id_map:
        return None
    iid = instance_id_map.get(obj_id)
    if iid is None:
        return None
    mask = _load_seg_mask(os.path.join(seg_mask_dir, f"{cam_name}_seg.npy"))
    if mask is None:
        return None
    ys, xs = np.where(mask == iid)
    if len(xs) < 10:
        return None
    H, W = mask.shape
    x1, y1 = int(xs.min()) - padding, int(ys.min()) - padding
    x2, y2 = int(xs.max()) + padding, int(ys.max()) + padding
    return (max(x1, 0), max(y1, 0), min(x2, W - 1), min(y2, H - 1))


def _snap_to_mask(
    px: int,
    py: int,
    obj_id: str | None,
    cam_name: str,
    seg_mask_dir: str | None,
    instance_id_map: dict | None = None,
) -> tuple[int, int]:
    """Snap a projected point to the visual center of the object on the seg mask.

    Strategy: find the largest connected component of the object's mask,
    then place the dot at its centroid (guaranteed to be near the densest
    region).  If the centroid pixel isn't on the mask (non-convex shape),
    snap to the nearest mask pixel.

    If *seg_mask_dir* is ``None`` or the mask file doesn't exist, returns
    ``(px, py)`` unchanged.  *instance_id_map* maps ``obj_id`` -> int instance_id
    used in the mask.
    """
    if not seg_mask_dir or not obj_id:
        return (px, py)
    mask_path = os.path.join(seg_mask_dir, f"{cam_name}_seg.npy")
    if not instance_id_map or obj_id not in instance_id_map:
        return (px, py)

    iid = instance_id_map[obj_id]
    mask = _load_seg_mask(mask_path)
    if mask is None:
        return (px, py)
    H, W = mask.shape

    obj_mask = mask == iid
    ys, xs = np.where(obj_mask)
    if len(xs) == 0:
        return (px, py)

    from scipy import ndimage

    labeled, n_components = ndimage.label(obj_mask)
    if n_components > 1:
        sizes = ndimage.sum(obj_mask, labeled, range(1, n_components + 1))
        largest = int(np.argmax(sizes)) + 1
        ys, xs = np.where(labeled == largest)

    mx, my = int(np.mean(xs)), int(np.mean(ys))

    mx = np.clip(mx, 0, W - 1)
    my = np.clip(my, 0, H - 1)
    if not obj_mask[my, mx]:
        dists = (xs.astype(np.int32) - mx) ** 2 + (ys.astype(np.int32) - my) ** 2
        nearest = int(np.argmin(dists))
        mx, my = int(xs[nearest]), int(ys[nearest])

    return (mx, my)


# ---------------------------------------------------------------------------
# Per-question-type image annotation + logging
# ---------------------------------------------------------------------------


def write_question_image(
    q: dict,
    qtype: str,
    i: int,
    src_image: str,
    images_dir: str,
    out_dir: str,
    fov: float,
    seg_mask_dir: str | None = None,
    instance_id_map: dict | None = None,
) -> None:
    """Annotate and save question image(s), set image_path key(s) on q, print summary."""
    if qtype == "object_matching_mv":
        iname_a = f"question_image_{i}_a.jpg"
        iname_b = f"question_image_{i}_b.jpg"
        obj_info = q["object"]
        dims = (
            obj_info.get("width", 0.3),
            obj_info.get("length", 0.3),
            obj_info.get("height", 0.3),
        )
        draw_single_bbox(
            os.path.join(images_dir, f"{q['camera_a']['name']}.jpg"),
            tuple(obj_info["centroid"]),
            dims,
            RED,
            q["camera_a"]["position"],
            q["camera_a"]["look_at"],
            output_path=os.path.join(out_dir, iname_a),
            horizontal_fov_deg=fov,
            obj_id=obj_info.get("id"),
            cam_name=q["camera_a"]["name"],
            seg_mask_dir=seg_mask_dir,
            instance_id_map=instance_id_map,
        )
        shutil.copy2(
            os.path.join(images_dir, f"{q['camera_b']['name']}.jpg"),
            os.path.join(out_dir, iname_b),
        )
        q["image_path_a"] = iname_a
        q["image_path_b"] = iname_b
        print(f"    [{i}] answer={q['answer']}  obj={obj_info['label']}")
    elif qtype in (
        "camera_relative_position",
        "camera_facing_direction",
        "viewpoint_change",
    ):
        iname_a = f"question_image_{i}_a.jpg"
        iname_b = f"question_image_{i}_b.jpg"
        shutil.copy2(
            os.path.join(images_dir, f"{q['camera_a']['name']}.jpg"),
            os.path.join(out_dir, iname_a),
        )
        shutil.copy2(
            os.path.join(images_dir, f"{q['camera_b']['name']}.jpg"),
            os.path.join(out_dir, iname_b),
        )
        q["image_path_a"] = iname_a
        q["image_path_b"] = iname_b
        print(
            f"    [{i}] answer={q['answer']}  "
            f"a={q['camera_a']['name']}  b={q['camera_b']['name']}"
        )
    elif qtype == "camera_motion":
        iname_a = f"question_image_{i}_a.jpg"
        iname_b = f"question_image_{i}_b.jpg"
        src_b = os.path.join(images_dir, f"{q['camera_b']['name']}.jpg")
        if not os.path.isfile(src_b):
            raise FileNotFoundError(f"Stepped camera image not found: {src_b}")
        shutil.copy2(src_image, os.path.join(out_dir, iname_a))
        shutil.copy2(src_b, os.path.join(out_dir, iname_b))
        q["image_path_a"] = iname_a
        q["image_path_b"] = iname_b
        print(
            f"    [{i}] answer={q['answer']}  "
            f"a={q['camera_a']['name']}  b={q['camera_b']['name']}"
        )
    elif qtype in ("distance_estimation", "depth_estimation"):
        iname = f"question_image_{i}.jpg"
        obj_info = q["object"]
        _snap_px = None
        if seg_mask_dir:
            _snap_px = _snap_to_mask(
                0,
                0,
                obj_info.get("id"),
                q.get("camera_name", ""),
                seg_mask_dir,
                instance_id_map,
            )
        draw_single_circle(
            src_image,
            tuple(obj_info["centroid"]),
            RED,
            q["camera_position"],
            q["camera_look_at"],
            output_path=os.path.join(out_dir, iname),
            horizontal_fov_deg=fov,
            override_pixel=_snap_px,
        )
        q["image_path"] = iname
        print(f"    [{i}] answer={q['answer']} m  label={obj_info['label']}")
    elif qtype == "camera_object_position":
        iname = f"question_image_{i}.jpg"
        obj_info = q["object"]
        dims = (
            obj_info.get("width", 0.3),
            obj_info.get("length", 0.3),
            obj_info.get("height", 0.3),
        )
        draw_single_bbox(
            src_image,
            tuple(obj_info["centroid"]),
            dims,
            RED,
            q["camera_position"],
            q["camera_look_at"],
            output_path=os.path.join(out_dir, iname),
            horizontal_fov_deg=fov,
            obj_id=obj_info.get("id"),
            cam_name=q.get("camera_name"),
            seg_mask_dir=seg_mask_dir,
            instance_id_map=instance_id_map,
        )
        q["image_path"] = iname
        print(f"    [{i}] answer={q['answer']}  label={obj_info['label']}")
    elif qtype == "object_size":
        iname = f"question_image_{i}.jpg"
        shutil.copy2(src_image, os.path.join(out_dir, iname))
        q["image_path"] = iname
        print(f"    [{i}] answer={q['answer']} m  label={q['object']['label']}")
    elif qtype == "depth_ordering":
        iname = f"question_image_{i}.jpg"
        _color_map = {"blue": BLUE, "red": RED, "green": GREEN}
        objects_xyz_colors = [
            (tuple(o["centroid"]), _color_map[o["circle_color"]]) for o in q["objects"]
        ]
        draw_multiple_circles(
            src_image,
            objects_xyz_colors,
            q["camera_position"],
            q["camera_look_at"],
            output_path=os.path.join(out_dir, iname),
            horizontal_fov_deg=fov,
        )
        q["image_path"] = iname
        labels = ", ".join(f"{o['circle_color']}={o['label']}" for o in q["objects"])
        print(f"    [{i}] answer={q['answer']}  {labels}")
    elif qtype in ("room_size", "object_count"):
        iname = f"question_image_{i}.jpg"
        shutil.copy2(src_image, os.path.join(out_dir, iname))
        q["image_path"] = iname
        unit = " m\u00b2" if qtype == "room_size" else ""
        extra = f"  type={q['counted_type']}" if qtype == "object_count" else ""
        print(f"    [{i}] answer={q['answer']}{unit}{extra}")
    elif qtype in ("object_placement", "free_space", "spatial_compatibility"):
        iname = f"question_image_{i}.jpg"
        shutil.copy2(src_image, os.path.join(out_dir, iname))
        q["image_path"] = iname
        if qtype in ("object_placement", "free_space"):
            generate_ground_truth(
                q,
                out_dir,
                i,
                images_dir=images_dir,
                fov=fov,
                seg_mask_dir=seg_mask_dir,
                instance_id_map=instance_id_map,
            )
        if qtype == "spatial_compatibility":
            a, b, d = (
                q.get("object_a", {}),
                q.get("object_b", {}),
                q.get("direction", ""),
            )
            print(
                f"    [{i}] answer={q['answer']}  {a.get('label', '')} {d} {b.get('label', '')}"
            )
        else:
            mov, surf = q.get("moveable_object", {}), q.get("surface", {})
            print(
                f"    [{i}] answer={q['answer']}  mov={mov.get('label', '')}  surface={surf.get('label', '')}"
            )
    elif qtype in ("object_category", "object_size_qualitative"):
        iname = f"question_image_{i}.jpg"
        obj_info = q["object"]
        dims = (
            obj_info.get("width", 0.3),
            obj_info.get("length", 0.3),
            obj_info.get("height", 0.3),
        )
        draw_single_bbox(
            src_image,
            tuple(obj_info["centroid"]),
            dims,
            RED,
            q["camera_position"],
            q["camera_look_at"],
            output_path=os.path.join(out_dir, iname),
            horizontal_fov_deg=fov,
        )
        q["image_path"] = iname
        print(f"    [{i}] answer={q['answer']}  label={obj_info['label']}")
    elif qtype == "object_grounding_bbox":
        iname = f"question_image_{i}.jpg"
        shutil.copy2(src_image, os.path.join(out_dir, iname))
        q["image_path"] = iname
        bbox_answer = q["answer"]
        generate_ground_truth(
            q,
            out_dir,
            i,
            images_dir=images_dir,
            fov=fov,
            seg_mask_dir=seg_mask_dir,
            instance_id_map=instance_id_map,
        )
        q["answer"] = bbox_answer
        print(f"    [{i}] answer={q['answer']}  label={q['object']['label']}")
    elif qtype == "object_object_position_mv":
        iname_a = f"question_image_{i}_a.jpg"
        iname_b = f"question_image_{i}_b.jpg"
        blue_m, red_m = q["blue"], q["red"]
        bbox_objs = [
            (
                tuple(blue_m["centroid"]),
                (
                    blue_m.get("width", 0.3),
                    blue_m.get("length", 0.3),
                    blue_m.get("height", 0.3),
                ),
                BLUE,
            ),
            (
                tuple(red_m["centroid"]),
                (
                    red_m.get("width", 0.3),
                    red_m.get("length", 0.3),
                    red_m.get("height", 0.3),
                ),
                RED,
            ),
        ]
        _mv_ids = [blue_m.get("id"), red_m.get("id")]
        draw_multiple_bboxes(
            os.path.join(images_dir, f"{q['camera_a']['name']}.jpg"),
            bbox_objs,
            q["camera_a"]["position"],
            q["camera_a"]["look_at"],
            output_path=os.path.join(out_dir, iname_a),
            horizontal_fov_deg=fov,
            obj_ids=_mv_ids,
            cam_name=q["camera_a"]["name"],
            seg_mask_dir=seg_mask_dir,
            instance_id_map=instance_id_map,
        )
        draw_multiple_bboxes(
            os.path.join(images_dir, f"{q['camera_b']['name']}.jpg"),
            bbox_objs,
            q["camera_b"]["position"],
            q["camera_b"]["look_at"],
            output_path=os.path.join(out_dir, iname_b),
            horizontal_fov_deg=fov,
            obj_ids=_mv_ids,
            cam_name=q["camera_b"]["name"],
            seg_mask_dir=seg_mask_dir,
            instance_id_map=instance_id_map,
        )
        q["image_path_a"] = iname_a
        q["image_path_b"] = iname_b
        print(
            f"    [{i}] answer={q['answer']}  blue={blue_m['label']}  red={red_m['label']}"
        )
    elif qtype in ("comparative_spatial_grounding", "ordinal_grounding"):
        iname = f"question_image_{i}.jpg"
        shutil.copy2(src_image, os.path.join(out_dir, iname))
        q["image_path"] = iname
        generate_ground_truth(
            q,
            out_dir,
            i,
            images_dir=images_dir,
            fov=fov,
            seg_mask_dir=seg_mask_dir,
            instance_id_map=instance_id_map,
        )
        extra = q.get(
            "area_name", q.get("object", q.get("surface", {})).get("label", "")
        )
        print(f"    [{i}] answer={q['answer'][:60]}...  {extra}")
    elif qtype in (
        "object_grounding",
        "camera_region_position",
        "object_region_position",
        "region_region_position",
        "route_planning",
    ):
        iname = f"question_image_{i}.jpg"
        shutil.copy2(src_image, os.path.join(out_dir, iname))
        q["image_path"] = iname
        if qtype == "object_grounding":
            generate_ground_truth(
                q,
                out_dir,
                i,
                images_dir=images_dir,
                fov=fov,
                seg_mask_dir=seg_mask_dir,
                instance_id_map=instance_id_map,
            )
            print(f"    [{i}] answer={q['answer']}  label={q['object']['label']}")
        elif qtype == "camera_region_position":
            print(f"    [{i}] answer={q['answer']}  area={q['area_name']}")
        elif qtype == "object_region_position":
            print(
                f"    [{i}] answer={q['answer']}  obj={q['object']['label']}  area={q['area_name']}"
            )
        elif qtype == "region_region_position":
            print(
                f"    [{i}] answer={q['answer']}  from={q['area_a']}  to={q['area_b']}"
            )
        elif qtype == "route_planning":
            print(
                f"    [{i}] answer={q['answer']}  {q['start_object']} -> {q['destination_object']}"
            )
    elif qtype == "compound_spatial_referring":
        iname = f"question_image_{i}.jpg"
        shutil.copy2(src_image, os.path.join(out_dir, iname))
        q["image_path"] = iname
        if q.get("sub_type") == "referring":
            generate_ground_truth(
                q,
                out_dir,
                i,
                images_dir=images_dir,
                fov=fov,
                seg_mask_dir=seg_mask_dir,
                instance_id_map=instance_id_map,
            )
            print(
                f"    [{i}] sub=referring  answer={q['answer']}  label={q['object']['label']}"
            )
        else:
            if q.get("context_camera"):
                iname_ctx = f"question_image_{i}_context.jpg"
                shutil.copy2(
                    os.path.join(images_dir, f"{q['context_camera']['name']}.jpg"),
                    os.path.join(out_dir, iname_ctx),
                )
                q["image_path_context"] = iname_ctx
            print(f"    [{i}] sub=multi_step  answer={q['answer']}")
    elif qtype == "spatial_imagination":
        _si_colors = {"green": GREEN, "blue": BLUE, "red": RED, "yellow": YELLOW}
        marked = [q["move_to"], q["face_toward"], q["query_object"]]
        if "reference_object" in q:
            marked.append(q["reference_object"])
        bbox_objs = [
            (
                tuple(m["centroid"]),
                (m.get("width", 0.3), m.get("length", 0.3), m.get("height", 0.3)),
                _si_colors[m["circle_color"]],
            )
            for m in marked
        ]
        _si_ids = [m.get("id") for m in marked]
        iname = f"question_image_{i}.jpg"
        draw_multiple_bboxes(
            src_image,
            bbox_objs,
            q["camera_position"],
            q["camera_look_at"],
            output_path=os.path.join(out_dir, iname),
            horizontal_fov_deg=fov,
            obj_ids=_si_ids,
            cam_name=q.get("camera_name"),
            seg_mask_dir=seg_mask_dir,
            instance_id_map=instance_id_map,
        )
        q["image_path"] = iname
        if q.get("context_camera"):
            iname_ctx = f"question_image_{i}_context.jpg"
            shutil.copy2(
                os.path.join(images_dir, f"{q['context_camera']['name']}.jpg"),
                os.path.join(out_dir, iname_ctx),
            )
            q["image_path_context"] = iname_ctx
        print(f"    [{i}] sub={q['sub_type']}  answer={q['answer']}")
    elif qtype == "relative_direction":
        iname = f"question_image_{i}.jpg"
        blue_m, red_m = q["blue"], q["red"]
        cam_name = q.get("camera_name", "")
        from PIL import Image as _PILImg, ImageDraw as _IDraw

        _img = _PILImg.open(src_image).convert("RGB")
        _draw = _IDraw.Draw(_img)
        for obj_m, color in [(blue_m, BLUE), (red_m, RED)]:
            bb = _bbox_from_seg_mask(
                obj_m.get("id"),
                cam_name,
                seg_mask_dir,
                instance_id_map,
            )
            if bb is None:
                bb = _project_obj_bbox_2d(
                    tuple(obj_m["centroid"]),
                    (
                        obj_m.get("width", 0.3),
                        obj_m.get("length", 0.3),
                        obj_m.get("height", 0.3),
                    ),
                    tuple(q["camera_position"]),
                    tuple(q["camera_look_at"]),
                    _img.width,
                    _img.height,
                    horizontal_fov_deg=fov,
                )
            if bb is not None:
                _draw.rectangle(bb, outline=color, width=6)
        out_path = os.path.join(out_dir, iname)
        os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
        _img.save(out_path, quality=95)
        q["image_path"] = iname
        print(
            f"    [{i}] answer={q['answer']}  blue={q['blue']['label']}  red={q['red']['label']}"
        )
    else:
        iname = f"question_image_{i}.jpg"
        _blue_snap = _red_snap = None
        if seg_mask_dir:
            _cam_name = q.get("camera_name", "")
            for _tag in ("blue", "red"):
                _obj = q.get(_tag, {})
                _snapped = _snap_to_mask(
                    0, 0, _obj.get("id"), _cam_name, seg_mask_dir, instance_id_map
                )
                if _tag == "blue":
                    _blue_snap = _snapped
                else:
                    _red_snap = _snapped
        draw_circles_on_image(
            src_image,
            tuple(q["blue"]["centroid"]),
            tuple(q["red"]["centroid"]),
            q["camera_position"],
            q["camera_look_at"],
            output_path=os.path.join(out_dir, iname),
            horizontal_fov_deg=fov,
            override_blue_pixel=_blue_snap,
            override_red_pixel=_red_snap,
        )
        q["image_path"] = iname
        suffix = " m" if qtype == "object_distance" else ""
        print(
            f"    [{i}] answer={q['answer']}{suffix}  "
            f"blue={q['blue']['label']}  red={q['red']['label']}"
        )
