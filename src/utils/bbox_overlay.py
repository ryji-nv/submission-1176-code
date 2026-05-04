"""
Generate bbox-annotated images from a scene.json + rendered camera images.

For each camera, draws visible-object wireframe 3D bounding boxes and saves
as 3d_bbox_images/{camera_name}.png under the scene directory.
"""

from __future__ import annotations

import math
import os

from .projection import project_world_to_pixel
from .occlusion import filter_visible_objects, object_centroid

_GREEN = (0, 255, 0)
_BBOX_EDGES = [
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
_CORNER_SIGNS = [
    (-1, -1, -1),
    (1, -1, -1),
    (-1, 1, -1),
    (1, 1, -1),
    (-1, -1, 1),
    (1, -1, 1),
    (-1, 1, 1),
    (1, 1, 1),
]


def generate_bbox_images(
    scene_json: str,
    images_dir: str,
    *,
    flip_y: bool = False,
    color: tuple[int, int, int] = _GREEN,
    line_width: int = 2,
) -> list[str]:
    """
    For each camera in scene.json, draw 3D bounding boxes of visible objects on the
    rendered image and save as {scene_dir}/3d_bbox_images/{camera_name}.png.

    Returns a list of output paths written.
    """
    try:
        from PIL import Image, ImageDraw
    except ImportError:
        raise ImportError("PIL is required. Install with: pip install Pillow")

    import json

    with open(scene_json) as f:
        scene = json.load(f)

    cameras = scene.get("cameras", [])
    objects = scene.get("objects", [])
    scene_dir = os.path.dirname(os.path.abspath(scene_json))
    out_dir = os.path.join(scene_dir, "3d_bbox_images")
    os.makedirs(out_dir, exist_ok=True)
    written = []

    for camera in cameras:
        if camera.get("step_direction"):
            continue  # stepped cameras share the same scene; skip to avoid missing-file noise
        src_image = os.path.join(images_dir, f"{camera['name']}.jpg")
        if not os.path.isfile(src_image):
            continue
        dst_image = os.path.join(out_dir, f"{camera['name']}.png")

        visible = filter_visible_objects(camera, objects)
        fov = camera.get("horizontal_fov_deg", 82.0)
        cam_pos = tuple(camera["position"])
        look_at = tuple(camera["look_at"])

        img = Image.open(src_image).convert("RGB")
        draw = ImageDraw.Draw(img)
        w, h = img.size

        for obj in visible:
            cx, cy, cz = object_centroid(obj)
            hw = (obj.get("width") or 0.0) / 2
            hl = (obj.get("length") or 0.0) / 2
            hh = (obj.get("height") or 0.0) / 2
            theta = math.radians(obj.get("rotation_z") or 0.0)
            cos_t, sin_t = math.cos(theta), math.sin(theta)
            # Oriented corners: rotate width (X) and length (Y) axes by rotation_z
            corners = [
                (
                    cx + sx * hw * cos_t - sy * hl * sin_t,
                    cy + sx * hw * sin_t + sy * hl * cos_t,
                    cz + sz * hh,
                )
                for sx, sy, sz in _CORNER_SIGNS
            ]
            pixels = [
                project_world_to_pixel(
                    pt,
                    cam_pos,
                    look_at,
                    w,
                    h,
                    flip_y=flip_y,
                    horizontal_fov_deg=fov,
                    clamp=False,
                )
                for pt in corners
            ]
            for i, j in _BBOX_EDGES:
                if pixels[i] and pixels[j]:
                    draw.line([pixels[i], pixels[j]], fill=color, width=line_width)

        img.save(dst_image)
        written.append(dst_image)

    return written
