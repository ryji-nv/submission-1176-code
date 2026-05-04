"""
Resolve SAGE scene layout paths and parse layout JSON for object coordinates.

Logic copied from MobilityGen/sdg/sage_scene.py (get_room_and_objects).
No changes to MobilityGen; no actor-simulation dependency.
"""

import glob
import json
import os


def list_scene_ids(n: int, *, seed: int | None = None) -> list[str]:
    """Return N SAGE scene IDs from HuggingFace.

    If *seed* is given, randomly sample N scenes (different seed → different
    selection).  Otherwise take the first N in sorted order.
    """
    import random
    from .downloader import list_scenes, HF_REPO_ID

    print(f"Fetching SAGE scene list from {HF_REPO_ID} ...")
    all_ids = list_scenes(HF_REPO_ID)
    if not all_ids:
        raise RuntimeError(f"No scenes found in {HF_REPO_ID}.")
    if seed is not None:
        rng = random.Random(seed)
        n = min(n, len(all_ids))
        scene_ids = rng.sample(all_ids, n)
        print(f"  Randomly selected {n} scene(s) (seed={seed})")
    else:
        scene_ids = all_ids[:n]
    return scene_ids


def get_layout_path(scene_id: str) -> str | None:
    """Download scene_id if needed and return its layout path, or None on failure."""
    from .downloader import download_scene, HF_REPO_ID
    from pathlib import Path

    scenes_dir = Path(get_scenes_dir()).expanduser().resolve()
    ok = download_scene(scene_id, scenes_dir, HF_REPO_ID)
    if not ok:
        return None
    hits = glob.glob(os.path.join(scenes_dir, scene_id, "layout_*.json"))
    return os.path.abspath(hits[0]) if hits else None


def get_scenes_dir() -> str:
    """Base directory for SAGE scenes (default: <project>/scenes)."""
    default = os.path.join(
        os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "scenes"
    )
    return os.path.expanduser(os.environ.get("SCENES_DIR", default))


def get_room_bounds(layout_json_path: str) -> tuple[float, float, float, float, float]:
    """
    Read layout JSON; return room bounds (rx, ry, rw, rl, rh) for first room.
    Matches SAGE/actor-simulation convention: width rw, length rl, height rh.
    """
    with open(layout_json_path) as f:
        layout_data = json.load(f)
    rooms = layout_data.get("rooms", [])
    room = rooms[0] if rooms else {}
    rdim = room.get("dimensions", {"width": 5, "length": 7, "height": 2.7})
    rpos = room.get("position", {"x": 0, "y": 0, "z": 0})
    rx = float(rpos.get("x", 0))
    ry = float(rpos.get("y", 0))
    rw = float(rdim.get("width", 5))
    rl = float(rdim.get("length", 7))
    rh = float(rdim.get("height", 2.7))
    return (rx, ry, rw, rl, rh)


def load_layout_objects(layout_json_path: str) -> list[dict]:
    """
    Read layout JSON and return a list of object dicts with coordinates and metadata.

    Each dict has at least: x, y, z, width, length, (height if in JSON), and any
    other keys from the layout (e.g. name, id, type). Room index and object index
    within room are included for reference.
    """
    with open(layout_json_path) as f:
        layout_data = json.load(f)
    rooms = layout_data.get("rooms", [])
    result = []
    for ri, rd in enumerate(rooms):
        for oi, obj in enumerate(rd.get("objects", [])):
            p = obj.get("position", {"x": 0, "y": 0, "z": 0})
            d = obj.get("dimensions", {"width": 0.5, "length": 0.5, "height": 0.5})
            entry = {
                "room_index": ri,
                "object_index": oi,
                "x": float(p.get("x", 0)),
                "y": float(p.get("y", 0)),
                "z": float(p.get("z", 0)),
                "width": float(d.get("width", 0.5)),
                "length": float(d.get("length", 0.5)),
                "height": float(d.get("height", 0.5)),
            }
            for k, v in obj.items():
                if k not in ("position", "dimensions"):
                    entry[k] = v
            result.append(entry)
    return result
