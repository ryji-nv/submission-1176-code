#!/usr/bin/env python3
"""
Download and discover SceneSmith scenes from HuggingFace (nepfaff/scenesmith-example-scenes).

Usage:
  python scripts/scenesmith_loader.py --subset House --start 0 --end 5
  python scripts/scenesmith_loader.py --list --subset House

Scenes are extracted to $SCENES_DIR/scenesmith/<scene_id>/ (default: ~/scenes/scenesmith/).
Requires: pip install huggingface_hub
Auth:     set HF_TOKEN or HUGGINGFACE_TOKEN env var
"""

from __future__ import annotations

import argparse
import os
import sys
import tarfile
from pathlib import Path

SCENESMITH_REPO_ID = "nepfaff/scenesmith-example-scenes"
SCENESMITH_DEFAULT_SUBSET = "House"


def _hf_token() -> str | None:
    return os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_TOKEN")


def list_scenesmith_scenes(subset: str = SCENESMITH_DEFAULT_SUBSET) -> list[str]:
    """List available scene IDs for the given subset from HuggingFace. Returns sorted list."""
    try:
        from huggingface_hub import list_repo_tree
    except ImportError:
        print(
            "Error: huggingface_hub is required. Run: pip install huggingface_hub",
            file=sys.stderr,
        )
        sys.exit(1)
    paths = [
        node.path
        for node in list_repo_tree(
            repo_id=SCENESMITH_REPO_ID,
            path_in_repo=subset,
            repo_type="dataset",
            token=_hf_token(),
        )
        if getattr(node, "path", "").endswith(".tar")
    ]
    return sorted(Path(p).stem for p in paths)


def download_scenesmith_scene(
    scene_id: str,
    scenes_dir: Path,
    subset: str = SCENESMITH_DEFAULT_SUBSET,
) -> bool:
    """
    Download and extract {subset}/{scene_id}.tar from HuggingFace.
    Extracts to scenes_dir/<scene_id>/. Skips if USD already present.
    Returns True on success.
    """
    local_dir = Path(scenes_dir).resolve() / scene_id
    if local_dir.is_dir() and list(local_dir.glob("mujoco/usd/*.usd*")):
        print(f"  [skip] already present: {local_dir}")
        return True

    try:
        from huggingface_hub import hf_hub_download
    except ImportError:
        print(
            "Error: huggingface_hub is required. Run: pip install huggingface_hub",
            file=sys.stderr,
        )
        sys.exit(1)

    local_dir.mkdir(parents=True, exist_ok=True)
    tar_filename = f"{subset}/{scene_id}.tar"
    print(f"  Downloading {tar_filename} from {SCENESMITH_REPO_ID} ...", flush=True)

    try:
        tar_path = Path(
            hf_hub_download(
                repo_id=SCENESMITH_REPO_ID,
                filename=tar_filename,
                repo_type="dataset",
                local_dir=local_dir.parent,
                token=_hf_token(),
            )
        )
    except Exception as e:
        print(f"  [error] download failed: {e}", file=sys.stderr)
        return False

    print(f"  Extracting {tar_path.name} ...", flush=True)
    with tarfile.open(tar_path, "r") as tf:
        tf.extractall(local_dir)

    # Flatten if tar extracted into a nested scene_id/ subdirectory
    nested = local_dir / scene_id
    if nested.is_dir() and not list(local_dir.glob("mujoco")):
        import shutil

        for p in nested.iterdir():
            shutil.move(str(p), str(local_dir / p.name))
        nested.rmdir()

    usd_files = list(local_dir.glob("mujoco/usd/*.usd*"))
    if not usd_files:
        print(f"  [warn] no USD file found in {local_dir}/mujoco/usd/", file=sys.stderr)
        return False

    print(f"  Done: {local_dir}")
    return True


def get_scenesmith_usd_path(scene_id: str, scenes_dir: Path) -> str:
    """Return the USD file path for a downloaded SceneSmith scene. Raises if not found."""
    usd_dir = Path(scenes_dir).resolve() / scene_id / "mujoco" / "usd"
    usd_files = sorted(usd_dir.glob("*.usd*"))
    if not usd_files:
        raise FileNotFoundError(
            f"No USD file found in {usd_dir}. "
            f"Run: python scripts/scenesmith_loader.py --subset House --start 0 --end N"
        )
    return str(usd_files[0])


def get_scenesmith_scenes_dir(base_scenes_dir: Path | str | None = None) -> Path:
    """Return the scenesmith scenes directory (default: <project>/scenes/scenesmith/)."""
    default = os.path.join(
        os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "scenes"
    )
    base = (
        Path(base_scenes_dir or os.environ.get("SCENES_DIR", default))
        .expanduser()
        .resolve()
    )
    return base / "scenesmith"


def list_scenesmith_scene_ids(
    n: int, subset: str, *, seed: int | None = None
) -> list[str]:
    """Return N SceneSmith scene IDs for the given subset.

    If *seed* is given, shuffle the full list so each pod processes scenes
    in a different order.  Otherwise return sorted order.
    """
    import random

    print(f"Fetching SceneSmith scene list (subset={subset}) ...")
    all_ids = list_scenesmith_scenes(subset)
    if not all_ids:
        raise RuntimeError(f"No SceneSmith scenes found for subset '{subset}'.")
    if seed is not None:
        rng = random.Random(seed)
        rng.shuffle(all_ids)
        print(f"  Shuffled scene order (seed={seed})")
    return all_ids[:n]


def get_scenesmith_usd_path_for_scene(sid: str, subset: str) -> str | None:
    """Download scene sid if needed and return its USD path, or None on failure."""
    scenes_dir = get_scenesmith_scenes_dir()
    ok = download_scenesmith_scene(sid, scenes_dir, subset)
    if not ok:
        return None
    try:
        return get_scenesmith_usd_path(sid, scenes_dir)
    except FileNotFoundError as e:
        print(f"  [warn] {e}", file=sys.stderr)
        return None


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument(
        "--subset",
        default=SCENESMITH_DEFAULT_SUBSET,
        help=f"Scene subset (default: {SCENESMITH_DEFAULT_SUBSET})",
    )
    ap.add_argument(
        "--list", action="store_true", help="List available scene IDs and exit"
    )
    ap.add_argument("--start", type=int, default=0, metavar="N")
    ap.add_argument("--end", type=int, default=None, metavar="N")
    ap.add_argument(
        "--scenes-dir",
        default=None,
        metavar="DIR",
        help="Base scenes directory (default: $SCENES_DIR or ~/scenes)",
    )
    args = ap.parse_args()

    if args.list:
        print(f"Listing {args.subset} scenes from {SCENESMITH_REPO_ID} ...")
        scenes = list_scenesmith_scenes(args.subset)
        for s in scenes:
            print(f"  {s}")
        print(f"Total: {len(scenes)}")
        return 0

    scenes_dir = get_scenesmith_scenes_dir(args.scenes_dir)
    print(f"Listing {args.subset} scenes from {SCENESMITH_REPO_ID} ...")
    all_scenes = list_scenesmith_scenes(args.subset)
    end = args.end if args.end is not None else len(all_scenes)
    selected = all_scenes[args.start : end]

    if not selected:
        print(
            f"No scenes in range [{args.start}, {end}) — {len(all_scenes)} available."
        )
        return 1

    print(f"Downloading {len(selected)} scene(s) to {scenes_dir}\n")
    n_ok = sum(
        download_scenesmith_scene(sid, scenes_dir, args.subset) for sid in selected
    )
    print(f"\nDone: {n_ok}/{len(selected)} scene(s) ready.")
    return 0 if n_ok == len(selected) else 1


if __name__ == "__main__":
    sys.exit(main())
