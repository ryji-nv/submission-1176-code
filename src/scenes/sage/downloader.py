#!/usr/bin/env python3
"""
Download SAGE scenes from HuggingFace (SAGE-10k) into ~/scenes.

Usage:
  python scripts/download_scenes.py --start 0 --end 5
  python scripts/download_scenes.py --start 10 --end 20 --scenes-dir /data/scenes

Requires: pip install huggingface_hub
Auth:     set HF_TOKEN or HUGGINGFACE_TOKEN env var (or huggingface-cli login)
"""

from __future__ import annotations

import argparse
import os
import shutil
import sys
import zipfile
from pathlib import Path

HF_REPO_ID = "SAGE-10k"


def list_scenes(repo_id: str) -> list[str]:
    try:
        from huggingface_hub import list_repo_tree
    except ImportError:
        print(
            "Error: huggingface_hub is required. Run: pip install huggingface_hub",
            file=sys.stderr,
        )
        sys.exit(1)
    paths = [
        n.path
        for n in list_repo_tree(
            repo_id=repo_id, path_in_repo="scenes", repo_type="dataset"
        )
        if n.path.endswith(".zip")
    ]
    return sorted(os.path.splitext(os.path.basename(p))[0] for p in paths)


def download_scene(scene_id: str, scenes_dir: Path, repo_id: str) -> bool:
    """Download and extract one scene. Skip if already present. Returns True on success."""
    scene_dir = scenes_dir / scene_id
    if scene_dir.is_dir() and list(scene_dir.glob("layout_*.json")):
        print(f"  [skip] already present: {scene_dir}")
        return True

    try:
        from huggingface_hub import hf_hub_download
    except ImportError:
        print(
            "Error: huggingface_hub is required. Run: pip install huggingface_hub",
            file=sys.stderr,
        )
        sys.exit(1)

    scenes_dir.mkdir(parents=True, exist_ok=True)
    print(f"  Downloading {scene_id} ...", flush=True)
    zip_path = Path(
        hf_hub_download(
            repo_id=repo_id,
            filename=f"scenes/{scene_id}.zip",
            repo_type="dataset",
            local_dir=scenes_dir.parent,
        )
    )

    print(f"  Extracting to {scene_dir} ...", flush=True)
    scene_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(scene_dir)

    # Flatten nested dir if zip extracted into <scene_id>/<scene_id>/
    nested = scene_dir / scene_id
    if (
        nested.is_dir()
        and list(nested.glob("layout_*.json"))
        and not list(scene_dir.glob("layout_*.json"))
    ):
        for p in nested.iterdir():
            shutil.move(str(p), str(scene_dir / p.name))
        nested.rmdir()

    if not list(scene_dir.glob("layout_*.json")):
        print(f"  [warn] no layout_*.json found in {scene_dir}", file=sys.stderr)
        return False

    print(f"  Done: {scene_dir}")
    return True


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument(
        "--start",
        type=int,
        default=0,
        metavar="N",
        help="Start index (inclusive, default: 0)",
    )
    ap.add_argument(
        "--end", type=int, required=True, metavar="N", help="End index (exclusive)"
    )
    ap.add_argument(
        "--scenes-dir",
        default=None,
        metavar="DIR",
        help="Destination directory (default: $SCENES_DIR or ~/scenes)",
    )
    ap.add_argument(
        "--repo",
        default=HF_REPO_ID,
        metavar="REPO",
        help=f"HuggingFace dataset repo (default: {HF_REPO_ID})",
    )
    args = ap.parse_args()

    scenes_dir = (
        Path(args.scenes_dir or os.environ.get("SCENES_DIR", "~/scenes"))
        .expanduser()
        .resolve()
    )

    print(f"Listing scenes from {args.repo} ...")
    all_scenes = list_scenes(args.repo)
    selected = all_scenes[args.start : args.end]

    if not selected:
        print(
            f"No scenes in range [{args.start}, {args.end}) — repo has {len(all_scenes)} scenes total."
        )
        return 1

    print(f"Downloading {len(selected)} scene(s) to {scenes_dir}\n")
    n_ok = sum(download_scene(sid, scenes_dir, args.repo) for sid in selected)
    print(f"\nDone: {n_ok}/{len(selected)} scene(s) ready.")
    return 0 if n_ok == len(selected) else 1


if __name__ == "__main__":
    sys.exit(main())
