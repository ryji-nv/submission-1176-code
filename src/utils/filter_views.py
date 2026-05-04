"""
Filter camera views using a VLM to assess suitability for spatial reasoning QA.

A view is GOOD if:
  - Lighting is sufficient to clearly see objects
  - No large occlusions blocking most of the scene
  - Objects are distinguishable (not washed out, too dark, or mostly hidden)

Results are written to {scene_dir}/view_filter.json and read by generate_questions.py
to skip BAD camera views.

Called via filter_scene() from src.sage.pipeline and src.scenesmith.pipeline.
Requires VLM_API_KEY env var; skipped with a warning if not set.
"""

import json
import os
import re
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed

from src.utils.vlm import call_vlm, encode_image

JUDGE_PROMPT = """\
You are a quality-control inspector for synthetic indoor scene images used to generate spatial reasoning questions.

You will be shown a camera view of an indoor scene. Evaluate it along two quality dimensions:

1. **poor_lighting** – Is the lighting too dark, too bright/washed out, or uneven such that objects are hard to identify?
2. **large_occlusion** – Does a wall, door frame, or large object severely block the view, leaving very few visible objects in the scene?

Then determine overall **quality**:
- **BAD** if ANY dimension above is true.
- **GOOD** if neither is true — objects are clearly visible and the scene is well-lit.

First, reason step by step inside <think></think> tags.

Then respond with a JSON object inside a ```json``` code fence:
```json
{
  "quality": "GOOD" or "BAD",
  "poor_lighting": true or false,
  "large_occlusion": true or false,
  "reason": "<brief explanation>"
}
```
"""


def _build_messages(image_path: str) -> list[dict]:
    image_url = encode_image(image_path)
    return [
        {
            "role": "user",
            "content": [
                {"type": "image_url", "image_url": {"url": image_url}},
                {"type": "text", "text": JUDGE_PROMPT},
            ],
        }
    ]


def _parse_response(content: str) -> dict:
    text = content.strip()
    match = re.search(r"```(?:json)?\s*\n?(.*?)```", text, re.DOTALL)
    if match:
        text = match.group(1).strip()
    return json.loads(text)


def judge_image(image_path: str) -> dict:
    """
    Judge a single camera image for spatial reasoning suitability.

    Returns dict with keys: quality, poor_lighting, large_occlusion, reason, thinking.
    """
    messages = _build_messages(image_path)
    content, thinking = call_vlm(messages)
    parsed = _parse_response(content)
    return {
        "quality": parsed.get("quality", "UNKNOWN").upper(),
        "poor_lighting": bool(parsed["poor_lighting"])
        if "poor_lighting" in parsed
        else None,
        "large_occlusion": bool(parsed["large_occlusion"])
        if "large_occlusion" in parsed
        else None,
        "reason": parsed.get("reason", ""),
        "thinking": thinking,
    }


def filter_scene(scene_json_path: str, force_rerun: bool = False) -> list[dict] | None:
    """
    Judge all non-stepped camera images for a scene.

    Returns None if VLM_API_KEY is not set (skipped with a warning).
    Reads cameras from scene.json, judges each image, and writes results to
    {scene_dir}/view_filter.json. Returns the list of result dicts.
    """
    if not os.environ.get("VLM_API_KEY", "").strip():
        raise RuntimeError("VLM_API_KEY is not set — cannot run VLM view filter.")

    scene_dir = os.path.dirname(scene_json_path)
    out_path = os.path.join(scene_dir, "view_filter.json")

    if not force_rerun and os.path.isfile(out_path):
        print(
            "  [filter] skipped (view_filter.json exists; use --force-rerun to regenerate)"
        )
        with open(out_path) as f:
            return json.load(f)

    with open(scene_json_path) as f:
        scene = json.load(f)

    images_dir = os.path.join(scene_dir, "images")
    cameras = [c for c in scene.get("cameras", []) if not c.get("step_direction")]

    jobs = []
    for cam in cameras:
        image_path = os.path.join(images_dir, f"{cam['name']}.jpg")
        if not os.path.isfile(image_path):
            print(
                f"  [filter] image not found, skipping: {image_path}", file=sys.stderr
            )
            continue
        jobs.append((cam["name"], image_path))

    print(f"  [filter] judging {len(jobs)} camera(s) in parallel ...")

    results_map: dict[str, dict] = {}
    with ThreadPoolExecutor(max_workers=len(jobs)) as executor:
        futures = {
            executor.submit(judge_image, image_path): cam_name
            for cam_name, image_path in jobs
        }
        for future in as_completed(futures):
            cam_name = futures[future]
            result = future.result()
            result["camera_name"] = cam_name
            results_map[cam_name] = result
            print(
                f"    {cam_name}: quality={result['quality']}  "
                f"poor_lighting={result['poor_lighting']}  "
                f"large_occlusion={result['large_occlusion']}  "
                f"reason: {result['reason']}"
            )

    # Preserve original camera order
    results = [results_map[name] for name, _ in jobs if name in results_map]

    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    good = sum(1 for r in results if r["quality"] == "GOOD")
    print(f"  [filter] {good}/{len(results)} views GOOD → {out_path}")

    return results
