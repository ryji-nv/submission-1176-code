"""Image quality filtering and post-processing for Marble renders.

Uses the mean alpha from gsplat rasterization as the quality signal
(set on each camera dict during rendering).  Bilateral filtering is
parallelised across CPU cores to smooth Gaussian splat artifacts.
"""

from __future__ import annotations

import os
from concurrent.futures import ThreadPoolExecutor

import cv2


def _bilateral_one(path: str, d: int = 3,
                   sigma_color: int = 25, sigma_space: int = 25) -> None:
    img = cv2.imread(path)
    if img is None:
        return
    filtered = cv2.bilateralFilter(img, d, sigma_color, sigma_space)
    cv2.imwrite(path, filtered, [cv2.IMWRITE_JPEG_QUALITY, 95])


def filter_and_smooth(
    images_dir: str,
    cameras: list[dict],
    *,
    min_alpha: float = 0.995,
    bilateral_workers: int = 8,
) -> tuple[list[dict], list[dict]]:
    """Keep rand cameras with sufficient alpha coverage, smooth the keepers.

    ``mean_alpha`` is expected to be set on each camera dict by
    ``render_cameras`` during the RGB phase.  Non-rand cameras are
    always kept.

    Returns (good_cameras, bad_cameras).
    """
    rand_cams = [c for c in cameras if c["name"].startswith("rand_")]
    non_rand = [c for c in cameras if not c["name"].startswith("rand_")]

    min_laplacian = 10.0

    good, bad = list(non_rand), []
    good_paths: list[str] = []
    for cam in rand_cams:
        alpha = cam.get("mean_alpha", 1.0)
        img_path = os.path.join(images_dir, f"{cam['name']}.jpg")
        passed = alpha >= min_alpha
        if passed and os.path.isfile(img_path):
            lap = cv2.Laplacian(
                cv2.imread(img_path, cv2.IMREAD_GRAYSCALE), cv2.CV_64F,
            ).var()
            if lap < min_laplacian:
                passed = False
        if passed:
            good.append(cam)
            if os.path.isfile(img_path):
                good_paths.append(img_path)
        else:
            bad.append(cam)
            if os.path.isfile(img_path):
                os.remove(img_path)
            seg = os.path.join(images_dir, f"{cam['name']}_seg.npy")
            if os.path.isfile(seg):
                os.remove(seg)

    if good_paths:
        with ThreadPoolExecutor(max_workers=bilateral_workers) as pool:
            pool.map(_bilateral_one, good_paths)

    return good, bad


def filter_cameras_vlm(
    images_dir: str,
    cameras: list[dict],
    *,
    model: str = os.environ.get("VLM_MODEL", "Qwen3-VL-235B-A22B-Thinking"),
    max_workers: int = 50,
) -> set[str]:
    """Send each camera image to the configured VLM to reject low-quality renders.

    Returns set of camera names that failed (should be excluded).
    """
    import base64
    import requests as _req

    api_key = os.environ.get("VLM_API_KEY", "")
    if not api_key:
        return set()

    prompt = (
        "Rate this 3D-rendered indoor scene image. "
        "Answer GOOD if the image is clear with recognizable objects "
        "and no major artifacts. "
        "Answer BAD if the image has: blurry/jittery regions, "
        "visible floating blobs, heavy occlusion (camera too close "
        "to a surface), or is mostly featureless. "
        "Answer only GOOD or BAD."
    )

    rand_cams = [c for c in cameras if c["name"].startswith("rand_")]
    if not rand_cams:
        return set()

    def _check_one(cam):
        img_path = os.path.join(images_dir, f"{cam['name']}.jpg")
        if not os.path.isfile(img_path):
            return cam["name"], False
        try:
            with open(img_path, "rb") as f:
                b64 = base64.b64encode(f.read()).decode()
            resp = _req.post(
                os.environ.get("VLM_API_BASE", "https://example.com/v1/chat/completions"),
                headers={
                    "Authorization": f"Bearer {api_key}",
                    "Content-Type": "application/json",
                },
                json={
                    "model": model,
                    "messages": [{
                        "role": "user",
                        "content": [
                            {"type": "text", "text": prompt},
                            {"type": "image_url", "image_url": {
                                "url": f"data:image/jpeg;base64,{b64}",
                                "detail": "low",
                            }},
                        ],
                    }],
                    "temperature": 0.0,
                    "max_completion_tokens": 8,
                },
                timeout=30,
            )
            resp.raise_for_status()
            answer = resp.json()["choices"][0]["message"]["content"].strip().upper()
            return cam["name"], "GOOD" in answer
        except Exception:
            return cam["name"], True

    print(f"  [cam-vlm] checking {len(rand_cams)} cameras with {model} "
          f"({max_workers} parallel)...")
    rejected: set[str] = set()
    from concurrent.futures import as_completed
    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = {pool.submit(_check_one, c): c for c in rand_cams}
        for fut in as_completed(futures):
            name, passed = fut.result()
            if not passed:
                rejected.add(name)

    if rejected:
        print(f"  [cam-vlm] rejected {len(rejected)}: {sorted(rejected)}")
    else:
        print("  [cam-vlm] all cameras passed")
    return rejected
