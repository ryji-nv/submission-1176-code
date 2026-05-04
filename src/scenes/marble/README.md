# Marble Pipeline

Single image or text prompt → 3D Gaussian Splat → segmentation → camera placement → QA generation.

## Quick Start

```bash
# Single image
python run.py --env marble --image photo.jpg --question-type all -n 50

# Text-to-scene
python run.py --env marble --text-prompt "a cozy kitchen" --question-type all

# Generation only (no QA)
python run.py --env marble --image photo.jpg --generation-only

# Batch mode (JSON input, 8 GPUs parallel)
python run.py --env marble --input-list inputs.json --question-type all -n 50 --output-root output_batch

# Rerun existing scene from a specific step
python run.py --env marble --scene-id 0000_txt_53487ac0730a --start-from 4b --question-type all --force-rerun -n 50 --output-root output_batch

# Rerun all scenes in batch from a specific step
python run.py --env marble --input-list /tmp/all_scenes.json --start-from 4b --question-type all --force-rerun -n 50 --output-root output_batch
```

## Batch Mode

Input JSON format (matches `captions.json`):
```json
[
  {"prompt": "a warm kitchen with...", "meta_info": {"image_pth": "/path/to/ref.jpg", "source": "dataset"}},
  {"prompt": "a modern office..."}
]
```

Two-phase execution:
- **Phase 1**: Marble API calls in parallel (10 threads, network-bound, ~5 min each)
- **Phase 2**: GPU pipeline in parallel (8 threads, 1 per GPU — seg + render + QA)

## Pipeline Steps

| Step | What | Time |
|------|------|------|
| 1 | Marble API → SPZ download | ~5 min |
| 2 | SPZ → PLY conversion | <1s |
| 3 | SAM3 segmentation (12 views) | ~60s |
| 4c | VLM 2D merge+label (multi-round, per-view) | ~30s |
| 4 | 3D labeling (project + vote + cluster) | ~13s |
| 4b | Postprocess (split + VLM 3D merge + VLM 3D filter) | ~50s |
| 5 | Camera generation + rendering | ~15s |
| 5b | Visibility computation (GPU) | ~3s |
| 5c | Dimension filter | <1s |
| 5e | VLM camera quality filter | ~10s |
| 6 | QA generation (32 types × up to 50 each) | ~5s |
| 7 | Local output finalization | varies |

`--generation-only` stops after step 5e.

### Step 4c: VLM 2D Merge+Label

Per-view 2D mask processing using the configured VLM:
1. **Pre-filter** — remove fragments < 500px
2. **Stage 1a** — propose merges + removals per view
3. **Stage 1b** — verify each merge pair (isolated overlays + RGB + panorama)
4. **Stage 1c** — check rejected-merge segments for removal
5. **Stage 2** — assign category labels per view
6. **Post-label filter** — remove "STRUCTURE" labels + thin strips

### Step 4b: Postprocess

Operates on 3D Gaussian instances after voting:
1. **Split components** — separate disconnected clusters (eps=0.10m)
2. **Spatial overlap merge** — merge co-located different-label instances (5cm/70% threshold)
3. **VLM 3D merge** (3 iterations) — ask VLM whether nearby pairs should merge, using RGB + 3D seg + panorama. Merges adjacent parts of same object, sub-components, and large connected structures (e.g. cabinets)
4. **VLM 3D filter** — verify each instance, remove only clear noise (scattered points, unidentifiable blobs, completely wrong labels)

## Camera Placement

- **64 view cameras**: orbital views from segmentation
- **64 random cameras**: uniformly sampled inside 70% convex hull of objects
  - Height: 0.7–1.75m
  - Look-at: nearest object direction (near/mid/far selection), maximizing FOV object count
  - Diversity: rejects cameras too close in position or direction to existing ones
  - Quality: alpha >= 0.995 from gsplat (retry up to 10 rounds of 32)
  - Post-process: bilateral filter to smooth Gaussian splat artifacts

## Output Structure

```
output/
  0000_scene_name/
    input.json          # full entry from batch JSON
    input.jpg           # source image (if image input)
    meta/
      scene.json        # QA-format scene (objects, cameras, room_bounds)
      scene.ply         # Gaussian splat PLY
      visibility.json   # per-camera visible objects
      dimension_filter.json
      images/           # rendered RGB + seg masks
      segmentation/     # SAM3 + 3D labels
    tasks/
      000_closest_object/
      001_depth_estimation/
      ...
```

## Key Files

| File | Role |
|------|------|
| `pipeline.py` | Orchestrates all steps, caching, retry logic |
| `scene.py` | Camera generation, gsplat rendering (batched GPU), scene.json |
| `quality.py` | Alpha-based quality filter + bilateral smoothing |
| `marble_api.py` | World Labs API client (text/image → 3D world) |
| `segment_scene.py` | SAM3 segmentation + view rendering |
| `label_gaussians.py` | VLM-based 3D instance labeling |
| `postprocess.py` | Split + spatial merge + VLM 3D merge + VLM 3D filter |

## Environment

```bash
export WLT_API_KEY=...          # World Labs Marble API
export VLM_API_KEY=...       # VLM labeling + dimension filter
```

Dependencies: `gsplat`, `sam3`, `gaussforge`, `torch`, `opencv-python`, `plyfile`
