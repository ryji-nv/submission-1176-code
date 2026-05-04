# NeurIPS 2026 Submission 1176 Code Artifact

This repository contains the submission code artifact for NeurIPS 2026
Submission 1176. The code implements the spatial data engine used to construct
simulator-grounded VLM training records from generated 3D scenes.

The artifact is provided for reviewer inspection and reproducibility of the data
engine design. It includes source adapters, rendering / visibility utilities,
task curators, prompt templates, and output-format logic.

## What Is Included

- `run.py`: command-line entry point for scene processing and QA generation.
- `src/scenes/`: adapters and pipelines for SAGE, SceneSmith, and generated-world
  inputs.
- `src/tasks/`: task curators for distance, depth, direction, object properties,
  grounding, placement, camera/viewpoint, and complex reasoning tasks.
- `src/utils/`: camera, projection, visibility, annotation, occlusion, and
  optional VLM-filter utilities.
- `LICENSE`: terms for using the submission code artifact.

## Requirements

- Python 3.12
- `uv` or `pip`
- Isaac Sim / Isaac Sim Python for rendering-based pipelines
- CUDA-capable GPU for rendering and generated-world post-processing
- Optional environment variables:
  - `HF_TOKEN` or `HUGGINGFACE_TOKEN` for gated scene-source downloads
  - `VLM_API_KEY`, `VLM_API_BASE`, and optionally `VLM_MODEL` for VLM-based
    quality filters and generated-world labeling
  - `WLT_API_KEY` for optional generated-world API calls

The paper's dataset artifact can be inspected without running this code. Running
the full data engine requires the corresponding scene sources, rendering stack,
and optional external services.

## Installation

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install -e .
```

If using Isaac Sim, run the pipeline with the Isaac Sim Python interpreter rather
than a vanilla Python interpreter.

## Minimal Commands

Generate questions for a small number of SAGE scenes:

```bash
python run.py --env sage --num-scenes 1 --question-type closest_object --num-questions 1
```

Generate questions for a SceneSmith subset:

```bash
python run.py --env scenesmith --scenesmith-subset Room --num-scenes 1 --question-type all --num-questions 1
```

Run generated-world processing from a text prompt:

```bash
python run.py --env marble --text-prompt "a kitchen with a table and chairs" --question-type all --num-questions 1
```

The generated-world path requires the external world-generation API and
additional dependencies described in `src/scenes/marble/README.md`.

## Output Layout

Outputs are written under `output/` by default:

```text
output/<scene_id>/
  scene.json
  images/
  instance_seg/
  3d_bbox_images/
  <task_type>/
    question_0.json
    question_image_0.jpg
    question_image_0_a.jpg
    question_image_0_b.jpg
    question_mask_0.png
```

Each task JSON stores the question, answer, task type, answer format, object
references, camera references, and task-specific metadata needed for auditing.
