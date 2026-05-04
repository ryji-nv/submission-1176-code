#!/usr/bin/env python3
"""SAM3 worker: process a subset of views on a specific GPU.

Usage: python _sam3_gpu_worker.py --gpu 0 --views 0,1,2 --objects-json X --views-dir Y --out-dir Z
"""

import os, sys, json, argparse
import numpy as np
import torch

AMBIGUOUS_OVERWRITE_LABELS = {
    "speaker",
    "phone",
    "remote",
    "calculator",
    "computer mouse",
    "mouse",
    "usb drive",
    "security camera",
    "power adapter",
    "adapter",
    "charger",
}
AMBIGUOUS_OVERWRITE_PENALTY = 0.15


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gpu", type=int, required=True)
    ap.add_argument("--views", required=True, help="Comma-separated view indices")
    ap.add_argument("--objects-json", required=True, help="Path to filtered object list JSON")
    ap.add_argument("--views-dir", required=True, help="Dir with cameras.json and view PNGs")
    ap.add_argument("--out-dir", required=True, help="Dir to save instance masks")
    ap.add_argument("--min-score", type=float, default=0.35)
    args = ap.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)

    view_indices = [int(v) for v in args.views.split(",")]
    object_names = json.load(open(args.objects_json))
    cameras = json.load(open(os.path.join(args.views_dir, "cameras.json")))

    from pathlib import Path
    Path(args.out_dir).mkdir(parents=True, exist_ok=True)

    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.autocast("cuda", dtype=torch.bfloat16).__enter__()

    from sam3.model_builder import build_sam3_image_model
    from sam3.model.sam3_image_processor import Sam3Processor
    from PIL import Image

    print(f"[GPU {args.gpu}] Loading SAM3...", flush=True)
    model = build_sam3_image_model()
    processor = Sam3Processor(model)
    print(f"[GPU {args.gpu}] SAM3 ready. Processing views: {view_indices}", flush=True)

    for vi in view_indices:
        cam = cameras[vi]
        name = cam["name"]
        img_path = cam["rgb_path"]
        print(f"[GPU {args.gpu}] {name}...", end=" ", flush=True)

        image = Image.open(img_path).convert("RGB")
        W, H = image.size
        state = processor.set_image(image)

        instance_mask = np.zeros((H, W), dtype=np.int32)
        score_mask = np.zeros((H, W), dtype=np.float32)
        label_map = {}
        next_id = 1

        for obj_name in object_names:
            processor.reset_all_prompts(state)
            try:
                output = processor.set_text_prompt(prompt=obj_name, state=state)
            except Exception:
                continue

            masks = output.get("masks")
            scores = output.get("scores")
            if masks is None or len(masks) == 0:
                continue

            for idx in range(len(masks)):
                score = float(scores[idx]) if scores is not None else 0.0
                if score < args.min_score:
                    continue
                m = masks[idx].cpu().numpy() if torch.is_tensor(masks[idx]) else np.array(masks[idx])
                if m.ndim == 3:
                    m = m[0]
                m_bool = m > 0.5
                effective_score = score - (
                    AMBIGUOUS_OVERWRITE_PENALTY
                    if obj_name in AMBIGUOUS_OVERWRITE_LABELS
                    else 0.0
                )
                overwrite = m_bool & (effective_score > score_mask)
                instance_mask[overwrite] = next_id
                score_mask[overwrite] = effective_score
                label_map[next_id] = obj_name
                next_id += 1

        np.save(os.path.join(args.out_dir, f"{name}_mask.npy"), instance_mask)
        json.dump({str(k): v for k, v in label_map.items()},
                  open(os.path.join(args.out_dir, f"{name}_labels.json"), "w"))

        n_labeled = (instance_mask > 0).sum()
        print(f"{len(label_map)} dets, {n_labeled} px ({n_labeled/instance_mask.size:.0%})", flush=True)

    print(f"[GPU {args.gpu}] Done.", flush=True)


if __name__ == "__main__":
    main()
