#!/usr/bin/env python3
"""
End-to-end benchmark pipeline: export → render → bbox → questions.

Must be run with Isaac Sim Python so the render step can use omni/replicator:
  ./app/python.sh run_pipeline.py --env sage --num-scenes 5 --question-type all
  ./app/python.sh run_pipeline.py --env scenesmith --num-scenes 3 --question-type all

Set HF_TOKEN or HUGGINGFACE_TOKEN env var for HuggingFace auth.
Render and export steps are skipped automatically if their outputs already exist.

Options:
  --env ENV              Environment: sage or scenesmith (default: sage)
  --num-scenes N         First N scenes to process (default: 1)
  --scenesmith-subset S  SceneSmith subset name (default: Room)
  --question-type TYPE   Question type or 'all'; omit to skip question generation.
                         See ALL_QUESTION_TYPES in src/tasks/__init__.py for valid values.
  --num-questions N      Questions per scene per type (default: 1)
  --output-root DIR      Root output directory (default: output)
  --force-rerun          Regenerate questions even if output already exists
  --force-rerun-all      Regenerate everything: pipeline + questions (slow)
  --use-vlm-filter       Run VLM-based view quality filter (requires VLM_API_KEY)
  --workers N            Parallel workers for QA generation (1 = sequential, 0 = auto/cpu_count)
"""

from __future__ import annotations

import argparse
import os
import sys

ROOT = os.path.dirname(os.path.abspath(__file__))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

# Load .env file if present (does not override existing env vars)
_env_path = os.path.join(ROOT, ".env")
if os.path.isfile(_env_path):
    with open(_env_path) as _f:
        for _line in _f:
            _line = _line.strip()
            if not _line or _line.startswith("#") or "=" not in _line:
                continue
            _k, _v = _line.split("=", 1)
            _k, _v = _k.strip(), _v.strip()
            if _k and _k not in os.environ:
                os.environ[_k] = _v

from src.tasks import ALL_QUESTION_TYPES


def _parse_args():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument(
        "--env",
        default="sage",
        choices=["sage", "scenesmith", "marble"],
        help="Environment to run (default: sage)",
    )
    ap.add_argument(
        "--num-scenes",
        "-N",
        type=int,
        default=1,
        metavar="N",
        help="Number of scenes to process (default: 1)",
    )
    ap.add_argument(
        "--scenesmith-subset",
        default="Room",
        metavar="SUBSET",
        help="SceneSmith HuggingFace subset name (default: Room)",
    )
    ap.add_argument(
        "--question-type",
        dest="question_type",
        default=None,
        choices=[*ALL_QUESTION_TYPES, "all"],
        help="Question type(s) to generate; omit to skip question generation",
    )
    ap.add_argument(
        "--num-questions",
        "-n",
        type=int,
        default=1,
        metavar="N",
        help="Questions per scene per type (default: 1)",
    )
    ap.add_argument(
        "--output-root",
        "-o",
        default="output",
        metavar="DIR",
        help="Root output directory (default: output)",
    )
    ap.add_argument(
        "--force-rerun",
        action="store_true",
        help="Regenerate questions even if output already exists",
    )
    ap.add_argument(
        "--force-rerun-all",
        action="store_true",
        help="Regenerate everything: pipeline + questions (bust all caches)",
    )
    ap.add_argument(
        "--use-vlm-filter",
        action="store_true",
        help="Run VLM-based view quality filter (requires VLM_API_KEY)",
    )
    ap.add_argument(
        "--seed",
        type=int,
        default=None,
        metavar="SEED",
        help="Random seed for scene selection (default: sequential)",
    )
    ap.add_argument(
        "--cleanup",
        action="store_true",
        help="Delete scene assets and intermediates after each scene to save disk",
    )
    ap.add_argument(
        "--workers",
        type=int,
        default=0,
        metavar="N",
        help="Parallel workers for question generation (0 = auto/cpu_count, 1 = sequential)",
    )
    ap.add_argument(
        "--image",
        default=None,
        metavar="PATH",
        help="Input image for Marble mode",
    )
    ap.add_argument(
        "--text-prompt",
        default=None,
        metavar="TEXT",
        help="Text prompt for Marble text-to-scene generation (no --image needed)",
    )
    ap.add_argument(
        "--input-list",
        default=None,
        metavar="FILE",
        help="JSON file with batch inputs: [{\"prompt\": ..., \"meta_info\": {\"image\": ...}}, ...]",
    )
    ap.add_argument(
        "--batch-workers",
        type=int,
        default=10,
        metavar="N",
        help="Concurrent API workers for batch mode Phase 1 (default: 10)",
    )
    ap.add_argument(
        "--generation-only",
        action="store_true",
        help="Stop after scene generation (skip QA question generation)",
    )
    ap.add_argument(
        "--scene-id",
        default=None,
        metavar="ID",
        help="Existing scene directory name in output-root (e.g. '0105_txt_2c5cea74cea2'). "
             "Use with --start-from to rerun from a specific step without --image/--text-prompt.",
    )
    ap.add_argument(
        "--start-from",
        default=None,
        metavar="STEP",
        help="Skip pipeline steps before STEP (e.g. '5d' to re-run label verification + QA). "
             "Valid steps: 1, 2, 3, 4, 4b, 5, 5b, 5c, 5d, 6, 7",
    )
    return ap.parse_known_args()[0]


def main():
    args = _parse_args()
    print("Config:")
    for k, v in vars(args).items():
        print(f"  {k}: {v}")
    if args.env == "sage":
        from src.scenes.sage.pipeline import run
    elif args.env == "scenesmith":
        from src.scenes.scenesmith.pipeline import run
    elif args.env == "marble":
        from src.scenes.marble.pipeline import run
    else:
        raise ValueError(f"Unknown env: {args.env!r}")

    input_list = getattr(args, "input_list", None)
    if input_list and args.env == "marble":
        import copy
        import json as _json
        import queue
        import threading
        from concurrent.futures import ThreadPoolExecutor

        with open(input_list) as f:
            entries = _json.load(f)

        # Single entry: run directly without batch mode (avoids recursion)
        if len(entries) == 1:
            entry = entries[0]
            args.text_prompt = entry.get("prompt")
            args.image = entry.get("image") or entry.get("image_pth")
            args._input_entry = entry
            args.input_list = None
            return run(args)

        api_workers = getattr(args, "batch_workers", 10) or 10
        try:
            import torch
            num_gpus = torch.cuda.device_count()
        except Exception:
            num_gpus = 1
        gpu_workers = min(num_gpus, len(entries))

        _skip_p1 = False
        _LATE_STEPS = {"3", "4", "4b", "5", "5b", "5c", "5d", "6", "7"}
        _start = getattr(args, "start_from", None)
        if _start and _start in _LATE_STEPS:
            _skip_p1 = True

        print(f"\nBatch mode: {len(entries)} entries")
        if _skip_p1:
            print(f"  P1 (API):  SKIPPED (--start-from {_start})")
        else:
            print(f"  P1 (API):  {api_workers} threads")
        print(f"  P2 (GPU):  {gpu_workers} threads ({num_gpus} GPUs)")

        def _make_args(entry):
            a = copy.deepcopy(args)
            a.text_prompt = entry.get("prompt")
            a.image = None
            a._input_entry = entry
            return a

        gpu_queue: queue.Queue = queue.Queue()
        rc_lock = threading.Lock()
        rc_total = [0]

        failed_entries: list[tuple[int, dict]] = []
        _failed_lock = threading.Lock()

        # P1 producer: API calls → push to GPU queue
        def _run_api(idx_entry):
            idx, entry = idx_entry
            a = _make_args(entry)
            a._marble_api_only = True
            try:
                rc = run(a)
            except Exception as e:
                print(f"  [P1 {idx}] FAILED: {e}")
                rc = 1
            status = "ok" if rc == 0 else "FAILED"
            print(f"  [P1] entry {idx} {status}", flush=True)
            if rc == 0:
                gpu_queue.put((idx, entry))
            else:
                with _failed_lock:
                    failed_entries.append((idx, entry))

        # P2 consumer: subprocess per GPU to avoid GIL/CUDA conflicts
        def _gpu_consumer(gpu_id):
            import subprocess
            import tempfile
            while True:
                try:
                    idx, entry = gpu_queue.get(timeout=30)
                except queue.Empty:
                    if api_done.is_set() and gpu_queue.empty():
                        break
                    continue
                env = {**os.environ, "CUDA_VISIBLE_DEVICES": str(gpu_id)}
                tmp = tempfile.NamedTemporaryFile(
                    mode="w", suffix=".json", delete=False, dir="/tmp")
                _json.dump([entry], tmp)
                tmp.close()
                cmd = [
                    sys.executable, os.path.join(ROOT, "run.py"),
                    "--env", "marble",
                    "--input-list", tmp.name,
                    "--output-root", os.path.abspath(args.output_root),
                    "--num-questions", str(getattr(args, "num_questions", 50)),
                ]
                qt = getattr(args, "question_type", None)
                if qt:
                    cmd.extend(["--question-type", qt])
                sf = getattr(args, "start_from", None)
                if sf:
                    cmd.extend(["--start-from", sf])
                if getattr(args, "generation_only", False):
                    cmd.append("--generation-only")
                max_gpu_retries = 3
                rc = 1
                for attempt in range(max_gpu_retries):
                    result = subprocess.run(cmd, env=env, cwd=os.getcwd())
                    rc = result.returncode
                    if rc == 0:
                        break
                    print(f"  [P2] entry {idx} (gpu{gpu_id}) "
                          f"retry {attempt + 1}/{max_gpu_retries}", flush=True)
                os.unlink(tmp.name)
                if rc != 0:
                    with rc_lock:
                        rc_total[0] = 1
                print(f"  [P2] entry {idx} (gpu{gpu_id}) "
                      f"{'ok' if rc == 0 else 'FAILED'}", flush=True)
                gpu_queue.task_done()

        api_done = threading.Event()

        if _skip_p1:
            # Queue all existing scene folders directly for P2
            print(f"\n{'='*60}\nRunning pipeline from step {_start} (P2 only)\n{'='*60}")
            _out_root = os.path.abspath(args.output_root)
            _prereq = "meta/scene.json"
            if _start in {"3", "4", "4b"}:
                _prereq = "meta/scene.ply"
            _all_entries = []
            if os.path.isdir(_out_root):
                for _d in sorted(os.listdir(_out_root)):
                    _pf = os.path.join(_out_root, _d, _prereq)
                    if os.path.isfile(_pf):
                        _ij = os.path.join(_out_root, _d, "input.json")
                        _entry = _json.load(open(_ij)) if os.path.isfile(_ij) else {}
                        _all_entries.append(_entry)
            print(f"  Found {len(_all_entries)} scenes (prereq: {_prereq})")

            # Run first scene on GPU 0 to warm up CUDA JIT caches
            if _all_entries:
                import subprocess as _sp
                import tempfile as _tf
                print("  Warming up CUDA (first scene on GPU 0)...")
                _wtmp = _tf.NamedTemporaryFile(
                    mode="w", suffix=".json", delete=False, dir="/tmp")
                _json.dump([_all_entries[0]], _wtmp)
                _wtmp.close()
                _wcmd = [
                    sys.executable, os.path.join(ROOT, "run.py"),
                    "--env", "marble",
                    "--input-list", _wtmp.name,
                    "--output-root", os.path.abspath(args.output_root),
                    "--num-questions", str(getattr(args, "num_questions", 50)),
                ]
                qt = getattr(args, "question_type", None)
                if qt:
                    _wcmd.extend(["--question-type", qt])
                sf = getattr(args, "start_from", None)
                if sf:
                    _wcmd.extend(["--start-from", sf])
                if getattr(args, "generation_only", False):
                    _wcmd.append("--generation-only")
                _wenv = {**os.environ, "CUDA_VISIBLE_DEVICES": "0"}
                _wr = _sp.run(_wcmd, env=_wenv, cwd=os.getcwd())
                os.unlink(_wtmp.name)
                if _wr.returncode == 0:
                    print("  Warmup done (scene 0 complete)")
                    _all_entries = _all_entries[1:]
                else:
                    print("  Warmup failed — continuing with all scenes")

            # Queue remaining scenes
            for _qi, _entry in enumerate(_all_entries):
                gpu_queue.put((_qi, _entry))
            print(f"  Queued {len(_all_entries)} remaining scenes for P2")
            api_done.set()

        # Start P2 consumers (1 thread per GPU, each spawns subprocess)
        gpu_threads = []
        for gid in range(gpu_workers):
            t = threading.Thread(target=_gpu_consumer, args=(gid,), daemon=True)
            t.start()
            gpu_threads.append(t)

        if not _skip_p1:
            # Run P1 producers
            print(f"\n{'='*60}\nRunning pipeline (P1→P2 streaming)\n{'='*60}")
            with ThreadPoolExecutor(max_workers=api_workers) as pool:
                list(pool.map(_run_api, enumerate(entries)))
            api_done.set()

        # Wait for GPU queue to drain
        gpu_queue.join()

        if not _skip_p1:
            # Auto-retry failed entries (up to 3 rounds)
            for retry_round in range(3):
                if not failed_entries:
                    break
                retry_list = list(failed_entries)
                failed_entries.clear()
                print(f"\n--- Retry round {retry_round + 1}: "
                      f"{len(retry_list)} entries ---", flush=True)
                import time as _time
                _time.sleep(30)
                for idx, entry in retry_list:
                    _run_api((idx, entry))
                gpu_queue.join()

            if failed_entries:
                print(f"\n{len(failed_entries)} entries still failed after retries",
                      flush=True)
                rc_total[0] = 1

        print(f"\nBatch complete: {_queued if _skip_p1 else len(entries)} entries")
        return rc_total[0]

    return run(args)


if __name__ == "__main__":
    sys.exit(main())
