"""Parallel task dispatch with process pool, output capture, and ordered results."""

from __future__ import annotations

import contextlib
import io
import os
import sys
import time
from typing import Callable

_worker_ctx: dict = {}
_worker_fn: Callable | None = None


def _init(ctx: dict, fn: Callable) -> None:
    """Process-pool initializer: set shared context + task function once per worker."""
    global _worker_ctx, _worker_fn
    _worker_ctx = ctx
    _worker_fn = fn


def _run(key: str) -> tuple[bool, str, float]:
    """Execute the task function with captured stdout/stderr and wall-clock timing."""
    buf = io.StringIO()
    t0 = time.monotonic()
    with contextlib.redirect_stdout(buf), contextlib.redirect_stderr(buf):
        try:
            ok = _worker_fn(key, _worker_ctx)  # type: ignore[misc]
        except Exception as exc:
            print(f"  [error] {key}: {exc}")
            ok = False
    return ok, buf.getvalue(), time.monotonic() - t0


def run_parallel(
    fn: Callable[[str, dict], bool],
    keys: list[str],
    ctx: dict,
    *,
    workers: int = 0,
) -> bool:
    """Run ``fn(key, ctx)`` for each key, optionally across processes.

    * Each invocation's stdout is captured and replayed in *keys* order.
    * Context is sent once per worker via the pool initializer (not per task).

    Args:
        fn:      ``fn(key, ctx) -> bool``.  Must be a picklable (module-level)
                 function for ``workers > 1``.
        keys:    Items to process.
        ctx:     Read-only context dict shared across workers.
        workers: 0 = auto (``cpu_count``), 1 = sequential, N = explicit.

    Returns:
        True if every task succeeded.
    """
    n = len(keys)
    if n == 0:
        return True

    n_workers = workers if workers > 0 else min(n, os.cpu_count() or 1)
    t0 = time.monotonic()

    if n_workers <= 1:
        _init(ctx, fn)
        results = {k: _run(k) for k in keys}
    else:
        from concurrent.futures import ProcessPoolExecutor, as_completed

        with ProcessPoolExecutor(
            max_workers=n_workers,
            initializer=_init,
            initargs=(ctx, fn),
        ) as pool:
            futs = {pool.submit(_run, k): k for k in keys}
            results: dict[str, tuple[bool, str, float]] = {}
            for fut in as_completed(futs):
                k = futs[fut]
                try:
                    results[k] = fut.result()
                except Exception as exc:
                    results[k] = (False, "", 0.0)
                    print(f"  [error] {k}: {exc}", file=sys.stderr)

    ok = True
    for k in keys:
        success, log, _ = results[k]
        if log:
            sys.stdout.write(log)
        if not success:
            ok = False

    elapsed = time.monotonic() - t0
    n_ok = sum(1 for s, _, _ in results.values() if s)
    print(
        f"\n  {n_ok}/{n} tasks | {elapsed:.1f}s wall "
        f"({n_workers} worker{'s' if n_workers != 1 else ''})"
    )
    return ok
