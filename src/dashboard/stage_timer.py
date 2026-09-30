"""Lightweight per-stage wall-clock timing for the dashboard pipeline.

Off by default: every call site takes an optional `stage_timer` that is None
unless `--profile` is passed, and `StageTimer.time()` on a None/disabled
timer is a no-op -- so instrumented code paths cost nothing and behave
identically when profiling is off. Nothing here alters any pipeline value.

GPU stages: CUDA kernel launches are asynchronous, so a bare
time.perf_counter() around a GPU call measures launch latency, not
execution. Stages named in `cuda_stages` therefore call
torch.cuda.synchronize() immediately before AND after the timed region, so
the measured interval covers exactly the work inside it and none queued
before it.
"""

from __future__ import annotations

import time
from collections import OrderedDict
from contextlib import contextmanager

import numpy as np


class StageTimer:
    def __init__(self, enabled: bool = True, cuda_stages: frozenset[str] = frozenset()):
        self.enabled = enabled
        self.cuda_stages = cuda_stages
        self._samples: "OrderedDict[str, list[float]]" = OrderedDict()
        self._not_applicable: dict[str, str] = {}
        self._recording = True

    def set_recording(self, on: bool) -> None:
        """Warm-up frames run the same code but their samples are dropped."""
        self._recording = on

    def mark_not_applicable(self, name: str, reason: str) -> None:
        self._not_applicable[name] = reason

    @contextmanager
    def time(self, name: str):
        if not self.enabled:
            yield
            return

        sync = name in self.cuda_stages
        if sync:
            import torch

            torch.cuda.synchronize()
        t0 = time.perf_counter()
        try:
            yield
        finally:
            if sync:
                torch.cuda.synchronize()
            dt_ms = (time.perf_counter() - t0) * 1000.0
            if self._recording:
                self._samples.setdefault(name, []).append(dt_ms)

    def samples(self, name: str) -> list[float]:
        return list(self._samples.get(name, []))

    def report(self, stage_order: list[str], total_name: str = "frame_total") -> str:
        """Raw per-stage table: n, mean, p50, p95 (ms) and % of mean frame time.

        `total_name` is a stage recorded by the caller around the whole
        frame. mean/p50/p95 are per occurrence; "% of frame" is amortized
        (sum of the stage's samples / n_frames / mean frame time).
        """
        total_samples = self._samples.get(total_name, [])
        total_mean = float(np.mean(total_samples)) if total_samples else float("nan")

        lines = []
        header = (
            f"{'stage':<28}{'n':>6}{'mean_ms':>11}{'p50_ms':>11}{'p95_ms':>11}"
            f"{'% of frame':>12}"
        )
        lines.append(header)
        lines.append("-" * len(header))
        for name in stage_order:
            s = self._samples.get(name)
            if not s:
                reason = self._not_applicable.get(name, "no samples recorded")
                lines.append(f"{name:<28}{'N/A':>6}   ({reason})")
                continue
            arr = np.asarray(s, dtype=np.float64)
            # Amortized share of mean frame time: sum(samples) / n_frames.
            # For a stage timed every frame this equals mean/total_mean; for a
            # stage that only runs on some frames (n < n_frames) it is that
            # stage's real contribution per frame, not its per-occurrence cost.
            pct = (
                100.0 * float(arr.sum()) / len(total_samples) / total_mean
                if total_samples
                else float("nan")
            )
            lines.append(
                f"{name:<28}{len(arr):>6}{arr.mean():>11.3f}{np.percentile(arr, 50):>11.3f}"
                f"{np.percentile(arr, 95):>11.3f}{pct:>11.1f}%"
            )
        if total_samples:
            arr = np.asarray(total_samples, dtype=np.float64)
            lines.append("-" * len(header))
            lines.append(
                f"{total_name:<28}{len(arr):>6}{arr.mean():>11.3f}{np.percentile(arr, 50):>11.3f}"
                f"{np.percentile(arr, 95):>11.3f}{100.0:>11.1f}%"
            )
        return "\n".join(lines)
