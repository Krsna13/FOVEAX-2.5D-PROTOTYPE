"""Performance Profiler for FOVEAX Pipeline."""

import time
import json
from pathlib import Path
import statistics
from typing import Dict, List

from src.deployment.vram_budget import get_available_vram_mb, get_total_vram_mb

class Profiler:
    def __init__(self):
        self.metrics: Dict[str, List[float]] = {}
        self._start_times: Dict[str, float] = {}
        self._peak_vram_used = 0.0
        
        self.out_dir = Path("outputs/phase11/profiling")
        self.out_dir.mkdir(parents=True, exist_ok=True)
        
    def start(self, module_name: str):
        self._start_times[module_name] = time.perf_counter()
        
    def stop(self, module_name: str):
        if module_name not in self._start_times:
            return
        dt = time.perf_counter() - self._start_times[module_name]
        
        if module_name not in self.metrics:
            self.metrics[module_name] = []
        self.metrics[module_name].append(dt * 1000.0) # ms
        
        # Poll VRAM
        self._poll_vram()
        
    def _poll_vram(self):
        total = get_total_vram_mb()
        free = get_available_vram_mb()
        if total > 0 and free > 0:
            used = total - free
            if used > self._peak_vram_used:
                self._peak_vram_used = used
                
    def get_summary(self) -> Dict:
        summary = {}
        for mod, times in self.metrics.items():
            if not times:
                continue
            summary[mod] = {
                "mean_ms": statistics.mean(times),
                "min_ms": min(times),
                "max_ms": max(times),
                "std_ms": statistics.stdev(times) if len(times) > 1 else 0.0,
                "calls": len(times)
            }
        return summary
        
    def write_report(self):
        summary = self.get_summary()
        
        # Identify top 3 bottlenecks
        sorted_mods = sorted(summary.items(), key=lambda x: x[1]['mean_ms'], reverse=True)
        top_3 = sorted_mods[:3]
        
        # Write Summary
        with open(self.out_dir / "profiling_summary.txt", "w") as f:
            f.write("FOVEAX Phase 11 Profiling Summary\n")
            f.write("=================================\n\n")
            for mod, stats in sorted_mods:
                f.write(f"[{mod}]\n")
                f.write(f"  Mean: {stats['mean_ms']:.2f} ms\n")
                f.write(f"  Min : {stats['min_ms']:.2f} ms\n")
                f.write(f"  Max : {stats['max_ms']:.2f} ms\n")
                f.write(f"  Std : {stats['std_ms']:.2f} ms\n")
                f.write(f"  Runs: {stats['calls']}\n\n")
                
            f.write("Memory\n")
            f.write("======\n")
            total = get_total_vram_mb()
            if total > 0:
                f.write(f"  Peak VRAM Used: {self._peak_vram_used:.0f} MB ({(self._peak_vram_used/total)*100:.1f}% of {total:.0f} MB)\n")
            else:
                f.write("  Peak VRAM Used: N/A\n")
                
        # Write Bottlenecks
        with open(self.out_dir / "bottleneck_report.txt", "w") as f:
            f.write("Top 3 Bottlenecks\n")
            f.write("=================\n")
            for i, (mod, stats) in enumerate(top_3):
                f.write(f"{i+1}. {mod}: {stats['mean_ms']:.2f} ms\n")
                
        # Write JSONL
        with open(self.out_dir / "profiling_session.jsonl", "w") as f:
            f.write(json.dumps({"type": "summary", "data": summary, "peak_vram_mb": self._peak_vram_used}) + "\n")
