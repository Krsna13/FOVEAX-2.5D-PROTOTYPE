import pytest
from src.deployment.profiler import Profiler

def test_profiler_timing():
    profiler = Profiler()
    profiler.start("test_module")
    import time
    time.sleep(0.05)
    profiler.stop("test_module")
    
    summary = profiler.get_summary()
    assert "test_module" in summary
    assert summary["test_module"]["calls"] == 1
    assert 40.0 <= summary["test_module"]["mean_ms"] <= 100.0
