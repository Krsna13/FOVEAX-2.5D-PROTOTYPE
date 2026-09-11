import pytest
from unittest.mock import patch
from src.deployment.optimized_inference import OptimizedFOVEAXPipeline

@patch('src.deployment.optimized_inference.check_concurrent_fit')
def test_pipeline_refuses_to_load_on_oom(mock_check):
    # Simulate that engines do NOT fit in VRAM
    mock_check.return_value = False
    
    with pytest.raises(MemoryError) as excinfo:
        pipeline = OptimizedFOVEAXPipeline(
            semantic_engine_path="dummy.engine",
            detection_engine_path="dummy.engine"
        )
        
    assert "Refusing to start to prevent OOM" in str(excinfo.value)

@patch('src.deployment.optimized_inference.check_concurrent_fit')
def test_pipeline_loads_when_fits(mock_check):
    # Simulate that engines DO fit in VRAM
    mock_check.return_value = True
    
    pipeline = OptimizedFOVEAXPipeline(
        semantic_engine_path="dummy.engine",
        detection_engine_path="dummy.engine"
    )
    
    assert pipeline.semantic_engine == "LOADED_SEMANTIC_TRT"
    assert pipeline.detection_engine == "LOADED_DETECTION_TRT"
