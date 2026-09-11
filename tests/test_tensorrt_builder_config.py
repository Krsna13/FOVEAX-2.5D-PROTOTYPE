import pytest
import os
from unittest.mock import patch
from src.deployment.tensorrt_builder import build_tensorrt_engine

def test_builder_missing_onnx():
    with pytest.raises(FileNotFoundError):
        build_tensorrt_engine("non_existent_model.onnx", "out.engine")

@patch('os.path.exists')
def test_builder_fallback_to_trtexec(mock_exists):
    mock_exists.return_value = True
    
    # If TRT is not installed, it should raise ImportError with trtexec instructions
    # We simulate trt missing by patching import (if it's not actually installed locally).
    with patch.dict('sys.modules', {'tensorrt': None}):
        with pytest.raises(ImportError) as excinfo:
            build_tensorrt_engine("fake.onnx", "fake.engine")
        
        assert "trtexec" in str(excinfo.value)
