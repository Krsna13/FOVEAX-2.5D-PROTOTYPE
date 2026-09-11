import pytest
from src.deployment.model_exporter import export_pytorch_to_onnx

def test_model_exporter_missing_model():
    with pytest.raises(ValueError) as excinfo:
        export_pytorch_to_onnx(None, None, "test.onnx", ["input"], ["output"])
    
    # Assert informative message
    assert "PyTorch model is None" in str(excinfo.value)
    assert "SalsaNext and PointPillars are external repositories" in str(excinfo.value)
