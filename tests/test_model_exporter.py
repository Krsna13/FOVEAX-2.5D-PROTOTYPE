import pytest
from src.deployment.model_exporter import export_pytorch_to_onnx

def test_model_exporter_missing_model():
    with pytest.raises(ValueError) as excinfo:
        export_pytorch_to_onnx(None, None, "test.onnx", ["input"], ["output"])
    
    # Assert informative message
    assert "PyTorch model is None" in str(excinfo.value)
    assert "SalsaNext and PointPillars are external repositories" in str(excinfo.value)


def test_model_exporter_handles_missing_onnx_deps():
    import torch
    import torch.nn as nn

    model = nn.Linear(4, 2)
    dummy = torch.randn(1, 4)

    with pytest.raises(RuntimeError) as excinfo:
        export_pytorch_to_onnx(model, dummy, "outputs/phase11/onnx/dummy.onnx", ["in"], ["out"])

    assert "ONNX export requires additional dependencies" in str(excinfo.value)

