"""Tests for the OpenPCDetPointPillarsDetector adapter."""

import pytest
import numpy as np
from pathlib import Path
from unittest.mock import patch, MagicMock

from src.perception.openpcdet_predictor import OpenPCDetPointPillarsDetector
from src.perception.object_detector import Detection3D

def test_missing_repo_raises_file_not_found(tmp_path):
    repo_path = tmp_path / "missing_repo"
    checkpoint_path = tmp_path / "model.pth"
    config_path = tmp_path / "config.yaml"
    checkpoint_path.touch()
    config_path.touch()

    with pytest.raises(FileNotFoundError, match="OpenPCDet repository not found"):
        OpenPCDetPointPillarsDetector(
            repo_path=repo_path,
            checkpoint_path=checkpoint_path,
            config_path=config_path,
        )

def test_missing_checkpoint_raises_file_not_found(tmp_path):
    repo_path = tmp_path / "repo"
    repo_path.mkdir()
    checkpoint_path = tmp_path / "missing_model.pth"
    config_path = tmp_path / "config.yaml"
    config_path.touch()

    with pytest.raises(FileNotFoundError, match="Checkpoint not found"):
        OpenPCDetPointPillarsDetector(
            repo_path=repo_path,
            checkpoint_path=checkpoint_path,
            config_path=config_path,
        )

def test_missing_config_raises_file_not_found(tmp_path):
    repo_path = tmp_path / "repo"
    repo_path.mkdir()
    checkpoint_path = tmp_path / "model.pth"
    checkpoint_path.touch()
    config_path = tmp_path / "missing_config.yaml"

    with pytest.raises(FileNotFoundError, match="Config not found"):
        OpenPCDetPointPillarsDetector(
            repo_path=repo_path,
            checkpoint_path=checkpoint_path,
            config_path=config_path,
        )

@patch("src.perception.openpcdet_predictor.sys.path", [])
@patch("builtins.__import__")
def test_device_resolver_cpu(mock_import, tmp_path):
    # Mocking torch and pcdet
    mock_torch = MagicMock()
    mock_torch.cuda.is_available.return_value = False
    mock_device_obj = MagicMock()
    mock_device_obj.type = "cpu"
    mock_torch.device.return_value = mock_device_obj
    
    mock_pcdet_config = MagicMock()
    mock_pcdet_models = MagicMock()
    mock_pcdet_datasets = MagicMock()

    # Create dummy files
    repo_path = tmp_path / "repo"
    repo_path.mkdir()
    checkpoint_path = tmp_path / "model.pth"
    checkpoint_path.touch()
    config_path = tmp_path / "config.yaml"
    config_path.touch()

    def side_effect(name, *args, **kwargs):
        if name == "torch": return mock_torch
        if name == "pcdet.config": return mock_pcdet_config
        if name == "pcdet.models": return mock_pcdet_models
        if name == "pcdet.datasets": return mock_pcdet_datasets
        return MagicMock()

    mock_import.side_effect = side_effect

    with patch.dict("sys.modules", {
        "torch": mock_torch, 
        "pcdet.config": mock_pcdet_config,
        "pcdet.models": mock_pcdet_models,
        "pcdet.datasets": mock_pcdet_datasets
    }):
        with pytest.warns(UserWarning, match="Inference running on CPU"):
            detector = OpenPCDetPointPillarsDetector(
                repo_path=repo_path,
                checkpoint_path=checkpoint_path,
                config_path=config_path,
                device="auto"
            )
        assert detector.device == mock_device_obj
        mock_torch.cuda.is_available.assert_called()

def test_deterministic_sort():
    detections = [
        Detection3D(np.array([1.0, 1.0, 0.0], dtype=np.float32), np.array([2.0, 2.0, 2.0], dtype=np.float32), 0.0, 1, "Car", 0.5, "test"),
        Detection3D(np.array([2.0, 1.0, 0.0], dtype=np.float32), np.array([2.0, 2.0, 2.0], dtype=np.float32), 0.0, 1, "Car", 0.9, "test"),
        Detection3D(np.array([0.0, 1.0, 0.0], dtype=np.float32), np.array([2.0, 2.0, 2.0], dtype=np.float32), 0.0, 1, "Car", 0.9, "test"),
    ]
    detections.sort(key=lambda d: (-d.confidence, d.class_name, d.center_xyz[0], d.center_xyz[1]))
    
    assert detections[0].confidence == 0.9
    assert detections[0].center_xyz[0] == 0.0
    assert detections[1].confidence == 0.9
    assert detections[1].center_xyz[0] == 2.0
    assert detections[2].confidence == 0.5
