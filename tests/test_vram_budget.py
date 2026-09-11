import pytest
from unittest.mock import patch
from src.deployment.vram_budget import check_concurrent_fit

@patch('src.deployment.vram_budget.get_available_vram_mb')
@patch('src.deployment.vram_budget.get_total_vram_mb')
@patch('src.deployment.vram_budget.estimate_engine_vram_mb')
def test_concurrent_fit_fails_on_8gb(mock_estimate, mock_total, mock_free):
    # Simulate an 8GB card with 6000MB free (OS taking 2GB)
    mock_total.return_value = 8192.0
    mock_free.return_value = 6000.0
    
    # Simulate two heavy engines that need 3.5GB each (7GB total)
    mock_estimate.return_value = 3500.0
    
    # 7GB + 20% safety margin of 8GB (1.6GB) = 8.6GB required.
    # 8.6GB > 6.0GB free, should fail.
    assert check_concurrent_fit(["engine1", "engine2"], safety_margin_pct=20.0) is False

@patch('src.deployment.vram_budget.get_available_vram_mb')
@patch('src.deployment.vram_budget.get_total_vram_mb')
@patch('src.deployment.vram_budget.estimate_engine_vram_mb')
def test_concurrent_fit_passes(mock_estimate, mock_total, mock_free):
    mock_total.return_value = 8192.0
    mock_free.return_value = 6000.0
    
    # Simulate lighter engines (1.5GB each)
    mock_estimate.return_value = 1500.0
    
    # 3.0GB + 1.6GB margin = 4.6GB. 
    # 4.6GB <= 6.0GB free, should pass.
    assert check_concurrent_fit(["engine1", "engine2"], safety_margin_pct=20.0) is True
