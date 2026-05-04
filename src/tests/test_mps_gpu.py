"""Test that MPS/GPU works correctly."""
import torch

def test_mps_available():
    """Check MPS is available on Apple Silicon."""
    assert torch.backends.mps.is_available(), "MPS should be available on M-series chips"

def test_mps_device_count():
    """Check MPS device count."""
    assert torch.mps.device_count() >= 1, "Should have at least 1 MPS device"

def test_mps_device_type():
    """Check MPS device type."""
    assert torch.device("mps").type == "mps", "Device type should be mps"

def test_mps_computation():
    """Test basic MPS computation."""
    if torch.backends.mps.is_available():
        t = torch.randn(100).to("mps")
        result = t + 1
        assert result.device.type == "mps", "Computation should use MPS"
