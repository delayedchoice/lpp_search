"""
GPU configuration module for PyMC with PyTorch MPS backend on Apple Silicon (M1/M2/M3).

Provides functions to detect/configure MPS (Metal Performance Shaders) GPU
or fall back to CPU. Set JAX_PLATFORMS environment variable to control.
"""

import os
import torch

def gpu_config_init(force_cpu=False):
    """
    Configure PyMC to use PyTorch MPS GPU OR CPU on demand.

    On Apple Silicon (M1/M2/M3 chips), PyMC can leverage PyTorch's MPS
    backend for GPU acceleration.

    Args:
        force_cpu: If True, force CPU usage regardless of GPU availability

    Returns:
        True if MPS GPU detected and available, False if CPU fallback
    """
    try:
        if force_cpu:
            print("Force CPU mode requested (CPU mode)")
            os.environ.setdefault('PYTORCH_MPS_DISABLE', '1')
            return False

        if not torch.backends.mps.is_available():
            print("MPS (Metal Performance Shaders) not available, using CPU")
            return False

        if not torch.backends.mps.is_built():
            print("MPS not built, using CPU")
            return False

        device_count = torch.mps.device_count()
        print(f"MPS GPU detected: {device_count} device(s)")

        try:
            device = torch.device("mps")
            test_tensor = torch.randn(100, 100).to(device)
            result = torch.matmul(test_tensor, test_tensor.T)
            print(f"MPS test successful on: {result.device}")
            return True
        except Exception as test_err:
            print(f"MPS test failed: {test_err}")
            print("Using CPU backend")
            return False

    except ImportError as ie:
        print(f"PyTorch not installed: {ie}")
        return False
    except Exception as e:
        print(f"MPS initialization failed: {e}")
        print("Using CPU backend")
        return False


# Module-level initialization
gpu_available = gpu_config_init(force_cpu=os.environ.get('FORCE_CPU', '').lower() in ('1', 'true', 'yes', 'cpu'))
