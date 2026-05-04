# Switching Between CPU and MPS GPU in LPP Search

## Quick Commands

### Enable MPS GPU (Default)
```bash
python your_script.py
```

### Force CPU Only
```bash
FORCE_CPU=1 python your_script.py
```

---

## Method 1: Environment Variable

**To use MPS GPU (default):**
```bash
python your_script.py
# Or explicitly:
FORCE_CPU=0 python your_script.py
```

**To force CPU:**
```bash
FORCE_CPU=1 python your_script.py
```

---

## Method 2: Code Control

```python
from src.utils.gpu_config import gpu_config_init

# Force CPU
result = gpu_config_init(force_cpu=True)
print(f"GPU mode: {result}")  # Will print False

# Use MPS (if available)
result = gpu_config_init(force_cpu=False)
print(f"GPU mode: {result}")  # Will print True if GPU available
```

---

## Method 3: PyTorch Backend Control

```python
import torch

# Check current status
print(f"MPS available: {torch.backends.mps.is_available()}")
print(f"MPS enabled: {torch.backends.mps.is_available()}")

# Force CPU
torch.backends.mps.set_enabled(False)
device = torch.device("cpu")

# Force GPU (if available)
torch.backends.mps.set_enabled(True)
device = torch.device("mps")

# Test device
test = torch.randn(10, 10).to(device)
print(f"Tensor on: {test.device}")
```

---

## Verify GPU/CPU Usage

```bash
# Check MPS availability
python -c "import torch; print('MPS:', torch.backends.mps.is_available(), 'devices:', torch.mps.device_count())"

# Check config result
python -c "from src.utils.gpu_config import gpu_config_init; gpu_config_init()"

# Force CPU check
FORCE_CPU=1 python -c "from src.utils.gpu_config import gpu_config_init; gpu_config_init()"
```

---

## Test in Your Project

```python
from src.utils.gpu_config import gpu_config_init, gpu_available

print(f"GPU available at import: {gpu_available}")

# Test GPU is working
from src.engines.pyMC_core import sample_until_converged
import pymc as pm

with pm.Model() as model:
    x = pm.Normal('x', mu=0, sigma=1)
    trace, attempt = sample_until_converged(model, max_attempts=1)
    print(f"Sampled: {len(trace.posterior.x)}")
```

---

## Expected Output

**GPU Mode (default):**
```
MPS GPU detected: 1 device(s)
MPS test successful on: mps:0
```

**CPU Mode:**
```
Force CPU mode requested (CPU mode)
Using CPU backend for stability
```

---

## Summary

| Mode          | Environment      | Code Parameter        |
|---------------|------------------|-----------------------|
| GPU (default) | `FORCE_CPU=0` or unset | `force_cpu=False`    |
| CPU           | `FORCE_CPU=1`    | `force_cpu=True`      |

Your code uses `nuts_sampler="pymc"` which automatically uses MPS when available.
