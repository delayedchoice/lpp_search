# Fix for `ValueError: I/O operation on closed file.` in `DT_analysis`

## Issue Description
The user reported a `ValueError: I/O operation on closed file.` when running `src/scripts/02_run_quick_singles.py 0`.

## Root Cause
The root cause was located in `src/stages/search_singles.py` within the `DT_analysis` function. When the parameter `DT_Quite=True` was passed, the function was executing:
```python
sys.stdout.close()
sys.stderr.close()
```
Closing `sys.stdout` and `s_sys.stderr` prevents any subsequent code in the process (including the main script) from printing to the console. Any attempt to use `print()` after these calls resulted in the reported `ValueError`.

## Resolution
The `DT_analysis` function was refactored to remove the logic that closed the standard output and error streams. Additionally, several debug print statements that were cluttering the output were removed to ensure a clean execution.

The updated `DT_analysis` function in `src/stages/search_singles.py` now looks like this:

```python
def DT_analysis(time, flux, flux_err, confidence, DT_Quite=True, is_flat=True):
    whatever = make_LightKurveObject(time, flux, flux_err)
    model = dt.DeepTransit(whatever, is_flat=is_flat)
    bboxes = model.transit_detection(str(con.MODEL_PATH), confidence_threshold=confidence)
    return bboxes
```

## Verification
The fix was verified by running the following command:
```bash
.venv/bin/python src/scripts/02_run_quick_singles.py 0
```
The command completed successfully, producing the expected output and finding transit-like signals without triggering any `ValueError` related to closed file descriptors.
