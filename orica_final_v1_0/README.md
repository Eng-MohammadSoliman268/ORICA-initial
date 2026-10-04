# ORICA_FINAL_V1.0

Modular production implementation of the validated EEG Online Recursive ICA pipeline.

## Production decision

The production profile uses **Cooling ORICA**:

- online block RLS whitening
- block ORICA update
- all-super-Gaussian nonlinearity `f(y) = -2*tanh(y)`
- symmetric orthogonalization after every block
- `CoolingController(lambda0=0.995, gamma=0.60)`
- nominal block size `8`
- `float64`
- NumPy or optional CuPy backend
- NSI retained for diagnostics only

`GeneralSelfTuningControllerV43` is included under `orica/experimental/` for research only. It is **not** selected by the production factory.

## Project layout

```text
orica/
├── __init__.py
├── core.py
├── controllers.py
├── api.py
├── diagnostics.py
├── experimental/
│   ├── __init__.py
│   └── adaptive_v43.py
└── tests/
    ├── test_core.py
    └── test_production_api.py

orica_full_algorithm.py
system_integration_example.py
```

## Production use

```python
from orica.api import create_final_orica, process_orica_block

orica, info = create_final_orica(
    n_channels=8,
    sfreq=500.0,
    block_size=8,
    backend="auto",
)

# X_block shape = (8, 8)
Y_block = process_orica_block(orica, X_block)
```

The ORICA stage expects the upstream pipeline to provide a consistent, preprocessed, full-rank block. Filtering, ring buffering, channel/rank management, artifact-IC decisions, reconstruction, and neurofeedback remain separate stages.

## Offline replay

```python
from orica.api import run_orica_array

Y, model, info = run_orica_array(
    X,
    sfreq=500.0,
    block_size=8,
    backend="auto",
)
```

## Tests

```bash
python -m pip install numpy pytest
pytest -q orica/tests
```

CuPy is optional and should match the CUDA version installed on the target system.

## Versioning

Production API version: `ORICA_FINAL_V1.0`

The production integration boundary is intentionally small:

- `create_final_orica(...)`
- `process_orica_block(...)`
- `run_orica_array(...)`

This allows later development of preprocessing, artifact detection, reconstruction, or experimental controllers without modifying the validated core interface.
