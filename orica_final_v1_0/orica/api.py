"""Stable production API for the final ORICA implementation.

The functions in this module are the intended integration boundary for the
larger EEG/neurofeedback system. Higher-level code should call these functions
rather than manipulate M/W/controller internals directly.
"""

from __future__ import annotations

from typing import Dict, Tuple

import numpy as np

from .controllers import CoolingController
from .core import ORICAConfig, OnlineORICA
from .diagnostics import state_health


ORICA_FINAL_VERSION = "ORICA_FINAL_V1.0"


def _select_backend(backend: str) -> str:
    if backend not in {"auto", "numpy", "cupy"}:
        raise ValueError("backend must be 'auto', 'numpy', or 'cupy'")

    if backend != "auto":
        return backend

    try:
        import cupy as cp  # type: ignore

        if cp.cuda.runtime.getDeviceCount() > 0:
            return "cupy"
    except Exception:
        pass

    return "numpy"


def create_final_orica(
    n_channels: int,
    sfreq: float,
    block_size: int = 8,
    backend: str = "auto",
    lambda0: float = 0.995,
    gamma: float = 0.60,
    dtype: str = "float64",
) -> Tuple[OnlineORICA, Dict[str, object]]:
    """Create the frozen production ORICA model.

    Production profile
    ------------------
    - Online RLS whitening
    - ORICA block update
    - all-super-Gaussian nonlinearity
    - symmetric orthogonalization after every block
    - CoolingController(lambda0=0.995, gamma=0.60) by default
    - NSI is diagnostic only
    """

    selected_backend = _select_backend(backend)

    block_dt = block_size / float(sfreq)
    delta_nsi = 0.05
    nsi_tau_s = -block_dt / np.log(1.0 - delta_nsi)

    config = ORICAConfig(
        n_channels=int(n_channels),
        sfreq=float(sfreq),
        block_size=int(block_size),
        backend=selected_backend,
        dtype=dtype,
        nsi_tau_s=float(nsi_tau_s),
        orthogonalize_every=1,
        diagnostics_every=0,
    )

    controller = CoolingController(lambda0=lambda0, gamma=gamma)
    model = OnlineORICA(config=config, controller=controller)

    info: Dict[str, object] = {
        "version": ORICA_FINAL_VERSION,
        "controller": "CoolingController",
        "n_channels": int(n_channels),
        "sfreq": float(sfreq),
        "block_size": int(block_size),
        "backend": model.backend,
        "dtype": dtype,
        "lambda0": float(lambda0),
        "gamma": float(gamma),
        "nsi_tau_s": float(nsi_tau_s),
    }

    return model, info


def process_orica_block(
    orica: OnlineORICA,
    X_block,
    return_diagnostics: bool = False,
    output_mode: str = "updated",
):
    """Process one block through the production ORICA stage.

    Parameters
    ----------
    orica:
        Stateful model returned by :func:`create_final_orica`.
    X_block:
        Array shaped ``(n_channels, n_samples)``. The validated real-EEG
        pipeline used microvolts after preprocessing/rank management.
    return_diagnostics:
        If True, also return a compact read-only numerical health dictionary.
    output_mode:
        ``"updated"`` (default): apply the newly updated total unmixing matrix
        to the current block before returning it.

        ``"learning"``: return the Y block that was actually used inside the
        recursive ICA update (computed with the previous W and updated M).

    Returns
    -------
    Y_block or (Y_block, diagnostics)
    """

    if output_mode not in {"updated", "learning"}:
        raise ValueError("output_mode must be 'updated' or 'learning'")

    result = orica.update(X_block, return_cpu=False)

    if output_mode == "learning":
        Y_backend = result["Y"]
        if orica.backend == "cupy":
            Y_block = orica.xp.asnumpy(Y_backend)
        else:
            Y_block = np.asarray(Y_backend).copy()
    else:
        Y_block = orica.transform(X_block, cpu=True)

    if not return_diagnostics:
        return Y_block

    state = orica.get_state(cpu=True)
    diagnostics = state_health(state)
    diagnostics.update(
        {
            "samples_seen": int(state["samples_seen"]),
            "blocks_seen": int(state["blocks_seen"]),
            "lambda_white": float(state["lambda_white"]),
            "nsi": float(state["nsi"]),
            "backend": state["backend"],
        }
    )

    return Y_block, diagnostics


def run_orica_array(
    X,
    sfreq: float,
    block_size: int = 8,
    backend: str = "auto",
    output_mode: str = "updated",
    keep_tail: bool = True,
):
    """Convenience runner for offline arrays using the same online API.

    This function simply feeds sequential non-overlapping blocks through the
    production stateful processor. It is useful for testing and replay.
    """

    X = np.asarray(X, dtype=np.float64)
    if X.ndim != 2:
        raise ValueError("X must have shape (channels, samples)")

    model, info = create_final_orica(
        n_channels=X.shape[0],
        sfreq=sfreq,
        block_size=block_size,
        backend=backend,
    )

    outputs = []
    N = X.shape[1]

    for start in range(0, N, block_size):
        stop = min(start + block_size, N)
        if stop - start < block_size and not keep_tail:
            break
        outputs.append(
            process_orica_block(
                model,
                X[:, start:stop],
                return_diagnostics=False,
                output_mode=output_mode,
            )
        )

    Y = np.concatenate(outputs, axis=1) if outputs else np.empty_like(X)
    return Y, model, info
