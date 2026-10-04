"""Diagnostics for ORICA state and output quality.

These functions are intentionally read-only. They never alter the ORICA state
and are not used by the production Cooling controller.
"""

from __future__ import annotations

from typing import Dict

import numpy as np


def orthogonality_error(W) -> float:
    """Maximum absolute error in ``W W.T - I``."""
    W = np.asarray(W, dtype=np.float64)
    I = np.eye(W.shape[0])
    return float(np.max(np.abs(W @ W.T - I)))


def normalized_nsi(Rn) -> float:
    """Return ``||Rn||_F / sqrt(D)``; NaN when Rn is unavailable."""
    if Rn is None:
        return float("nan")
    Rn = np.asarray(Rn, dtype=np.float64)
    D = Rn.shape[0]
    return float(np.linalg.norm(Rn, ord="fro") / np.sqrt(D))


def whitening_metrics(M, X_window) -> Dict[str, float]:
    """Evaluate covariance of ``V = M X`` against the identity matrix."""
    M = np.asarray(M, dtype=np.float64)
    X = np.asarray(X_window, dtype=np.float64)
    V = M @ X
    C = (V @ V.T) / V.shape[1]
    I = np.eye(C.shape[0])

    rel_error = np.linalg.norm(C - I, ord="fro") / np.linalg.norm(I, ord="fro")
    mean_diag = np.mean(np.diag(C))

    off = C - np.diag(np.diag(C))
    D = C.shape[0]
    mean_abs_offdiag = np.sum(np.abs(off)) / max(D * (D - 1), 1)

    return {
        "whitening_rel_error": float(rel_error),
        "whitening_mean_diag": float(mean_diag),
        "whitening_mean_abs_offdiag": float(mean_abs_offdiag),
    }


def energy_dependence(Y) -> float:
    """RMS off-diagonal correlation of squared, standardized outputs.

    This is a diagnostic only; it is not a convergence criterion and is not
    used to control the production algorithm.
    """
    Y = np.asarray(Y, dtype=np.float64)
    Y = Y - np.mean(Y, axis=1, keepdims=True)
    Y = Y / (np.std(Y, axis=1, keepdims=True) + 1e-12)
    E = Y**2
    C = np.corrcoef(E)
    off = C - np.eye(C.shape[0])
    D = C.shape[0]
    return float(np.sqrt(np.sum(off**2) / max(D * (D - 1), 1)))


def state_health(state) -> Dict[str, float | bool]:
    """Compact numerical health report from ``OnlineORICA.get_state()``."""
    M = np.asarray(state["M"], dtype=np.float64)
    W = np.asarray(state["W"], dtype=np.float64)
    Rn = state.get("Rn")

    return {
        "all_finite": bool(
            np.all(np.isfinite(M))
            and np.all(np.isfinite(W))
            and (Rn is None or np.all(np.isfinite(Rn)))
        ),
        "orthogonality_error": orthogonality_error(W),
        "cond_M": float(np.linalg.cond(M)),
        "cond_W": float(np.linalg.cond(W)),
        "nsi_normalized": normalized_nsi(Rn),
    }
