"""Core Online Recursive ICA implementation.

This module contains the numerically validated ORICA core used by the project.
The production controller is intentionally kept outside this file so the core
can remain frozen while controller strategies are developed independently.

Model
-----
    B_n = W_n M_n
    v_n = M_n x_n
    y_n = W_n v_n

The whitening stage follows the block RLS update used by the reference ORICA
implementation. The ICA stage uses the block recursive update with the
super-Gaussian nonlinearity f(y) = -2*tanh(y), followed by symmetric
orthogonalization.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional

import numpy as np


@dataclass
class ORICAConfig:
    """Configuration for :class:`OnlineORICA`.

    Parameters
    ----------
    n_channels:
        Number of full-rank input dimensions.
    sfreq:
        Sampling frequency in Hz.
    block_size:
        Nominal processing block size. The core also accepts shorter blocks,
        which is useful for the final tail of an offline array.
    backend:
        ``"numpy"``, ``"cupy"`` or ``"auto"``.
    dtype:
        Floating-point dtype. Production validation used ``float64``.
    nsi_tau_s:
        Time constant for the leaky model-fitness / NSI state. If ``None``,
        a per-block delta of 0.05 is used.
    orthogonalize_every:
        Orthogonalize W every N blocks. Production validation used 1.
    diagnostics_every:
        Reserved for higher-level monitoring. The core itself does not print.
    eigenvalue_floor:
        Numerical floor used only inside symmetric orthogonalization.
    """

    n_channels: int
    sfreq: float
    block_size: int = 8
    backend: str = "auto"
    dtype: str = "float64"
    nsi_tau_s: Optional[float] = None
    orthogonalize_every: int = 1
    diagnostics_every: int = 0
    eigenvalue_floor: float = 1e-15

    def __post_init__(self) -> None:
        if self.n_channels < 1:
            raise ValueError("n_channels must be >= 1")
        if self.sfreq <= 0:
            raise ValueError("sfreq must be > 0")
        if self.block_size < 1:
            raise ValueError("block_size must be >= 1")
        if self.backend not in {"auto", "numpy", "cupy"}:
            raise ValueError("backend must be 'auto', 'numpy', or 'cupy'")
        if self.orthogonalize_every < 1:
            raise ValueError("orthogonalize_every must be >= 1")
        if self.nsi_tau_s is not None and self.nsi_tau_s <= 0:
            raise ValueError("nsi_tau_s must be > 0 when provided")


class OnlineORICA:
    """Stateful online RLS whitening + ORICA processor.

    Notes
    -----
    The core expects *full-rank*, approximately zero-mean input. Any channel
    selection, filtering, centering/rank management, or artifact rejection is
    deliberately kept outside this class so those stages can be developed
    independently.
    """

    def __init__(self, config: ORICAConfig, controller: Any):
        self.config = config
        self.controller = controller
        self.xp, self.backend = self._resolve_backend(config.backend)
        self.dtype = getattr(self.xp, config.dtype)
        self.reset()

    # ------------------------------------------------------------------
    # backend helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _resolve_backend(name: str):
        if name == "numpy":
            return np, "numpy"

        if name in {"auto", "cupy"}:
            try:
                import cupy as cp  # type: ignore

                if cp.cuda.runtime.getDeviceCount() > 0:
                    return cp, "cupy"
            except Exception:
                if name == "cupy":
                    raise RuntimeError(
                        "CuPy backend requested but no usable CUDA/CuPy runtime was found."
                    )

        return np, "numpy"

    def _to_cpu(self, value):
        if self.backend == "cupy":
            return self.xp.asnumpy(value)
        return np.asarray(value)

    def _copy_out(self, value, cpu: bool):
        if cpu:
            return self._to_cpu(value).copy()
        return value.copy()

    # ------------------------------------------------------------------
    # state
    # ------------------------------------------------------------------
    def reset(self) -> "OnlineORICA":
        D = self.config.n_channels
        xp = self.xp

        self.M = xp.eye(D, dtype=self.dtype)
        self.W = xp.eye(D, dtype=self.dtype)
        self.Rn = None

        self.samples_seen = 0
        self.blocks_seen = 0
        self.last_lambda_white = np.nan
        self.last_lambda_ica = None
        self.last_nsi = np.nan
        self.last_nsi_normalized = np.nan

        if hasattr(self.controller, "reset"):
            self.controller.reset()

        return self

    @property
    def nsi_delta(self) -> float:
        if self.config.nsi_tau_s is None:
            return 0.05

        block_dt = self.config.block_size / self.config.sfreq
        return float(1.0 - np.exp(-block_dt / self.config.nsi_tau_s))

    # ------------------------------------------------------------------
    # numerical kernels
    # ------------------------------------------------------------------
    def _symmetric_orthogonalize(self, W):
        xp = self.xp
        gram = W @ W.T
        eigvals, eigvecs = xp.linalg.eigh(gram)
        eigvals = xp.maximum(eigvals, self.config.eigenvalue_floor)
        inv_sqrt = (eigvecs * (1.0 / xp.sqrt(eigvals))[None, :]) @ eigvecs.T
        return inv_sqrt @ W

    def _validate_input(self, X):
        xp = self.xp
        X = xp.asarray(X, dtype=self.dtype)

        if X.ndim != 2:
            raise ValueError("X_block must have shape (channels, samples)")
        if X.shape[0] != self.config.n_channels:
            raise ValueError(
                f"Expected {self.config.n_channels} channels, got {X.shape[0]}"
            )
        if X.shape[1] < 1:
            raise ValueError("X_block must contain at least one sample")

        if not bool(self._to_cpu(xp.all(xp.isfinite(X)))):
            raise ValueError("X_block contains NaN or Inf")

        return X

    # ------------------------------------------------------------------
    # main online update
    # ------------------------------------------------------------------
    def update(self, X_block, return_cpu: bool = False) -> Dict[str, Any]:
        """Process one block and update the internal ORICA state.

        Parameters
        ----------
        X_block:
            Array shaped ``(n_channels, n_samples)``.
        return_cpu:
            When using CuPy, copy returned arrays to CPU if True.

        Returns
        -------
        dict
            Contains the whitened block ``V``, the source block ``Y`` used by
            the ICA update, whitening/ICA forgetting factors, and NSI values.
            Adding these outputs does not alter the validated state trajectory.
        """

        xp = self.xp
        X = self._validate_input(X_block)
        D, L = X.shape

        # Positive, one-based sample count. This matches the validated cooling
        # schedule lambda(n) = lambda0 / n**gamma.
        sample_indices = xp.arange(
            self.samples_seen + 1,
            self.samples_seen + L + 1,
            dtype=self.dtype,
        )

        # ================================================================
        # 1) Online RLS whitening
        # ================================================================
        lambda_white = self.controller.whitening_lambda(sample_indices, xp)
        lambda_white = xp.asarray(lambda_white, dtype=self.dtype).reshape(())

        lambda_white_cpu = float(self._to_cpu(lambda_white))
        if not np.isfinite(lambda_white_cpu) or not (0.0 < lambda_white_cpu < 1.0):
            raise ValueError(f"Invalid whitening forgetting factor: {lambda_white_cpu}")

        # v_old is formed with the previous whitening matrix, exactly as in
        # the reference block update.
        v_old = self.M @ X
        retention = 1.0 - lambda_white

        # Q_white = retention/lambda + mean(||v||^2)
        q_white = retention / lambda_white + xp.trace(v_old.T @ v_old) / L

        self.M = (1.0 / retention) * (
            self.M - ((v_old @ v_old.T) / L / q_white) @ self.M
        )

        # Current block is then whitened by the updated M.
        V = self.M @ X

        # ================================================================
        # 2) ORICA source activation and model fitness
        # ================================================================
        Y = self.W @ V
        F = -2.0 * xp.tanh(Y)  # all-super-Gaussian production setting

        model_fitness = xp.eye(D, dtype=self.dtype) + (Y @ F.T) / L

        delta = self.nsi_delta
        if self.Rn is None:
            self.Rn = model_fitness.copy()
        else:
            self.Rn = (1.0 - delta) * self.Rn + delta * model_fitness

        nsi = xp.linalg.norm(self.Rn, ord="fro")
        nsi_norm = nsi / xp.sqrt(xp.asarray(float(D), dtype=self.dtype))

        nsi_cpu = float(self._to_cpu(nsi))
        if not np.isfinite(nsi_cpu):
            raise ValueError(f"Invalid NSI: {nsi_cpu}")

        # ================================================================
        # 3) Controller forgetting factors for ORICA
        # ================================================================
        lambda_k = self.controller.ica_lambdas(sample_indices, nsi, xp)
        lambda_k = xp.asarray(lambda_k, dtype=self.dtype).reshape(-1)

        if lambda_k.size != L:
            raise ValueError(
                f"Controller returned {lambda_k.size} ICA lambdas for a block of {L} samples"
            )

        lambda_cpu = self._to_cpu(lambda_k)
        if not np.all(np.isfinite(lambda_cpu)):
            raise ValueError("ICA forgetting factors contain NaN or Inf")
        if np.any(lambda_cpu <= 0.0) or np.any(lambda_cpu >= 1.0):
            raise ValueError("ICA forgetting factors must lie strictly between 0 and 1")

        # ================================================================
        # 4) ORICA block update
        # ================================================================
        lambda_prod = xp.prod(1.0 / (1.0 - lambda_k))
        Q = 1.0 + lambda_k * (xp.sum(F * Y, axis=0) - 1.0)

        if not bool(self._to_cpu(xp.all(xp.isfinite(Q)))):
            raise ValueError("ORICA Q contains NaN or Inf")
        if bool(self._to_cpu(xp.any(xp.abs(Q) < 1e-15))):
            raise FloatingPointError("ORICA Q is numerically singular")

        weighted_Y = Y * (lambda_k / Q)[None, :]
        self.W = lambda_prod * (self.W - weighted_Y @ F.T @ self.W)

        self.blocks_seen += 1
        if self.blocks_seen % self.config.orthogonalize_every == 0:
            self.W = self._symmetric_orthogonalize(self.W)

        self.samples_seen += L
        self.last_lambda_white = lambda_white_cpu
        self.last_lambda_ica = lambda_k.copy()
        self.last_nsi = nsi_cpu
        self.last_nsi_normalized = float(self._to_cpu(nsi_norm))

        result = {
            "V": V,
            "Y": Y,
            "lambda_white": lambda_white,
            "lambda_ica": lambda_k,
            "nsi": nsi,
            "nsi_normalized": nsi_norm,
            "sample_indices": sample_indices,
        }

        if return_cpu:
            return {
                key: (
                    float(self._to_cpu(value))
                    if getattr(value, "ndim", None) == 0
                    else self._to_cpu(value).copy()
                )
                for key, value in result.items()
            }

        return result

    # ------------------------------------------------------------------
    # read-only use of learned transform
    # ------------------------------------------------------------------
    def get_unmixing(self, cpu: bool = True):
        """Return the total unmixing matrix ``B = W @ M``."""
        B = self.W @ self.M
        return self._copy_out(B, cpu=cpu)

    def transform(self, X_block, cpu: bool = True):
        """Apply the current total unmixing matrix without changing state."""
        X = self._validate_input(X_block)
        Y = (self.W @ self.M) @ X
        return self._copy_out(Y, cpu=cpu)

    def get_state(self, cpu: bool = True) -> Dict[str, Any]:
        """Return a snapshot of the current ORICA state."""
        state = {
            "M": self._copy_out(self.M, cpu=cpu),
            "W": self._copy_out(self.W, cpu=cpu),
            "Rn": None if self.Rn is None else self._copy_out(self.Rn, cpu=cpu),
            "samples_seen": int(self.samples_seen),
            "blocks_seen": int(self.blocks_seen),
            "lambda_white": float(self.last_lambda_white),
            "lambda_ica": (
                None
                if self.last_lambda_ica is None
                else self._copy_out(self.last_lambda_ica, cpu=cpu)
            ),
            "nsi": float(self.last_nsi),
            "nsi_normalized": float(self.last_nsi_normalized),
            "backend": self.backend,
        }
        return state

    def set_state(self, state: Dict[str, Any]) -> "OnlineORICA":
        """Restore M/W/Rn and counters from a previously saved snapshot.

        This method is intentionally conservative: controller-internal state is
        not restored. It is primarily intended for the production Cooling
        controller, which is stateless apart from the global sample index.
        """
        xp = self.xp
        self.M = xp.asarray(state["M"], dtype=self.dtype)
        self.W = xp.asarray(state["W"], dtype=self.dtype)
        self.Rn = (
            None
            if state.get("Rn") is None
            else xp.asarray(state["Rn"], dtype=self.dtype)
        )
        self.samples_seen = int(state.get("samples_seen", 0))
        self.blocks_seen = int(state.get("blocks_seen", 0))
        self.last_lambda_white = float(state.get("lambda_white", np.nan))
        self.last_nsi = float(state.get("nsi", np.nan))
        self.last_nsi_normalized = float(state.get("nsi_normalized", np.nan))
        return self
