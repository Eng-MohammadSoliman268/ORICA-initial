"""ORICA_FINAL_V1.0 — self-contained production build.

This single file contains the final production algorithm used for integration
when a package layout is inconvenient. The modular ``orica/`` package remains
the source-of-truth layout for development.
"""

from __future__ import annotations


import math


class CoolingController:
    """Validated monotonically decreasing ORICA forgetting factor.

    The schedule is

        lambda(n) = lambda0 / n**gamma

    with one-based sample index ``n``.

    For the whitening block update, the reference implementation uses the
    middle sample of the block (MATLAB ``ceil(end/2)``). For an 8-sample block
    this is the 4th sample.
    """

    def __init__(self, lambda0: float = 0.995, gamma: float = 0.60):
        if not (0.0 < lambda0 < 1.0):
            raise ValueError("lambda0 must lie strictly between 0 and 1")
        if gamma <= 0:
            raise ValueError("gamma must be > 0")
        self.lambda0 = float(lambda0)
        self.gamma = float(gamma)

    def reset(self):
        return self

    def _vector(self, sample_indices, xp):
        idx = xp.asarray(sample_indices, dtype=xp.float64)
        nonpositive = xp.any(idx <= 0)
        if hasattr(nonpositive, "item"):
            nonpositive = nonpositive.item()
        if bool(nonpositive):
            raise ValueError("CoolingController requires positive one-based sample indices")
        return self.lambda0 / xp.power(idx, self.gamma)

    def whitening_lambda(self, sample_indices, xp):
        lam = self._vector(sample_indices, xp)
        L = int(lam.size)
        # MATLAB ceil(L/2) converted to zero-based Python index.
        mid = int(math.ceil(L / 2.0)) - 1
        return lam[mid]

    def ica_lambdas(self, sample_indices, nsi, xp):
        del nsi
        return self._vector(sample_indices, xp)

    def get_state(self):
        return {
            "profile": "cooling",
            "lambda0": self.lambda0,
            "gamma": self.gamma,
        }


class ConstantController:
    """Simple constant forgetting factor, mainly for controlled experiments."""

    def __init__(self, lambda_value: float):
        if not (0.0 < lambda_value < 1.0):
            raise ValueError("lambda_value must lie strictly between 0 and 1")
        self.lambda_value = float(lambda_value)

    def reset(self):
        return self

    def whitening_lambda(self, sample_indices, xp):
        del sample_indices
        return xp.asarray(self.lambda_value, dtype=xp.float64)

    def ica_lambdas(self, sample_indices, nsi, xp):
        del nsi
        return xp.full(sample_indices.shape, self.lambda_value, dtype=xp.float64)

    def get_state(self):
        return {
            "profile": "constant",
            "lambda": self.lambda_value,
        }



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



from typing import Dict, Tuple

import numpy as np



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