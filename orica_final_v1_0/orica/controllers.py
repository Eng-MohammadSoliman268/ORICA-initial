"""Forgetting-factor controllers for ORICA.

The production controller is :class:`CoolingController` with
``lambda0=0.995`` and ``gamma=0.60``.
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
