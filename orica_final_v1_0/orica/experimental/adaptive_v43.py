"""Experimental V4.3 adaptive ORICA controller.

Status
------
EXPERIMENTAL. Not used by the production ORICA_FINAL_V1.0 profile.

Design
------
V4.3 uses an *exact* CoolingController as its immutable backbone and allows an
adaptive detector to add an upward forgetting-factor boost. Therefore, when
boost == 0, V4.3 is exactly equivalent to Cooling at both whitening and ICA
controller outputs.

The synthetic validation was useful, but the final real 120-D HBN experiment
showed that adaptive boost events could degrade whitening relative to pure
Cooling. The file is retained for research/development, not deployment.
"""

from __future__ import annotations

import math
from collections import deque
from statistics import NormalDist

import numpy as np

from ..controllers import CoolingController


class GeneralSelfTuningControllerV43:
    """Exact Cooling backbone + optional self-tuning upward boost."""

    def __init__(
        self,
        n_components,
        sfreq,
        block_size=8,
        cooling_lambda0=0.995,
        cooling_gamma=0.60,
        protect_base_s=5.0,
        protect_dimension_scale_s=3.0,
        protect_min_s=10.0,
        protect_max_s=30.0,
        handoff_s=20.0,
        baseline_seed_s=8.0,
        baseline_memory_s=60.0,
        baseline_min_s=3.0,
        baseline_freeze_gain=0.50,
        baseline_scale_floor_rel=0.05,
        tracking_memory_s=0.50,
        p_soft=0.01,
        p_hard=0.001,
        evidence_tau_s=0.25,
        boost_rise_s=0.50,
        boost_fall_s=2.50,
        boost_gamma=1.0,
        lambda_max=0.995,
        nsi_is_normalized=False,
        eps=1e-12,
    ):
        self.D = int(n_components)
        self.sfreq = float(sfreq)
        self.block_size = int(block_size)
        self.block_dt = self.block_size / self.sfreq
        self.eps = float(eps)
        self.nsi_is_normalized = bool(nsi_is_normalized)

        self.cooling_lambda0 = float(cooling_lambda0)
        self.cooling_gamma = float(cooling_gamma)
        self.lambda_max = float(lambda_max)
        self.cooling_backbone = CoolingController(
            lambda0=self.cooling_lambda0,
            gamma=self.cooling_gamma,
        )

        raw_protect_s = (
            protect_base_s
            + protect_dimension_scale_s * math.log2(max(self.D, 2))
        )
        self.protect_s = float(
            np.clip(raw_protect_s, protect_min_s, protect_max_s)
        )
        self.handoff_s = max(float(handoff_s), self.block_dt)
        self.full_adaptive_s = self.protect_s + self.handoff_s

        self.baseline_seed_s = min(float(baseline_seed_s), self.protect_s)
        self.baseline_seed_start_s = max(
            0.0, self.protect_s - self.baseline_seed_s
        )
        self.baseline_seed_start_sample = int(
            round(self.baseline_seed_start_s * self.sfreq)
        )

        self.lambda_tracking = 1.0 - math.exp(
            -1.0 / (self.sfreq * float(tracking_memory_s))
        )
        self.lambda_tracking = float(
            np.clip(self.lambda_tracking, 0.0, self.lambda_max)
        )

        self.p_soft = float(p_soft)
        self.p_hard = float(p_hard)
        self.q_soft = NormalDist().inv_cdf(1.0 - self.p_soft)
        self.q_hard = NormalDist().inv_cdf(1.0 - self.p_hard)

        self.evidence_tau_s = float(evidence_tau_s)
        self.evidence_eta = 1.0 - math.exp(
            -self.block_dt / self.evidence_tau_s
        )

        self.baseline_memory_s = float(baseline_memory_s)
        self.baseline_capacity = max(
            10,
            int(math.ceil(self.baseline_memory_s / self.block_dt)),
        )
        self.baseline_min_blocks = max(
            3,
            int(math.ceil(float(baseline_min_s) / self.block_dt)),
        )
        self.baseline_freeze_gain = float(baseline_freeze_gain)
        self.baseline_scale_floor_rel = float(baseline_scale_floor_rel)

        self.boost_rise_coeff = 1.0 - math.exp(
            -1.0 / (self.sfreq * float(boost_rise_s))
        )
        self.boost_fall_coeff = 1.0 - math.exp(
            -1.0 / (self.sfreq * float(boost_fall_s))
        )
        self.boost_gamma = float(boost_gamma)

        self.reset()

    def reset(self):
        self.phase = "PROTECTED"
        self.samples_seen = 0
        self.time_s = 0.0
        self.handoff_weight = 0.0

        self.G_instant = 0.0
        self.G_detector = 0.0
        self.G_effective = 0.0
        self.q = 0.0

        self.boost_target = 0.0
        self.boost_value = 0.0

        self.lambda_cooling = float(self.cooling_lambda0)
        self.lambda_value = float(self.cooling_lambda0)
        self.lambda_target = float(self.cooling_lambda0)

        self.z = np.nan
        self.baseline = deque(maxlen=self.baseline_capacity)
        self.baseline_median = np.nan
        self.baseline_scale = np.nan
        self.baseline_frozen = False

        self.cooling_backbone.reset()
        return self

    @staticmethod
    def _scalar(value):
        try:
            if hasattr(value, "item"):
                return float(value.item())
        except Exception:
            pass
        return float(np.asarray(value).reshape(-1)[0])

    @staticmethod
    def _to_numpy_indices(sample_indices, xp):
        try:
            if hasattr(xp, "asnumpy"):
                return np.asarray(xp.asnumpy(sample_indices), dtype=np.float64)
        except Exception:
            pass
        return np.asarray(sample_indices, dtype=np.float64)

    @staticmethod
    def _to_numpy_float(value, xp):
        try:
            if hasattr(xp, "asnumpy"):
                return np.asarray(xp.asnumpy(value), dtype=np.float64)
        except Exception:
            pass
        return np.asarray(value, dtype=np.float64)

    def _normalize_nsi(self, nsi):
        z = self._scalar(nsi)
        if not self.nsi_is_normalized:
            z /= math.sqrt(self.D)
        return float(z)

    def _phase_from_time(self, t):
        if t < self.protect_s:
            return "PROTECTED"
        if t < self.full_adaptive_s:
            return "HANDOFF"
        return "ADAPTIVE"

    def _handoff_weight(self, t):
        if t <= self.protect_s:
            return 0.0
        if t >= self.full_adaptive_s:
            return 1.0
        x = float(np.clip((t - self.protect_s) / self.handoff_s, 0.0, 1.0))
        return float(3.0 * x**2 - 2.0 * x**3)

    def _baseline_stats(self):
        if len(self.baseline) == 0:
            return np.nan, np.nan

        values = np.asarray(self.baseline, dtype=np.float64)
        median = float(np.median(values))
        mad = float(np.median(np.abs(values - median)))
        robust_sigma = 1.4826 * mad
        scale_floor = self.baseline_scale_floor_rel * max(abs(median), self.eps)
        scale = max(robust_sigma, scale_floor, self.eps)
        return median, scale

    def _q_to_gain(self, q):
        if q <= self.q_soft:
            return 0.0
        if q >= self.q_hard:
            return 1.0
        x = (q - self.q_soft) / (self.q_hard - self.q_soft)
        return float(x**2 * (3.0 - 2.0 * x))

    def whitening_lambda(self, sample_indices, xp):
        base = self.cooling_backbone.whitening_lambda(sample_indices, xp)
        base_float = self._scalar(base)

        boost = float(np.clip(self.boost_value, 0.0, 1.0))
        increase_range = max(self.lambda_tracking - base_float, 0.0)
        final_lambda = base_float + boost * increase_range
        final_lambda = float(np.clip(final_lambda, 0.0, self.lambda_max))

        return xp.asarray(final_lambda, dtype=xp.float64)

    def ica_lambdas(self, sample_indices, nsi, xp):
        idx_cpu = self._to_numpy_indices(sample_indices, xp)
        L = int(idx_cpu.size)
        if L <= 0:
            return xp.asarray([], dtype=xp.float64)

        last_index = float(idx_cpu[-1])
        self.samples_seen = int(round(last_index))
        self.time_s = self.samples_seen / self.sfreq

        self.phase = self._phase_from_time(self.time_s)
        self.handoff_weight = self._handoff_weight(self.time_s)
        self.z = self._normalize_nsi(nsi)

        if (
            self.samples_seen >= self.baseline_seed_start_sample
            and self.phase == "PROTECTED"
            and np.isfinite(self.z)
        ):
            self.baseline.append(self.z)

        if self.phase == "PROTECTED":
            self.q = 0.0
            self.G_instant = 0.0
            self.G_detector = 0.0
        else:
            median, scale = self._baseline_stats()
            self.baseline_median = median
            self.baseline_scale = scale

            if (
                len(self.baseline) >= self.baseline_min_blocks
                and np.isfinite(median)
                and np.isfinite(scale)
                and np.isfinite(self.z)
            ):
                self.q = max(0.0, (self.z - median) / scale)
                self.G_instant = self._q_to_gain(self.q)
            else:
                self.q = 0.0
                self.G_instant = 0.0

            self.G_detector = (
                (1.0 - self.evidence_eta) * self.G_detector
                + self.evidence_eta * self.G_instant
            )
            self.G_detector = float(np.clip(self.G_detector, 0.0, 1.0))

        self.G_effective = float(
            np.clip(self.handoff_weight * self.G_detector, 0.0, 1.0)
        )

        if self.phase != "PROTECTED":
            self.baseline_frozen = self.G_detector >= self.baseline_freeze_gain
            if not self.baseline_frozen and np.isfinite(self.z):
                self.baseline.append(self.z)
            self.baseline_median, self.baseline_scale = self._baseline_stats()

        cooling_vector_xp = self.cooling_backbone.ica_lambdas(
            sample_indices, nsi, xp
        )
        cooling_vector = self._to_numpy_float(cooling_vector_xp, xp).reshape(-1)
        if cooling_vector.size != L:
            raise RuntimeError(
                "Cooling backbone returned wrong lambda-vector size: "
                f"{cooling_vector.size} != {L}"
            )

        boost = float(self.boost_value)
        lambda_values = np.empty(L, dtype=np.float64)

        for k in range(L):
            sample_index = float(idx_cpu[k])
            t_sample = sample_index / self.sfreq
            h_sample = self._handoff_weight(t_sample)
            G_sample = float(
                np.clip(h_sample * self.G_detector, 0.0, 1.0)
            )
            target_boost = G_sample**self.boost_gamma

            coeff = (
                self.boost_rise_coeff
                if target_boost > boost
                else self.boost_fall_coeff
            )
            boost = boost + coeff * (target_boost - boost)
            boost = float(np.clip(boost, 0.0, 1.0))

            base_lambda = float(cooling_vector[k])
            increase_range = max(self.lambda_tracking - base_lambda, 0.0)
            lambda_values[k] = base_lambda + boost * increase_range

        self.boost_value = float(boost)
        self.boost_target = float(self.G_effective**self.boost_gamma)
        self.lambda_cooling = float(cooling_vector[-1])
        self.lambda_value = float(lambda_values[-1])
        self.lambda_target = self.lambda_cooling + self.boost_target * max(
            self.lambda_tracking - self.lambda_cooling, 0.0
        )

        return xp.asarray(lambda_values, dtype=xp.float64)

    def get_state(self):
        return {
            "profile": "experimental_v4.3",
            "phase": self.phase,
            "samples_seen": self.samples_seen,
            "time_s": self.time_s,
            "protect_s": self.protect_s,
            "handoff_s": self.handoff_s,
            "full_adaptive_s": self.full_adaptive_s,
            "handoff_weight": self.handoff_weight,
            "NSI_normalized": self.z,
            "q": self.q,
            "q_soft": self.q_soft,
            "q_hard": self.q_hard,
            "G_instant": self.G_instant,
            "G_detector": self.G_detector,
            "G_effective": self.G_effective,
            "boost_target": self.boost_target,
            "boost_value": self.boost_value,
            "lambda": self.lambda_value,
            "lambda_value": self.lambda_value,
            "lambda_target": self.lambda_target,
            "lambda_cooling": self.lambda_cooling,
            "lambda_tracking": self.lambda_tracking,
            "cooling_lambda0": self.cooling_lambda0,
            "cooling_gamma": self.cooling_gamma,
            "baseline_count": len(self.baseline),
            "baseline_median": self.baseline_median,
            "baseline_scale": self.baseline_scale,
            "baseline_frozen": self.baseline_frozen,
        }
