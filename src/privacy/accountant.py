"""Per-client Opacus RDP accounting without Opacus DP optimizers."""

from __future__ import annotations

import math
import time
import opacus

from opacus.accountants import RDPAccountant


class ClientRDPAccountant:
    """Track actual committed Poisson DP-SGD steps for one client.

    A zero noise multiplier is explicitly a non-DP control: it records the
    number of steps but does not ask RDPAccountant for a misleading finite
    epsilon.
    """

    def __init__(self, noise_multiplier: float, sample_rate: float,
                 max_grad_norm: float, delta: float, state=None):
        self.noise_multiplier = float(noise_multiplier)
        self.sample_rate = float(sample_rate)
        self.max_grad_norm = float(max_grad_norm)
        self.delta = float(delta)
        if not math.isfinite(self.noise_multiplier) or self.noise_multiplier < 0:
            raise ValueError("noise_multiplier must be finite and non-negative")
        if not 0 < self.sample_rate <= 1:
            raise ValueError("sample_rate must be in (0, 1]")
        if not math.isfinite(self.max_grad_norm) or self.max_grad_norm <= 0:
            raise ValueError("max_grad_norm must be finite and positive")
        if not 0 < self.delta < 1:
            raise ValueError("delta must be in (0, 1)")
        self.steps = 0
        self._epsilon_cache = None
        self.accountant = RDPAccountant() if self.noise_multiplier > 0 else None
        if state is not None:
            self._load(state)

    @property
    def is_dp(self):
        return self.noise_multiplier > 0

    def step(self):
        """Account one completed optimizer step, including an empty draw."""
        if self.is_dp:
            self.accountant.step(noise_multiplier=self.noise_multiplier,
                                 sample_rate=self.sample_rate)
        self.steps += 1

    def epsilon(self):
        if not self.is_dp or self.steps == 0:
            return None
        if self._epsilon_cache is not None and self._epsilon_cache[0] == self.steps:
            return self._epsilon_cache[1]
        value = float(self.accountant.get_epsilon(delta=self.delta))
        value = value if math.isfinite(value) else None
        self._epsilon_cache = (self.steps, value)
        return value

    def state_dict_with_timing(self):
        started = time.perf_counter()
        accountant_state = self.accountant.state_dict() if self.is_dp else None
        state_read_seconds = time.perf_counter() - started
        epsilon_started = time.perf_counter()
        epsilon = self.epsilon()
        epsilon_seconds = time.perf_counter() - epsilon_started
        state = {
            "noise_multiplier": self.noise_multiplier,
            "sample_rate": self.sample_rate,
            "max_grad_norm": self.max_grad_norm,
            "delta": self.delta,
            "steps": self.steps,
            "accountant_name": "opacus.RDPAccountant",
            "accountant_version": opacus.__version__,
            "accountant_state": accountant_state,
            "theoretical_epsilon": epsilon,
            "epsilon_claim_allowed": False,
            "status": ("theoretical_rdp_research_only" if self.is_dp
                       else "non_dp_sigma_zero_epsilon_undefined"),
        }
        return state, {"accountant_state_read_seconds": state_read_seconds,
                       "epsilon_calculation_seconds": epsilon_seconds}

    def state_dict(self):
        return self.state_dict_with_timing()[0]

    def _load(self, state):
        for key, current in (("noise_multiplier", self.noise_multiplier),
                             ("sample_rate", self.sample_rate),
                             ("max_grad_norm", self.max_grad_norm),
                             ("delta", self.delta)):
            if float(state[key]) != current:
                raise ValueError("Accountant resume mismatch: " + key)
        if state.get("accountant_name") != "opacus.RDPAccountant":
            raise ValueError("Unsupported privacy accountant in checkpoint")
        if state.get("accountant_version") != opacus.__version__:
            raise ValueError("RDP accountant version differs from the checkpoint")
        self.steps = int(state["steps"])
        if self.steps < 0:
            raise ValueError("Invalid accountant step count")
        accountant_state = state.get("accountant_state")
        if self.is_dp:
            if accountant_state is None:
                raise ValueError("DP checkpoint is missing its RDP accountant state")
            self.accountant.load_state_dict(accountant_state)
            recorded = sum(int(item[2]) for item in accountant_state.get("history", []))
            if recorded != self.steps:
                raise ValueError("RDP history count does not match committed DP steps")
        elif accountant_state is not None:
            raise ValueError("Sigma-zero checkpoint must not contain a finite-DP accountant")


def secure_noise_backend_available() -> bool:
    """The G3.1 implementation has no reviewed CSPRNG backend."""
    # Deliberately do not infer security from torchcsprng's mere presence.
    # G3.1 does not wire or audit a secure Gaussian source, so claim mode stays
    # unavailable until a separately reviewed backend is implemented.
    return False


def require_privacy_claim_backend(mode: str):
    if mode == "privacy_claim" and not secure_noise_backend_available():
        raise RuntimeError(
            "privacy_claim mode is unavailable: this G3.1 build has no reviewed "
            "cryptographically secure Gaussian noise backend; no PRNG fallback was used"
        )
    if mode not in {"research", "privacy_claim"}:
        raise ValueError("mode must be research or privacy_claim")
