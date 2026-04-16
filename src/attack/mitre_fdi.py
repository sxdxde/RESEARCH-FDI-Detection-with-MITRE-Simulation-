"""
MITRE ATT&CK for ICS — EV Charging FDI Attack Emulator
════════════════════════════════════════════════════════
Maps False Data Injection on EV charging infrastructure to the
MITRE ATT&CK for Industrial Control Systems (ICS) framework.

Primary technique:
  T0836 – Modify Parameter
    Adversary inflates the `minutesAvailable` field reported by the EV
    to the EVSE management backend.  The charging algorithm then tries to
    deliver proportionally more energy, pushing kWhDeliveredPerTimeStamp
    beyond the model's expectation.

Delivery vector:
  T0830 – Adversary-in-the-Middle
    OCPP (Open Charge Point Protocol) messages between the EVSE and the
    cloud-based Charging Management System (CMS) are intercepted and
    re-written in flight.

Secondary effect:
  T0856 – Spoof Reporting Message
    The corrupted session record propagates to the billing / analytics
    backend; an over-billed customer is the financial impact.

Physics:
    scale(θ) = 1 + θ / REFERENCE_SESSION_MIN
    where θ = extra minutes injected by the attacker.
    Additive injection avoids the zero-value blind-spot:
        y_obs = y_true + (scale - 1) * baseline_kWh
"""

from dataclasses import dataclass
from typing import Optional

import numpy as np

# ── MITRE technique catalogue ─────────────────────────────────────
TECHNIQUES: dict[str, dict] = {
    "T0836": {
        "name":        "Modify Parameter",
        "tactic":      "Impair Process Control",
        "description": "Attacker inflates minutesAvailable → charger over-delivers kWh",
        "url":         "https://attack.mitre.org/techniques/T0836/",
    },
    "T0830": {
        "name":        "Adversary-in-the-Middle",
        "tactic":      "Collection",
        "description": "OCPP message interception between EVSE and CMS",
        "url":         "https://attack.mitre.org/techniques/T0830/",
    },
    "T0856": {
        "name":        "Spoof Reporting Message",
        "tactic":      "Inhibit Response Function",
        "description": "Falsified session data propagated to billing backend",
        "url":         "https://attack.mitre.org/techniques/T0856/",
    },
}

REFERENCE_SESSION_MIN = 60.0   # median ACN session length (minutes)


@dataclass
class AttackStatus:
    active:           bool
    technique_id:     str
    technique_name:   str
    tactic:           str
    theta:            float   # minutes of extra time injected
    scale_factor:     float   # multiplicative uplift
    burst_remaining:  int     # timesteps left in current burst


class MITREFDIAttacker:
    """
    Stateful T0836 FDI injector with MITRE metadata.

    Usage
    -----
    attacker = MITREFDIAttacker()
    attacker.set_baseline(mean_kWh)

    # Manual burst
    attacker.start_attack(theta=20, duration=15)

    # Per-step: returns (possibly modified) observed value
    for y_true in stream:
        y_obs = attacker.inject(y_true)          # if manual
        y_obs = attacker.maybe_inject(y_true, p)  # if probabilistic
    """

    def __init__(self, seed: int = 42):
        self._rng             = np.random.default_rng(seed)
        self._burst_remaining = 0
        self._theta           = 20.0
        self._scale           = 1.0 + self._theta / REFERENCE_SESSION_MIN
        self._baseline_kWh    = 0.05   # mean positive charging rate (kWh/step)

    # ── Configuration ─────────────────────────────────────────────
    def set_baseline(self, baseline_kWh: float) -> None:
        self._baseline_kWh = max(1e-6, float(baseline_kWh))

    # ── Attack control ────────────────────────────────────────────
    def start_attack(self, theta: float, duration: int) -> None:
        """Begin a T0836 burst of `duration` timesteps at intensity θ."""
        self._theta           = max(1.0, float(theta))
        self._scale           = 1.0 + self._theta / REFERENCE_SESSION_MIN
        self._burst_remaining = max(1, int(duration))

    def stop_attack(self) -> None:
        self._burst_remaining = 0

    # ── Injection ─────────────────────────────────────────────────
    def inject(self, y_true: float) -> float:
        """
        Apply T0836 modification.  Decrements burst counter.
        Returns y_obs (attacked if active, else y_true unchanged).
        """
        if self._burst_remaining > 0:
            self._burst_remaining -= 1
            delta = (self._scale - 1.0) * self._baseline_kWh
            return float(y_true) + delta
        return float(y_true)

    def maybe_inject(self, y_true: float, prob: float) -> float:
        """Auto-start a random-duration attack with probability `prob` per step."""
        if self._burst_remaining <= 0 and prob > 0 and self._rng.random() < prob:
            dur = int(self._rng.integers(10, 25))
            self.start_attack(self._theta, dur)
        return self.inject(y_true)

    # ── State query ───────────────────────────────────────────────
    @property
    def is_active(self) -> bool:
        return self._burst_remaining > 0

    @property
    def theta(self) -> float:
        return self._theta

    def status(self) -> AttackStatus:
        tid = "T0836" if self.is_active else ""
        t   = TECHNIQUES.get(tid, {})
        return AttackStatus(
            active=          self.is_active,
            technique_id=    tid,
            technique_name=  t.get("name", ""),
            tactic=          t.get("tactic", ""),
            theta=           self._theta,
            scale_factor=    self._scale,
            burst_remaining= self._burst_remaining,
        )
