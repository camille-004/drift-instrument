"""Drift accumulator, the core of the instrument's autonomous behavior.

Raw acceleration drives velocity, velocity drives position. Position is the drift value that maps to sound.
"""

from __future__ import annotations

from dataclasses import dataclass

from prototype.constants import (
    ACCEL_SCALE,
    DRIFT_MAX,
    DRIFT_MIN,
    DT,
    VEL_DECAY,
)


@dataclass
class DriftAxis:
    """One axis of accumulated drift.

    Raw acceleration drives velocity, velocity drives position. Position is
    what we map to sound.
    """

    velocity: float = 0.0
    position: float = 0.0

    def update(self, accel_raw: float, bias_velocity: float = 0.0) -> None:
        """Advance one control tick."""
        accel = accel_raw * ACCEL_SCALE
        self.velocity = self.velocity * VEL_DECAY + accel * DT + bias_velocity * DT
        self.position += self.velocity * DT

        # Wrap position cyclically.
        span = DRIFT_MAX - DRIFT_MIN
        if self.position > DRIFT_MAX:
            self.position -= span
        elif self.position < DRIFT_MIN:
            self.position += span

    def kick(self, velocity_impulse: float) -> None:
        """Inject momentum directly into velocity. Called by tap gestures."""
        self.velocity += velocity_impulse
