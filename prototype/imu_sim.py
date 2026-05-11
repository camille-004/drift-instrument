"""Simulated MPU-6050 for offline prototype development.

Produces noisy 3-axis accelerometer readings with gravity.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from prototype.constants import GRAVITY_LSB, SENSOR_NOISE_STDDEV


@dataclass
class IMU:
    """Pretends to be an MPU-6050.

    Produces noisy 3-axis readings. Gravity sits on Z (+1g upright,
    -1g inverted).
    """

    inverted: bool = False

    def read(self) -> tuple[float, float, float]:
        """Return (ax, ay, az) in raw LSB units, like the real chip would."""
        gz = -GRAVITY_LSB if self.inverted else GRAVITY_LSB

        ax = float(np.random.normal(0, SENSOR_NOISE_STDDEV))
        ay = float(np.random.normal(0, SENSOR_NOISE_STDDEV))
        az = float(np.random.normal(0, SENSOR_NOISE_STDDEV) + gz)

        return ax, ay, az