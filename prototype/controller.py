"""Control loop.

Also contains the keyboard listener and the shared history/gesture buffers
the plot reads from.
"""

from __future__ import annotations

import queue
import threading
import time

import numpy as np
from pynput import keyboard

from prototype.constants import (
    CONTROL_RATE,
    DT,
    GRAVITY_LSB,
    HISTORY_LEN,
    HISTORY_SECONDS,
    INVERT_BIAS,
    SMOOTHING,
    TAP_MOMENTUM,
)
from prototype.drift import DriftAxis
from prototype.imu_sim import IMU
from prototype.synth import SynthState, drift_to_pitch, drift_to_range


# ---------------------------------------------------------------------------
# Shared buffers for the live plot
# ---------------------------------------------------------------------------

history_x: list[float] = [0.0] * HISTORY_LEN
history_y: list[float] = [0.0] * HISTORY_LEN
history_z: list[float] = [0.0] * HISTORY_LEN
history_lock = threading.Lock()

gesture_events: list[tuple[float, str]] = []
gesture_lock = threading.Lock()


def push_history(x: float, y: float, z: float) -> None:
    """Add a new drift reading to the rolling history buffer."""
    with history_lock:
        history_x.append(x)
        history_y.append(y)
        history_z.append(z)
        if len(history_x) > HISTORY_LEN:
            history_x.pop(0)
            history_y.pop(0)
            history_z.pop(0)


def log_gesture(label: str) -> None:
    """Record a gesture event for plotting + console output."""
    now = time.monotonic()
    with gesture_lock:
        gesture_events.append((now, label))
        cutoff = now - HISTORY_SECONDS
        while gesture_events and gesture_events[0][0] < cutoff:
            gesture_events.pop(0)
    print(f"[gesture] {label}", flush=True)


# ---------------------------------------------------------------------------
# Control loop
# ---------------------------------------------------------------------------


def control_loop(
    synth: SynthState,
    imu: IMU,
    drift_x: DriftAxis,
    drift_y: DriftAxis,
    drift_z: DriftAxis,
    events: queue.Queue[str],
    stop_flag: threading.Event,
) -> None:
    """Run the drift → synth control loop at CONTROL_RATE Hz until stop_flag."""
    next_tick = time.monotonic()

    while not stop_flag.is_set():
        # Handle queued gesture events
        while not events.empty():
            event = events.get_nowait()
            if event == "tap":
                # Random 3D unit vector — distribute momentum across all axes
                theta = np.random.uniform(0, 2 * np.pi)
                phi = np.random.uniform(0, np.pi)
                kx = TAP_MOMENTUM * np.sin(phi) * np.cos(theta)
                ky = TAP_MOMENTUM * np.sin(phi) * np.sin(theta)
                kz = TAP_MOMENTUM * np.cos(phi)
                drift_x.kick(kx)
                drift_y.kick(ky)
                drift_z.kick(kz)
                log_gesture("TAP")
            elif event == "invert":
                imu.inverted = not imu.inverted
                log_gesture("INVERT ON" if imu.inverted else "INVERT OFF")

        # Read sensor, update drift
        ax, ay, az = imu.read()
        az_gravity = -GRAVITY_LSB if imu.inverted else GRAVITY_LSB

        # Inversion biases drift toward zero — destabilizes into pendulum motion
        if imu.inverted:
            bx = -drift_x.position * INVERT_BIAS
            by = -drift_y.position * INVERT_BIAS
            bz = -drift_z.position * INVERT_BIAS
        else:
            bx = by = bz = 0.0

        drift_x.update(ax, bx)
        drift_y.update(ay, by)
        drift_z.update(az - az_gravity, bz)

        push_history(drift_x.position, drift_y.position, drift_z.position)

        # Map drift to synth params, smoothed
        pitch_hz = drift_to_pitch(drift_x.position)
        timbre = drift_to_range(drift_y.position, 0.0, 1.0)
        detune_hz = drift_to_range(drift_z.position, -8.0, 8.0)

        with synth.lock:
            synth.pitch_hz += (pitch_hz - synth.pitch_hz) * SMOOTHING
            synth.timbre += (timbre - synth.timbre) * SMOOTHING
            synth.detune_hz += (detune_hz - synth.detune_hz) * SMOOTHING

        # Sleep until next tick for stable timing
        next_tick += DT
        sleep_for = next_tick - time.monotonic()
        if sleep_for > 0:
            time.sleep(sleep_for)
        else:
            next_tick = time.monotonic()


# ---------------------------------------------------------------------------
# Keyboard input
# ---------------------------------------------------------------------------


def make_keyboard_listener(
    events: queue.Queue[str],
    stop_flag: threading.Event,
) -> keyboard.Listener:
    """Build a pynput listener that translates keypresses into queue events.

    SPACE → "tap"
    i     → "invert"
    q     → set stop_flag and stop listener
    """

    def on_press(key) -> bool | None:
        try:
            char = key.char
        except AttributeError:
            char = None

        if key == keyboard.Key.space:
            events.put("tap")
        elif char == "i":
            events.put("invert")
        elif char == "q":
            stop_flag.set()
            return False

        return None

    return keyboard.Listener(on_press=on_press)