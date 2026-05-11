"""Live drift plot with gesture markers.

Renders three rolling lines (X/Y/Z drift positions) plus vertical markers
for tap and inversion events.
"""

from __future__ import annotations

import threading
import time

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation

from prototype.constants import DRIFT_MAX, DRIFT_MIN, HISTORY_LEN, HISTORY_SECONDS
from prototype.controller import (
    gesture_events,
    gesture_lock,
    history_lock,
    history_x,
    history_y,
    history_z,
)


def run_plot(stop_flag: threading.Event) -> None:
    """Show a live plot of the drift accumulators. Blocks until window closes."""
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.set_ylim(DRIFT_MIN, DRIFT_MAX)
    ax.set_xlim(0, HISTORY_SECONDS)
    ax.set_xlabel("Seconds (rolling)")
    ax.set_ylabel("Drift Position")
    ax.set_title("Drift Instrument — SPACE: tap   i: invert   q: quit")
    ax.axhline(0, color="gray", linewidth=0.5)

    t_axis = np.linspace(0, HISTORY_SECONDS, HISTORY_LEN)
    (line_x,) = ax.plot(t_axis, history_x, label="X (pitch)", linewidth=1.5)
    (line_y,) = ax.plot(t_axis, history_y, label="Y (timbre)", linewidth=1.5)
    (line_z,) = ax.plot(t_axis, history_z, label="Z (detune)", linewidth=1.5)
    ax.legend(loc="upper right")

    gesture_artists: list = []

    def update(_frame):
        with history_lock:
            line_x.set_ydata(history_x)
            line_y.set_ydata(history_y)
            line_z.set_ydata(history_z)

        for artist in gesture_artists:
            artist.remove()
        gesture_artists.clear()

        now = time.monotonic()
        with gesture_lock:
            events_snapshot = list(gesture_events)
        for ts, label in events_snapshot:
            age = now - ts
            if 0 <= age <= HISTORY_SECONDS:
                x = HISTORY_SECONDS - age
                color = "red" if label == "TAP" else "purple"
                vline = ax.axvline(x, color=color, linestyle="--", alpha=0.6, linewidth=1)
                text = ax.text(
                    x, DRIFT_MAX * 0.92, label,
                    rotation=90, fontsize=8, color=color, alpha=0.8,
                    verticalalignment="top",
                )
                gesture_artists.extend([vline, text])

        return [line_x, line_y, line_z, *gesture_artists]

    _anim = FuncAnimation(fig, update, interval=33, blit=False, cache_frame_data=False)

    fig.canvas.mpl_connect("close_event", lambda _evt: stop_flag.set())

    plt.show()
