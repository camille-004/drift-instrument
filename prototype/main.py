"""Drift Instrument prototype entry point."""

from __future__ import annotations

import queue
import threading
import time

import numpy as np
import sounddevice as sd
from scipy.io import wavfile

from prototype.constants import BLOCK_SIZE, SAMPLE_RATE
from prototype.controller import control_loop, make_keyboard_listener
from prototype.drift import DriftAxis
from prototype.imu_sim import IMU
from prototype.plot import run_plot
from prototype.synth import SynthState, make_audio_callback


def main() -> None:
    synth = SynthState()
    imu = IMU()
    drift_x = DriftAxis()
    drift_y = DriftAxis()
    drift_z = DriftAxis()

    events: queue.Queue[str] = queue.Queue()
    stop_flag = threading.Event()

    control_thread = threading.Thread(
        target=control_loop,
        args=(synth, imu, drift_x, drift_y, drift_z, events, stop_flag),
        daemon=True,
        name="control",
    )
    listener = make_keyboard_listener(events, stop_flag)

    audio_callback = make_audio_callback(synth)
    stream = sd.OutputStream(
        samplerate=SAMPLE_RATE,
        blocksize=BLOCK_SIZE,
        channels=1,
        callback=audio_callback,
    )

    print("Drift Instrument Prototype")
    print("  SPACE - tap (inject momentum)")
    print("  i     - toggle inverted orientation")
    print("  q     - quit (or close the plot window)")

    control_thread.start()
    listener.start()
    stream.start()

    try:
        run_plot(stop_flag)
    finally:
        stop_flag.set()

        # Ramp master gain to 0 to prevent click on abrupt stop
        fade_steps = 40
        for i in range(fade_steps):
            synth.master_gain = 1.0 - (i + 1) / fade_steps
            time.sleep(0.002)
        synth.record_enabled = False
        sd.sleep(100)

        stream.stop()
        stream.close()
        listener.stop()
        control_thread.join(timeout=1.0)

        if synth.recording:
            audio = np.concatenate(synth.recording).flatten()
            n_fade = min(int(SAMPLE_RATE * 0.05), len(audio))
            audio[-n_fade:] *= np.linspace(1.0, 0.0, n_fade)
            audio_int16 = (audio * 32767).astype(np.int16)
            wavfile.write("drift_session.wav", SAMPLE_RATE, audio_int16)
            duration_sec = len(audio) / SAMPLE_RATE
            print(f"Saved {duration_sec:.1f}s recording to drift_session.wav")

        print("Stopped.")


if __name__ == "__main__":
    main()