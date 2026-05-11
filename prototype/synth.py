"""Single-voice synthesizer driven by three drift dimensions.

  X drift → pitch (log-mapped fundamental)
  Y drift → timbre (sine ↔ sine+3rd-harmonic morph)
  Z drift → detune (shadow oscillator offset, creates beating)

The audio callback runs in PortAudio's thread at ~86 Hz (44100 / 512).
Synth parameters are updated from the control thread at 128 Hz. They
communicate via shared float fields on SynthState.
"""

from __future__ import annotations

import sys
import threading
from dataclasses import dataclass, field

import numpy as np

from prototype.constants import (
    DRIFT_MAX,
    DRIFT_MIN,
    PITCH_HI_HZ,
    PITCH_LO_HZ,
    SAMPLE_RATE,
)


@dataclass
class SynthState:
    """Shared state between control thread and audio thread.

    Synth params (pitch_hz, timbre, detune_hz) are updated by the control
    loop based on drift.
    """

    pitch_hz: float = 110.0
    timbre: float = 0.0
    detune_hz: float = 0.0

    fundamental_phase: float = 0.0
    shadow_phase: float = 0.0

    master_gain: float = 1.0

    # For offline inspection
    recording: list = field(default_factory=list)
    record_enabled: bool = True

    lock: threading.Lock = field(default_factory=threading.Lock)


def drift_to_range(position: float, lo: float, hi: float) -> float:
    """Map a drift position in [DRIFT_MIN, DRIFT_MAX] to [lo, hi] linearly."""
    t = (position - DRIFT_MIN) / (DRIFT_MAX - DRIFT_MIN)
    return lo + t * (hi - lo)


def drift_to_pitch(position: float) -> float:
    """Map drift position to pitch logarithmically."""
    t = (position - DRIFT_MIN) / (DRIFT_MAX - DRIFT_MIN)
    return PITCH_LO_HZ * (PITCH_HI_HZ / PITCH_LO_HZ) ** t


def make_audio_callback(synth: SynthState, log_every: int = 86):
    """Create the real-time audio callback bound to a SynthState.

    Implementation notes:

    1. Advance phase per-sample by a frequency-dependent increment so sine
       waves stay continuous across block boundaries even when frequency 
       changes between blocks. Naive `sin(2π*f*t + phase)` per block produces
       clicks at boundaries whenever f changes.
       Reference: https://gkbrk.com/wiki/PhaseAccumulator
       DDS background: https://www.analog.com/en/resources/analog-dialogue/articles/all-about-direct-digital-synthesis.html

    2. The audio callback must never wait on a lock held by another thread, or
       it risks buffer underruns audible as clicks. We read shared floats directly
       (single-attribute reads are atomic under CPython's GIL) and accept rare torn
       reads — control-rate smoothing prevents jumps that would matter.
       Reference: https://python-sounddevice.readthedocs.io/en/latest/api/misc.html
    """
    log_counter = [0]

    def audio_callback(outdata: np.ndarray, frames: int, _time_info, status) -> None:
        if status:
            print(f"AUDIO STATUS: {status}", file=sys.stderr, flush=True)

        # Lock-free reads
        f = synth.pitch_hz
        timbre = synth.timbre
        detune = synth.detune_hz
        p_fund = synth.fundamental_phase
        p_shad = synth.shadow_phase
        gain = synth.master_gain

        log_counter[0] += 1
        if log_counter[0] % log_every == 0:
            print(
                f"[audio] f={f:6.1f}Hz  timbre={timbre:.2f}  detune={detune:+.2f}Hz",
                flush=True,
            )

        dphase_fund = 2 * np.pi * f / SAMPLE_RATE
        dphase_shad = 2 * np.pi * (f + detune) / SAMPLE_RATE

        phase_fund = p_fund + dphase_fund * np.arange(frames)
        phase_shad = p_shad + dphase_shad * np.arange(frames)

        sine = np.sin(phase_fund)
        third = np.sin(3 * phase_fund) * 0.3
        fundamental = sine + timbre * third
        shadow = np.sin(phase_shad)

        mix = (fundamental + 0.5 * shadow) / 1.5 * 0.4 * gain
        outdata[:] = mix.reshape(-1, 1)

        if synth.record_enabled:
            synth.recording.append(mix.copy())

        # Advance phase for next block
        synth.fundamental_phase = (p_fund + dphase_fund * frames) % (2 * np.pi)
        synth.shadow_phase = (p_shad + dphase_shad * frames) % (2 * np.pi)

    return audio_callback
