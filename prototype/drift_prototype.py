from __future__ import annotations

import queue
import sys
import threading
import time
from dataclasses import dataclass, field

import matplotlib.pyplot as plt
import numpy as np
import sounddevice as sd
from matplotlib.animation import FuncAnimation
from pynput import keyboard
from scipy.io import wavfile


SAMPLE_RATE = 44100
BLOCK_SIZE = 512         # Samples per audio callback
CONTROL_RATE = 128       # Drift/synth updates per second
DT = 1.0 / CONTROL_RATE  # Seconds between control ticks

# Simulated MPU-6050
ACCEL_LSB_PER_G = 16384   # LSB/g for a ±2g range
GRAVITY_LSB = ACCEL_LSB_PER_G
SENSOR_NOISE_STDDEV = 80  # Approximate noise floor of MPU-6050

# Drift accumulator tuning
ACCEL_SCALE = 0.1   # Scales raw accel into velocity. Smaller = slower drift
VEL_DECAY = 0.9995  # Per-tick velocity decay. 1.0 = no decay
DRIFT_MAX = 500.0   # Drift position wraps when it exceeds this
DRIFT_MIN = -500.0
PITCH_LO_HZ = 80.0
PITCH_HI_HZ = 320.0

# Gestures
TAP_MOMENTUM = 25.0   # Veloctiy injected by a tap
INVERT_BIAS = 0.4     # How much inversion biases the drift
SMOOTHING = 0.05      # How fast synth params follow drift


@dataclass
class DriftAxis:
    """One axis of accumulated drift.

    Raw acceleration drives velocity, velocity drives position. Position is what we map to sound.
    """

    velocity: float = 0.0
    position: float = 0.0

    def update(self, accel_raw: float, bias_velocity: float = 0.0) -> None:
            """Advance one control tick. `bias_velocity` is a gentle pull added each tick."""
            accel = accel_raw * ACCEL_SCALE
            # Velocity halves itself every ~230 ticks, prevents drift from exploding
            self.velocity = self.velocity * VEL_DECAY + accel * DT + bias_velocity * DT
            self.position += self.velocity * DT

            # Wrap position cyclically
            span = DRIFT_MAX - DRIFT_MIN
            if self.position > DRIFT_MAX:
                self.position -= span
            elif self.position < DRIFT_MIN:
                self.position += span
        
    def kick(self, velocity_impulse: float) -> None:
        """Inject momentum directly into velocity. Called by tap gestures."""
        self.velocity += velocity_impulse


# Simulated accelerometer
@dataclass
class IMU:
    """Pretends to be an MPU-6050 sitting on a table.

    Produces noisy 3-axis readings. Gravity sits on Z (+1g upright, -1g inverted). Taps inject one-tick impulsive.
    """

    inverted: bool = False
    pending_tap_x: float = 0.0
    pending_tap_y: float = 0.0
    pending_tap_z: float = 0.0

    def read(self) -> tuple[float, float, float]:
        """Return (ax, ay, az) in raw LSB units."""
        gz = -GRAVITY_LSB if self.inverted else GRAVITY_LSB

        ax = np.random.normal(0, SENSOR_NOISE_STDDEV) + self.pending_tap_x
        ay = np.random.normal(0, SENSOR_NOISE_STDDEV) + self.pending_tap_y
        az = np.random.normal(0, SENSOR_NOISE_STDDEV) + gz + self.pending_tap_z

        self.pending_tap_x = 0.0
        self.pending_tap_y = 0.0
        self.pending_tap_z = 0.0

        return ax, ay, az
    
    def tap(self) -> None:
        """Inject a sharp impulse in a random 3D direction."""
        theta = np.random.uniform(0, 2 * np.pi)
        phi = np.random.uniform(0, np.pi)
        magnitude = 30000.0  # ~1.8g
        self.pending_tap_x = magnitude * np.sin(phi) * np.cos(theta)
        self.pending_tap_y = magnitude * np.sin(phi) * np.sin(theta)
        self.pending_tap_z = magnitude * np.cos(phi)


@dataclass
class SynthState:
    """Single-voice synth state."""

    pitch_hz: float = 110.0
    timbre: float = 0.0
    detune_hz: float = 0.0

    fundamental_phase: float = 0.0
    shadow_phase: float = 0.0

    lock: threading.Lock = field(default_factory=threading.Lock)

    recording: list = field(default_factory=list)
    record_enabled: bool = True
    
    master_gain: float = 1.0


def drift_to_range(position: float, lo: float, hi: float) -> float:
    """Map a drift position in [DRIFT_MIN, DRIFT_MAX] to [lo, hi]."""
    t = (position - DRIFT_MIN) / (DRIFT_MAX - DRIFT_MIN)
    return lo + t * (hi - lo)


def drift_to_pitch(position: float) -> float:
    """Map drift position to pitch logarithmically (octaves)."""
    t = (position - DRIFT_MIN) / (DRIFT_MAX - DRIFT_MIN)
    return PITCH_LO_HZ * (PITCH_HI_HZ / PITCH_LO_HZ) ** t


def make_audio_callback(synth: SynthState):
    """Create real-time audio callback bound to a SynthState."""

    log_counter = [0]
    LOG_EVERY = 86

    # Phase-accumulator DDS oscillator pattern: advance phase per-sample by a
    # frequency-dependent increment so the sine wave stays continuous across
    # block boundaries even when frequency changes between blocks.
    # Reference: https://gkbrk.com/wiki/PhaseAccumulator
    # DDS architecture background:
    #   https://www.analog.com/en/resources/analog-dialogue/articles/all-about-direct-digital-synthesis.html
    #
    # Lock-free read pattern: the audio callback must never wait on a lock held
    # by another thread, or it causes buffer underruns audible as clicks. We
    # read shared floats directly (GIL makes single-attribute reads effectively
    # atomic in CPython) and accept the rare torn read — control-rate smoothing
    # ensures no jumps the audio cares about.
    # Reference: https://python-sounddevice.readthedocs.io/en/latest/api/misc.html
    #            ("avoid anything that could block the callback function")

    def audio_callback(outdata: np.ndarray, frames: int, _time_info, status) -> None:
        if status:
            print(f"AUDIO STATUS: {status}", file=sys.stderr, flush=True)

        # Lock-free reads
        f = synth.pitch_hz
        timbre = synth.timbre
        detune = synth.detune_hz
        p_fund = synth.fundamental_phase
        p_shad = synth.shadow_phase

        log_counter[0] += 1
        if log_counter[0] % LOG_EVERY == 0:
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

        gain = synth.master_gain
        mix = (fundamental + 0.5 * shadow) / 1.5 * 0.4 * gain
        outdata[:] = mix.reshape(-1, 1)
        
        if synth.record_enabled:
            synth.recording.append(mix.copy())

        # Write phase back without locking — same atomicity argument
        synth.fundamental_phase = (p_fund + dphase_fund * frames) % (2 * np.pi)
        synth.shadow_phase = (p_shad + dphase_shad * frames) % (2 * np.pi)

    return audio_callback

HISTORY_SECONDS = 15
HISTORY_LEN = HISTORY_SECONDS * CONTROL_RATE

history_x: list[float] = [0.0] * HISTORY_LEN
history_y: list[float] = [0.0] * HISTORY_LEN
history_z: list[float] = [0.0] * HISTORY_LEN
history_lock = threading.Lock()

gesture_events: list[tuple[float, str]] = []
gesture_lock = threading.Lock()


def log_gesture(label: str) -> None:
    """Record a gesture event for plotting + console output."""
    now = time.monotonic()
    with gesture_lock:
        gesture_events.append((now, label))
        cutoff = now - HISTORY_SECONDS
        while gesture_events and gesture_events[0][0] < cutoff:
            gesture_events.pop(0)
    print(f"[gesture] {label}", flush=True)


def push_history(x: float, y: float, z: float) -> None:
    """Add a new drift reading to the history buffer."""
    with history_lock:
        history_x.append(x)
        history_y.append(y)
        history_z.append(z)
        if len(history_x) > HISTORY_LEN:
            history_x.pop(0)
            history_y.pop(0)
            history_z.pop(0)


def control_loop(
    synth: SynthState,
    imu: IMU,
    drift_x: DriftAxis,
    drift_y: DriftAxis,
    drift_z: DriftAxis,
    events: queue.Queue[str],
    stop_flag: threading.Event,
) -> None:
    """Run the drift -> synth control loop until stop_flag."""
    next_tick = time.monotonic()

    while not stop_flag.is_set():
        while not events.empty():
            event = events.get_nowait()
            if event == "tap":
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
        
        ax, ay, az = imu.read()
        az_gravity = -GRAVITY_LSB if imu.inverted else GRAVITY_LSB

        if imu.inverted:
            bx = -drift_x.position * INVERT_BIAS
            by = -drift_y.position * INVERT_BIAS
            bz = -drift_z.position * INVERT_BIAS
        else:
            bx = by = bz = 0.0

        drift_x.update(ax, bx)
        drift_y.update(ay, by)
        drift_z.update(az - az_gravity, bz)
        for label, axis in (("X", drift_x), ("Y", drift_y), ("Z", drift_z)):
            if abs(axis.position) > DRIFT_MAX:
                print(
                    f"{label} out of range: pos={axis.position:+.2f}  vel={axis.velocity:+.2f}",
                    flush=True,
                )

        push_history(drift_x.position, drift_y.position, drift_z.position)

        pitch_hz = drift_to_pitch(drift_x.position)
        timbre = drift_to_range(drift_y.position, 0.0, 1.0)
        detune_hz = drift_to_range(drift_z.position, -8.0, 8.0)

        with synth.lock:
            synth.pitch_hz += (pitch_hz - synth.pitch_hz) * SMOOTHING
            synth.timbre += (timbre - synth.timbre) * SMOOTHING
            synth.detune_hz += (detune_hz - synth.detune_hz) * SMOOTHING
        
        next_tick += DT
        sleep_for = next_tick - time.monotonic()
        if sleep_for > 0:
            time.sleep(sleep_for)
        else:
            next_tick = time.monotonic()


def make_keyboard_listener(
    events: queue.Queue[str],
    stop_flag: threading.Event,
) -> keyboard.Listener:
    """Build a pynput listener that translates keypresses into queue events."""

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


def run_plot(stop_flag: threading.Event) -> None:
    """Show a live plot of the drift accumulators."""
    fig, ax = plt.subplots(figsize=(10, 15))
    ax.set_ylim(DRIFT_MIN, DRIFT_MAX)
    ax.set_xlim(0, HISTORY_SECONDS)
    ax.set_xlabel("Seconds (rolling)")
    ax.set_ylabel("Drift Position")
    ax.set_title("Drift Instrument - SPACE: tap    i: invert    q: quit")
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
    
    anim = FuncAnimation(fig, update, interval=33, blit=False, cache_frame_data=False)

    fig.canvas.mpl_connect("close_event", lambda _evt: stop_flag.set())

    plt.show()


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
        name="control"
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
    print("  SPACE - tap/inject momentum")
    print("  i - toggle inverted orientation")
    print("  q - quit")

    control_thread.start()
    listener.start()
    stream.start()

    try:
        run_plot(stop_flag)
    finally:
        stop_flag.set()

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
