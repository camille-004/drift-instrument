"""Tunable constants for the Drift Instrument.

All instrument behavior knobs live here.
"""

# ---------------------------------------------------------------------------
# Audio + control rates
# ---------------------------------------------------------------------------

SAMPLE_RATE = 44100      # Audio samples per second
BLOCK_SIZE = 512         # Samples per audio callback
CONTROL_RATE = 128       # Drift/synth updates per second
DT = 1.0 / CONTROL_RATE  # Seconds between control ticks

# ---------------------------------------------------------------------------
# Simulated MPU-6050 (±2g range, matches real chip default)
# ---------------------------------------------------------------------------

ACCEL_LSB_PER_G = 16384   # At ±2g, 1g = 16384 raw LSB
GRAVITY_LSB = ACCEL_LSB_PER_G
SENSOR_NOISE_STDDEV = 80  # Approximate noise floor of a real MPU-6050

# ---------------------------------------------------------------------------
# Drift accumulator
# ---------------------------------------------------------------------------

ACCEL_SCALE = 0.1   # Raw accel → drift velocity scale
VEL_DECAY = 0.9995  # Per-tick velocity decay (1.0 = no decay)
DRIFT_MAX = 500.0   # Drift position wraps when exceeded
DRIFT_MIN = -500.0

# ---------------------------------------------------------------------------
# Gestures
# ---------------------------------------------------------------------------

TAP_MOMENTUM = 25.0  # Velocity injected by a tap (~7x natural drift velocity)
INVERT_BIAS = 0.4    # Strength of inversion's pull-toward-zero
SMOOTHING = 0.05     # Per-tick lowpass on synth parameter changes

# ---------------------------------------------------------------------------
# Pitch range (log-mapped)
# ---------------------------------------------------------------------------

PITCH_LO_HZ = 80.0    # ~E2
PITCH_HI_HZ = 320.0   # ~E4

# ---------------------------------------------------------------------------
# Plot
# ---------------------------------------------------------------------------

HISTORY_SECONDS = 15
HISTORY_LEN = HISTORY_SECONDS * CONTROL_RATE
