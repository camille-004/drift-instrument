# Tuning Log

Newer entries on top. Capture what works, what didn't, and why. Keep it short.

---

## 2026-05-11 — First working tune

Sounds musical. Drift wanders, taps are felt, inversion destabilizes (pendulum).

**Working constants:**
- `ACCEL_SCALE = 0.1`, `VEL_DECAY = 0.9995` — drift wanders ±10-20 Hz over 30s
- `TAP_MOMENTUM = 25.0` — ~7x natural drift velocity, felt but not violent
- `INVERT_BIAS = 0.4` — pendulum oscillation, restless not calming
- `SMOOTHING = 0.05` — no zipper noise, no audible step changes
- `PITCH_LO_HZ = 80`, `PITCH_HI_HZ = 320` — log-mapped, two octaves

**Notable misses:**
- Initial `ACCEL_SCALE = 0.00008` → drift effectively static. Bumped 1000x.
- Smoothing written `synth.x = (target - synth.x) * S` (missing `+=`) → pitch decayed to 46 Hz. Found via WAV inspection, not by ear.
- Saw waves for timbre → aliasing buzz. Replaced with sine + 3rd harmonic.
- Three voices (drone/glitch/shimmer) → fought the single-drift concept. Collapsed to one voice across three dimensions.

**Open questions for hardware:**
- Does INVERT_BIAS = 0.4 still feel right when invert is a physical motion?
- Does random-direction tap feel less coherent than physically directed taps?
- Wrap-at-boundary pitch jumps: feature or bug on real hardware?