# Drift Instrument

A handheld synthesizer that plays itself through the unavoidable drift of its own sensor errors, where taps and tilts don't control the sound but add new momentum the drift carries forward.

Accelerometers always have tiny measurement errors. Integrate those errors twice (into "velocity," then "position") and they accumulate into imaginary motion. This project uses that drift as the source of sound.

## Running the Prototype

```bash
uv run python -m prototype.main
```

A window shows the drift wandering; audio plays through your default output.

- `SPACE` — tap
- `i` — toggle inversion
- `q` — quit

Each session saves to `drift_session.wav`. Inspect with `tools/inspect_recording.py`.

See `docs/tuning-log.md` for constants and tuning history.