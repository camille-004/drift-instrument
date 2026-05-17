# Mozzi_MPU6050 (reference sketch)

Adapted from https://github.com/algomusic/Mozzi_MPU6050

Demonstrates non-blocking MPU-6050 reads alongside Mozzi audio synthesis,
using Mozzi's built-in `twi_nonblock` to avoid the Wire.h conflict.

## Status

Verified compiling and running on Arduino Nano (ATmega328P) with Mozzi 2.0.4
on 2026-05-16. The `MozziGuts.h` include is deprecated in Mozzi 2.x but still
works; may need updating to `Mozzi.h` in future Mozzi versions.

Used as the foundation for the Drift Instrument firmware.
