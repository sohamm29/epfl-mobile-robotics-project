# Rotation calibration

Record one rotation trial per row in `trials.csv`. Use signed heading change in
degrees, with counterclockwise positive and clockwise negative. Measure the
total rotation during each trial; if using a camera whose heading wraps at
±180°, keep turns below 180° or unwrap the heading measurements.

Run trials in both directions at motor-command differences you expect to use,
and repeat each setting to measure variability. For an in-place turn, use
equal and opposite left/right targets. Make sure the robot has clear space.

After entering measurements, run from the repository root:

```powershell
python sandbox\calibration\rotation\analyze_rotation.py
```

The script computes angular velocity as heading change divided by duration and
fits `omega = coefficient * (right_target - left_target)` through the origin.
it prints the coefficient in rad/s and deg/s per TDM target-difference unit
and saves `rotation_fit.png` with both angular-velocity units shown. Use the
fitted model only over the command range tested.
