# Straight-line motor calibration

This experiment measures the Thymio's forward speed for equal left and right
motor commands. It calibrates the straight-line part of the motion model; it
does not calibrate turning or the camera.

Use [run_straight_line_experiment.ipynb](./run_straight_line_experiment.ipynb)
to run a trial with both motor targets set to 50. The notebook currently uses
a 2-second duration; measure the distance yourself and record it in `trials.csv`.
The separate `trials_10s.csv` file contains the later 10-second measurements.

## Procedure

1. Use a flat, clear area. Mark a start line and a direction of travel.
2. Measure the robot's starting and ending position using the same reference
   point on the robot. Record forward displacement in meters in `trials.csv`.
   If possible, also record sideways displacement to check that it drove
   straight.
3. Keep the run duration at 2 seconds. Test several safe forward motor command
   levels, using the same command for both motors. Repeat each level at least
   three times.
4. Start from rest for each run, use the same robot orientation, and measure
   the displacement after the motor command stops. Reset to the start line
   between runs.
5. Enter the actual command values sent to the robot and the measured values
   in `trials.csv`. Do not enter example or planned values as measurements.
6. Analyze the data:

   ```powershell
   uv run python sandbox/calibration/straight_line/analyze_straight_line.py
   ```

   To analyze the 10-second measurements without mixing them with the earlier
   trials, pass their CSV explicitly:

   ```powershell
   uv run python sandbox/calibration/straight_line/analyze_straight_line.py --data sandbox/calibration/straight_line/trials_10s.csv
   ```

The script fits forward velocity (distance / duration) against the average of
the left and right motor commands, constrained to pass through zero. It prints
the fitted speed-per-command coefficient and R-squared, and saves
the plot next to the selected CSV. The fitted line is an empirical model for
the tested command range; do not assume it applies outside that range.

If the left and right commands differ, the average alone cannot describe
turning. Keep them equal for this experiment and use a separate turning
calibration for the differential-drive motion model. Log units for motor
commands in the `motor_command_unit` column and keep them consistent.
