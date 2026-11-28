# Odometry test

Checks the wheel odometry (`PoseEstimator.move` in `src/pose_estimation.py`)
against a tape measure. Needs the Thymio and TDM, no camera or map.

1. Mark the robot's start position and direction with tape.
2. In [run_odometry_test.ipynb](./run_odometry_test.ipynb), set `PLAN` (a list
   of `(left, right, seconds)` commands) and run the notebook.
3. It prints the predicted end pose: forward distance, sideways distance to the
   right, and heading change (clockwise positive).
4. Measure the real end pose and compare. Repeat a few times per plan.

Suggested plans:

- straight: `[(200, 200, 5.0)]`
- turn in place: `[(150, -150, 1.7)]` (about 90 deg right)
- arc: `[(200, 100, 4.0)]` (forward while turning right)

The average error shows a systematic problem (for example `w_coef` off); the
spread between runs is the noise level to use in the particle filter.
