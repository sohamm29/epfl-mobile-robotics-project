import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


DATA_FILE = Path(__file__).with_name("trials.csv")
PLOT_FILE = Path(__file__).with_name("straight_line_fit.png")


def load_trials(path: Path) -> tuple[np.ndarray, np.ndarray]:
    motor_commands = []
    velocities = []

    with path.open(newline="", encoding="utf-8") as csv_file:
        reader = csv.DictReader(csv_file)
        required_columns = {
            "left_speed",
            "right_speed",
            "duration_s",
            "distance_m",
        }
        missing_columns = required_columns.difference(reader.fieldnames or [])
        if missing_columns:
            raise ValueError(f"CSV is missing columns: {', '.join(sorted(missing_columns))}")

        for row_number, row in enumerate(reader, start=2):
            try:
                left_speed = float(row["left_speed"])
                right_speed = float(row["right_speed"])
                duration = float(row["duration_s"])
                distance = float(row["distance_m"])
            except (TypeError, ValueError) as error:
                raise ValueError(f"Invalid numeric value on CSV row {row_number}") from error

            values = (left_speed, right_speed, duration, distance)
            if not all(np.isfinite(value) for value in values):
                raise ValueError(f"Non-finite numeric value on CSV row {row_number}")
            if duration <= 0 or distance < 0:
                raise ValueError(
                    f"Duration must be positive and distance non-negative "
                    f"on CSV row {row_number}"
                )

            motor_commands.append((left_speed + right_speed) / 2)
            velocities.append(distance / duration)

    if len(motor_commands) < 2:
        raise ValueError("Enter at least two measured trials in trials.csv")
    if len(set(motor_commands)) < 2:
        raise ValueError("Trials must include at least two different average motor commands")

    return np.asarray(motor_commands), np.asarray(velocities)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data",
        type=Path,
        default=DATA_FILE,
        help="CSV file with motor commands, durations, and measured distances",
    )
    args = parser.parse_args()
    data_file = args.data
    plot_file = (
        PLOT_FILE
        if data_file == DATA_FILE
        else data_file.with_name(f"{data_file.stem}_fit.png")
    )

    motor_commands, velocities = load_trials(data_file)
    slope = np.dot(motor_commands, velocities) / np.dot(
        motor_commands, motor_commands
    )
    predicted = slope * motor_commands
    residual_sum_squares = np.sum((velocities - predicted) ** 2)
    total_sum_squares = np.sum((velocities - np.mean(velocities)) ** 2)
    r_squared = (
        1 - residual_sum_squares / total_sum_squares
        if total_sum_squares > 0
        else float("nan")
    )
    slope_cm_s = slope * 100

    print(f"Trials: {len(velocities)}")
    print(
        "Model: forward_velocity_cm_s = "
        f"{slope_cm_s:.6g} * average_motor_command"
    )
    print(f"R-squared: {r_squared:.4f}")
    print("Use this model only over the range of motor commands measured.")

    command_range = np.linspace(
        min(0, np.min(motor_commands)), max(0, np.max(motor_commands)), 100
    )
    plt.scatter(motor_commands, velocities * 100, label="Measured trials")
    plt.plot(
        command_range,
        slope_cm_s * command_range,
        label="Linear fit through origin",
    )
    plt.xlabel("Average left/right motor command")
    plt.ylabel("Forward velocity (cm/s)")
    plt.title(
        "Thymio straight-line motor calibration\n"
        f"Coefficient: {slope_cm_s:.6g} cm/s per target unit"
    )
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(plot_file, dpi=150)
    print(f"Plot saved to {plot_file}")


if __name__ == "__main__":
    main()
