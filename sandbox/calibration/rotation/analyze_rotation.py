import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


DATA_FILE = Path(__file__).with_name("trials.csv")
PLOT_FILE = Path(__file__).with_name("rotation_fit.png")


def load_trials(path: Path) -> tuple[np.ndarray, np.ndarray]:
    command_differences = []
    angular_velocities = []

    with path.open(newline="", encoding="utf-8") as csv_file:
        reader = csv.DictReader(csv_file)
        required_columns = {
            "left_speed",
            "right_speed",
            "duration_s",
            "heading_change_deg",
        }
        missing_columns = required_columns.difference(reader.fieldnames or [])
        if missing_columns:
            raise ValueError(f"CSV is missing columns: {', '.join(sorted(missing_columns))}")

        for row_number, row in enumerate(reader, start=2):
            try:
                left_speed = float(row["left_speed"])
                right_speed = float(row["right_speed"])
                duration = float(row["duration_s"])
                heading_change_deg = float(row["heading_change_deg"])
            except (TypeError, ValueError) as error:
                raise ValueError(f"Invalid numeric value on CSV row {row_number}") from error

            values = (left_speed, right_speed, duration, heading_change_deg)
            if not all(np.isfinite(value) for value in values):
                raise ValueError(f"Non-finite numeric value on CSV row {row_number}")
            if duration <= 0:
                raise ValueError(f"Duration must be positive on CSV row {row_number}")

            command_differences.append(right_speed - left_speed)
            angular_velocities.append(np.deg2rad(heading_change_deg) / duration)

    if not command_differences:
        raise ValueError("Enter measured rotation trials in trials.csv")
    if not any(difference != 0 for difference in command_differences):
        raise ValueError("At least one trial must have different left and right targets")

    return np.asarray(command_differences), np.asarray(angular_velocities)


def main() -> None:
    command_differences, angular_velocities = load_trials(DATA_FILE)
    coefficient = np.dot(command_differences, angular_velocities) / np.dot(
        command_differences, command_differences
    )
    predicted = coefficient * command_differences
    residual_sum_squares = np.sum((angular_velocities - predicted) ** 2)
    total_sum_squares = np.sum(
        (angular_velocities - np.mean(angular_velocities)) ** 2
    )
    r_squared = (
        1 - residual_sum_squares / total_sum_squares
        if total_sum_squares > 0
        else float("nan")
    )

    print(f"Trials: {len(angular_velocities)}")
    print(
        "Model: omega_rad_s = "
        f"{coefficient:.6g} * (right_target - left_target)"
    )
    print(f"Coefficient: {coefficient:.6g} rad/s per target-difference unit")
    print(
        "Coefficient: "
        f"{np.rad2deg(coefficient):.6g} deg/s per target-difference unit"
    )
    print(f"R-squared: {r_squared:.4f}")
    print("Use this model only over the range of motor commands measured.")

    command_range = np.linspace(
        np.min(command_differences), np.max(command_differences), 100
    )
    plt.scatter(command_differences, angular_velocities, label="Measured trials")
    plt.plot(
        command_range,
        coefficient * command_range,
        label="Linear fit through origin",
    )
    plt.xlabel("Right target - left target")
    plt.ylabel("Angular velocity (rad/s)")
    plt.title(
        "Thymio rotation calibration\n"
        f"Coefficient: {coefficient:.6g} rad/s/unit "
        f"({np.rad2deg(coefficient):.6g} deg/s/unit)"
    )
    plt.gca().secondary_yaxis(
        "right",
        functions=(np.rad2deg, np.deg2rad),
    ).set_ylabel("Angular velocity (deg/s)")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(PLOT_FILE, dpi=150)
    print(f"Plot saved to {PLOT_FILE}")


if __name__ == "__main__":
    main()
