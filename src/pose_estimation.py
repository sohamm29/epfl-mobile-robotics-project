from data_types import SensorData, State, Point
import yaml
from utils import angle_clamp
import numpy as np


class PoseEstimator:
    """
    Responsible for estimating pose of Robot
    """

    def __init__(self, debug: bool = False, x_pos: float = 0.0, y_pos: float = 0.0, heading: float = 0.0) -> None:
        """
        Instantiate class for Pose Estimation
        """
        self._debug = debug
        self._state = State(position=[x_pos, y_pos], heading=heading)

        with open("config.yaml") as f:
            self._config = yaml.safe_load(f)

    def _display(self) -> None:
        """
        Display the probability distribution of states.
        """
        raise NotImplementedError("TBD")

    def is_kidnapped(self, current_state: State, data: SensorData) -> bool:
        """
        Returns True if detected that robot has been kidnapped (i.e. lifted off the ground.)
        """
        return False

    def _estimate_w(self, left_speed: int, right_speed: int) -> float:
        """
        Estimate the angular velocity of the robot using calibrated values
        """
        return (left_speed - right_speed) * self._config["calibration"]["w_coef"]

    def _estimate_v(self, left_speed: int, right_speed: int) -> float:
        """
        Estimate the linear velocity of the robot using calibrated values
        """
        return 0.5 * (left_speed + right_speed) * self._config["calibration"]["v_coef"]

    def move(self, left_speed: int, right_speed: int, dt: float) -> None:
        """
        Updated probability with motion rules updates.
        """
        # Current method: midpoint method for differential drive odometry
        # May need to integrate exact arc integration in the future - for now, assuming timestep is small enough
        w_est = self._estimate_w(left_speed=left_speed, right_speed=right_speed)
        v_est = self._estimate_v(left_speed=left_speed, right_speed=right_speed)

        new_heading = angle_clamp(self._state.heading + w_est * dt)
        mid_heading = angle_clamp(self._state.heading + 0.5 * w_est * dt)

        new_x = self._state.position[0] + v_est * dt * np.cos(mid_heading)
        new_y = self._state.position[1] + v_est * dt * np.sin(mid_heading)

        self._state = State(position=[new_x, new_y], heading=new_heading)

        return

    def sense(self, data: SensorData) -> None:
        """
        Update probability with sensed data.
        """
        return

    def get_estimate(self) -> State:
        """
        Returns an estimate of the state of where the Robot is.
        If debug flag is true, displays the Probability distribution Grid
        """
        raise self._state
