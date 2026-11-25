import numpy as np
from data_types import Command, Path, State
from loguru import logger
from utils import angle_diff


class PIDController:
    """
    PID Controller used to follow the path.
    """

    def __init__(self, lookahead_dist: int = 3, heading_threshold: int = 2, debug: bool = False) -> None:
        """
        Instantiate Path Follower with dbeug flag.
        """
        self._debug: bool = debug
        self._lookahead_dist = lookahead_dist
        self._tgt_waypoint_idx: int = 0
        self._heading_threshold: int = heading_threshold

    def setup(
        self,
        path: Path,
    ) -> None:
        """
        Set the path and gains, not in init in case reset is required, and to reuse same class object.
        """
        self._path = path
        if len(path.waypoints) > 1:
            self._tgt_waypoint_idx = 1

    def reset(self) -> None:
        """
        Restore param to new values and defaults, useful in cases where Path Following has to be paused.
        """
        raise NotImplementedError("TBD")

    def _state_delta(self, current_state: State) -> tuple[float, float]:
        """
        Calculate difference between current state and target position. Returns Euclidean distance and heading difference.
        """
        target_pos = self._path.waypoints[self._tgt_waypoint_idx]
        target_heading = np.atan2((target_pos[0] - current_state.position[0]), (target_pos[1] - current_state.position[1])) * (
            180.0 / np.pi
        )

        logger.debug(f"Current State={current_state} Target Pos={target_pos}, Target Heading={target_heading}")

        delta_pos = np.sqrt((target_pos[0] - current_state.position[0]) ** 2 + (target_pos[1] - current_state.position[1]) ** 2)
        delta_heading = angle_diff(current_state.heading, target_heading)
        if np.abs(delta_heading) < self._heading_threshold:
            delta_heading = 0

        logger.debug(f"Delta Pos={delta_pos} Delta Heading={delta_heading}")
        return delta_pos, delta_heading

    def next(self, current_state: State, dt: float) -> tuple[Command, bool] | None:
        """
        Generate new commands given the current_state, calculates target state and returns the command and True if goal reached.
        """
        if not self._path:
            logger.error("Call set_path() before iterating.")
            return None

        base_speed = 100
        turn_speed = 50
        turn_multiplier = 50.0 / 180.0

        delta_pos, delta_heading = self._state_delta(current_state=current_state)

        if delta_pos <= self._lookahead_dist and self._tgt_waypoint_idx < len(self._path.waypoints) - 1:
            self._tgt_waypoint_idx += 1
            return self.next(current_state=current_state, dt=dt)
        elif delta_pos < 1 and self._tgt_waypoint_idx == len(self._path.waypoints) - 1:
            return Command(), True

        if abs(delta_heading) > 5:
            turn = int(turn_speed + abs(delta_heading) * turn_multiplier)
            d = 1 if delta_heading >= 0 else -1
            return Command(motor_left_speed=d * turn, motor_right_speed=-d * turn), False
        else:
            return Command(motor_left_speed=base_speed, motor_right_speed=base_speed), False
