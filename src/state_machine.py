from data_types import Command, Map, Mode, SensorData, State
from local_avoidance import LocalAvoider
from loguru import logger
from map_generation import MapGenerator
from path_following_pid import PIDController
from path_planning import PathPlanner
from pose_estimation import PoseEstimator

COMMAND_STOP = Command(motor_left_speed=0, motor_right_speed=0, leds_top=(0, 0, 0))
COMMAND_KIDNAPPED = Command(motor_left_speed=0, motor_right_speed=0, leds_top=(32, 0, 0))
COMMAND_RESTART = Command(motor_left_speed=0, motor_right_speed=0, leds_top=(0, 0, 32))
COMMAND_GOAL_REACHED = Command(motor_left_speed=0, motor_right_speed=0, leds_top=(0, 32, 0))
COMMAND_ERROR = Command(motor_left_speed=0, motor_right_speed=0, leds_top=(32, 0, 32))


class StateMachine:
    """
    Defines the Real-time State Machine used to Control Robot.
    """

    def __init__(
        self,
        map: Map,
        map_generator: MapGenerator,
        path_planner: PathPlanner,
        pose_estimator: PoseEstimator,
        path_follower: PIDController,
        local_avoider: LocalAvoider,
        debug: bool = False,
    ) -> None:
        """
        Instantiate instance of State Machine for Robot Operation.
        """
        self._debug: bool = debug
        self._mode: Mode = Mode.PATH_FOLLOWING
        self._current_state: State = map.start_state

        self._map = map
        self._map_generator: MapGenerator = map_generator
        self._path_planner: PathPlanner = path_planner
        self._path_follower: PIDController = path_follower
        self._pose_estimator: PoseEstimator = pose_estimator
        self._local_avoider: LocalAvoider = local_avoider

    @property
    def mode(self) -> Mode:
        """Current operating mode of the State Machine."""
        return self._mode

    def step(self, sensor_data: SensorData) -> Command:
        """
        Defines what instructions to send to robot.
        """
        logger.info(f"(Pre-step) MODE={self._mode} SENSOR_DATA={sensor_data} CURRENT_STATE={self._current_state}")
        if self._mode in (Mode.PATH_FOLLOWING, Mode.LOCAL_AVOIDANCE):
            left_speed, right_speed = sensor_data.motor_speeds[0], sensor_data.motor_speeds[1]
            self._pose_estimator.move(left_speed, right_speed, sensor_data.dt)
            self._pose_estimator.sense(sensor_data)
            self._current_state = self._pose_estimator.get_estimate()

        kidnapped = self._pose_estimator.is_kidnapped(self._current_state, sensor_data)
        if kidnapped:
            logger.info("Robot is Kidnapped")
            self._mode = Mode.KIDNAPPED
        elif not kidnapped and self._mode == Mode.KIDNAPPED:
            logger.info("Robot no longer Kidnapped, Reset to Path Planner.")
            self._mode = Mode.RESTART
        elif self._mode in (Mode.PATH_FOLLOWING, Mode.LOCAL_AVOIDANCE):
            if self._local_avoider.should_activate(sensor_data):
                logger.info("Robot is entering local avoidance")
                self._mode = Mode.LOCAL_AVOIDANCE
            elif self._mode == Mode.LOCAL_AVOIDANCE:
                logger.info("Robot no longer avoiding, returning to Path Following")
                self._mode = Mode.PATH_FOLLOWING
                self._path_follower.reset()
            logger.info("Robot is Path Following.")

        match self._mode:
            case Mode.KIDNAPPED:
                return COMMAND_KIDNAPPED

            case Mode.RESTART:
                if not sensor_data.pose_from_camera:
                    logger.info("[MODE=RESTART] No Camera Input Available.")
                    return COMMAND_KIDNAPPED

                self._current_state = sensor_data.pose_from_camera
                self._map.start_state = self._current_state
                self._pose_estimator.sense(sensor_data)

                logger.info("[MODE=RESTART] Regenerating Path..")
                path, _, _ = self._path_planner.plan(map=self._map)
                if not path:
                    logger.error("[MODE=RESTART] Unable to generate Path.")
                    self._mode = Mode.ERROR
                    return COMMAND_ERROR

                pruned_new_path = self._path_planner.LOS_pruning(map=self._map, path=path)
                self._path_follower.setup(pruned_new_path)
                self._mode = Mode.PATH_FOLLOWING
                logger.info("[MODE=RESTART] Updated Robot to Path Following")
                return COMMAND_RESTART

            case Mode.LOCAL_AVOIDANCE:
                cmd = self._local_avoider.avoid(self._current_state, sensor_data)
                logger.info(f"[MODE=LOCAL_AVOIDANCE] Avoiding Local Obstacle, cmd={cmd}")
                return cmd

            case Mode.PATH_FOLLOWING:
                step = self._path_follower.next(self._current_state, sensor_data.dt)
                if step:
                    cmd, goal_reached = step
                    if goal_reached:
                        logger.info("[MODE=PATH_FOLLOWING] Goal Reached")
                        self._mode = Mode.GOAL_REACHED
                        return COMMAND_GOAL_REACHED
                    logger.info(f"[MODE=PATH_FOLLOWING] Following Path, cmd={cmd}")
                    return cmd

            case Mode.GOAL_REACHED:
                return COMMAND_GOAL_REACHED

            case Mode.ERROR:
                return COMMAND_ERROR

        logger.error("Unable to match Mode")
        return COMMAND_ERROR
