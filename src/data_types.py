import enum
from dataclasses import dataclass, field

import numpy as np

# Type alias or structure representing the occupancy grid from MapHandler
type Grid = np.ndarray | None
type Point = tuple[float, float]
type Heading = float | None  # (Degrees) 0 at north, ranging over (-180, 180]. None values indicate heading is not important


@dataclass(frozen=True)
class State:
    """
    Represents Robot State at any given instance
    """

    position: Point
    heading: Heading


@dataclass
class ArucoTag:
    """
    Represents data recieved from Aruco Tag Scan
    """

    tid: int  # tag id
    top_left_corner_pos: Point
    top_right_corner_pos: Point
    bottom_left_corner_pos: Point
    bottom_right_corner_pos: Point
    state: State  # uses centroid Point


@dataclass
class Command:
    """
    Represents commands that are sent to the robot.
    """

    motor_left_speed: int = 0
    motor_right_speed: int = 0
    leds_top: tuple[int, int, int] = (0, 0, 0)


@dataclass
class SensorData:
    """
    Represents the suite of data collected from all available sensors required from thr Thymio Robot
    """

    prox_horizontal: list[int]
    prox_ground: list[int]
    motor_speeds: list[int]
    dt: float
    pose_from_camera: State | None


class Mode(enum.StrEnum):
    """
    Represents the possible modes the robot takes
    """

    PATH_FOLLOWING = "path_following"
    LOCAL_AVOIDANCE = "local_avoidance"
    KIDNAPPED = "kidnapped"
    RESTART = "restart"
    GOAL_REACHED = "goal_reached"
    ERROR = "error"


@dataclass
class Map:
    """
    Represents our global Map object, and all potentionally required elements within it.
    """

    grid: Grid = None
    dimensions: tuple[int, int] = (0, 0)
    cell_resolution: int = 0
    start_state: State = State(position=(0, 0), heading=0)
    goal_state: State = State(position=(0, 0), heading=None)


@dataclass
class Path:
    """Represents a planned trajectory through the grid."""

    waypoints: list[Point] = field(default_factory=list)
    cost: float = 0.0
