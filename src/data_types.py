from dataclasses import dataclass, field
import numpy as np

# Type alias or structure representing the occupancy grid from MapGenerator
type Grid = np.ndarray | None
type Point = tuple[int, int]
type Heading = float | None # (Degrees) 0 at north, ranging over (-180, 180]. None values indicate heading is not important
type State = tuple[Point, Heading]

@dataclass
class Map:
    """
    Represents our global Map object, and all potentionally required elements within it.
    """

    grid: Grid = None
    dimensions: tuple[int, int] = (0, 0)
    start_state: State = ((0, 0), 0)
    goal_state: State = ((0, 0), None)

@dataclass
class Path:
    """Represents a planned trajectory through the grid."""

    waypoints: list[Point] = field(default_factory=list)
    cost: float = 0.0