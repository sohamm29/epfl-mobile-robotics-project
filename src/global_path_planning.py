from dataclasses import dataclass, field
from map_generation import Map, Point


@dataclass
class Path:
    """Represents a planned trajectory through the grid."""

    waypoints: list[Point] = field(default_factory=list)
    cost: float = 0.0


class PathPlanner:
    def __init__(self, debug: bool = False):
        """
        Function called on class instantiation, i.e. with planner = PathPlanner().
        Handles A* configuration and path computation over a generated Grid.

        Args:
         - debug (bool): Activates logging and debug visualization for the search
                         process and final path.
        """
        self.debug = debug

    def _heuristic(self, node: Point, goal: Point) -> float | None:
        """
        Compute the heuristic cost estimate h(n)
        between the current node and the goal.
        """
        pass

    def _display_path(self, map: Map, path: Path) -> None:
        """
        Function to display the output Path overlaid on the occupancy Grid.
        """
        pass

    def plan(
        self, map: Map
    ) -> Path | None:
        """
        Run the A* search algorithm on the input Grid,
        start -> goal and return a Path object.
        """
        pass
