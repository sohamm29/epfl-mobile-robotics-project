from dataclasses import dataclass, field
from loguru import logger
from map_generation import Grid


@dataclass
class Path:
    """Represents a planned trajectory through the grid."""
    waypoints: list[tuple[int, int]] = field(default_factory=list)
    cost: float = 0.0

class PathPlanner:

    def __init__(self, debug: bool = True):
        """
        Function called on class instantiation, i.e. with planner = PathPlanner().
        Handles A* configuration and path computation over a generated Grid.

        Args:
         - debug (bool): Activates logging and debug visualization for the search process and final path.
        """
        self.debug = debug

    def _heuristic(self, node: tuple[int, int], goal: tuple[int, int]) -> float | None:
        """Compute the heuristic cost estimate h(n) between the current node and the goal."""
        pass

    def _display_path(self, grid: Grid, path: Path) -> None:
        """Function to display the output Path overlaid on the occupancy Grid."""
        pass

    def plan(self, grid: Grid, start: tuple[int, int], goal: tuple[int, int]) -> Path | None:
        """Run the A* search algorithm on the input Grid from start to goal and return a Path object."""
        pass
