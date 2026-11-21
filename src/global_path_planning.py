from dataclasses import dataclass, field
from heapq import heappush, heappop

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

    def _display_path(self, map: Map, path: Path) -> None:
        """
        Function to display the output Path overlaid on the occupancy Grid.
        """
        pass

    def _heuristic(a: Point, b: Point) -> int:
        """
        Simple heuristic for A* path planning algorithm, determining Euclidean distance to goal
        
        Args:
         - a (Point): Current position in grid
         - b (Point): Goal position in grid
        """
        return ((a[0] - b[0])**2 + (a[1] - b[1])**2)

    def plan(self, map: Map) -> Path | None:
        """
        Run the A* search algorithm on the input Grid,
        start -> goal and return a Path object.
        """
        S = map.start_pos
        G = map.goal_pos
        came_from = {}
        g_costs = {S:0}
        explored = set()
        operation_count = 0

        open_set = {S:(self._heuristic(S, G), 0, S)}

        while open_set:
            current_f_cost, current_g_cost, current_pos = heappop(open_set)
            explored.add(current_pos)

            if current_pos == G:
                break

            neighbours = [ # Scan all surrounding positions, including diagonally 
                (current_pos[0]+1, current_pos[1]), # Right
                (current_pos[0]-1, current_pos[1]), # Left
                (current_pos[0], current_pos[1]+1), # Down
                (current_pos[0], current_pos[1]-1), # Up
                (current_pos[0]+1, current_pos[1]+1), # Down-Right
                (current_pos[0]-1, current_pos[1]+1), # Down-Left
                (current_pos[0]+1, current_pos[1]-1), # Up-Right
                (current_pos[0]-1, current_pos[1]-1), # Down-Right
            ]

            for neighbour in neighbours:
                # Check the position is within bounds
                if 0 <= neighbour[0] < map.dimensions[0] and 0 <= neighbour[1] < map.dimensions[1]:
                    # Check the position is not an obstacle
                    if map.grid[neighbour[0]][neighbour[1]] != 1:
                        tentative_g_cost = current_g_cost + 1 + map.grid[neighbour[0], neighbour[1]]

                        if neighbour not in g_costs or g_costs[neighbour] > tentative_g_cost:
                            g_costs[neighbour] = tentative_g_cost
                            came_from[neighbour] = current_pos

                            operation_count += 1

                            f_cost = tentative_g_cost + self._heuristic(neighbour, G)
                            heappush(open_set, (f_cost, tentative_g_cost, neighbour))
        
        if current_pos == G:
            path = Path()
            ## TODO WORK HERE <--------------------------------

