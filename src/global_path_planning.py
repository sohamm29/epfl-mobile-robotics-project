from dataclasses import dataclass, field
from heapq import heappush, heappop
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Patch

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

    def _display_path(self, map: Map, path: Path, explored: set = None) -> None:
        """
        Function to display the output Path overlaid on the occupancy Grid.
        """
        if map.grid is None:
            return
        
        fig, ax = plt.subplots(figsize=(5, 5))

        display_grid = map.grid.astype(float)
        ax.imshow(display_grid, cmap="Greys", vmin=0.0, vmax=1.0, origin="upper")

        path_rows = [row for row, _ in path.waypoints]
        path_cols = [col for _, col in path.waypoints]
        path_overlay = np.zeros_like(display_grid)
        for row, col in path.waypoints:
            path_overlay[row, col] = 1.0
        if explored:
            for waypoint in explored:
                if path_overlay[waypoint[0], waypoint[1]] != 1.0:
                    path_overlay[waypoint[0], waypoint[1]] = 0.5
        ax.imshow(
            np.ma.masked_equal(path_overlay, 0.0),
            cmap="Greens",
            vmin=0.0,
            vmax=1.0,
            alpha=0.6,
            origin="upper",
        )

        rows, cols = map.dimensions
        ax.set_xticks(np.arange(cols))
        ax.set_yticks(np.arange(rows))
        ax.set_xticks(np.arange(-0.5, cols, 1), minor=True)
        ax.set_yticks(np.arange(-0.5, rows, 1), minor=True)
        ax.grid(which="minor", color="lightgray", linestyle="-", linewidth=0.75)
        ax.tick_params(which="minor", bottom=False, left=False)
        ax.plot(path_cols, path_rows, color="limegreen", linewidth=2, marker="o", label="Path")
        ax.scatter(map.start_pos[1], map.start_pos[0], c="blue", s=80, label="Start", zorder=3)
        ax.scatter(map.goal_pos[1], map.goal_pos[0], c="red", s=80, label="Goal", zorder=3)

        ax.set_title("Occupancy Grid w/ Path")

        handles, labels = ax.get_legend_handles_labels()
        handles.append(Patch(facecolor="black", edgecolor="black", label="Obstacle"))
        ax.legend(
            handles=handles, loc="upper left", bbox_to_anchor=(1.05, 1), borderaxespad=0.0
        )

        plt.tight_layout()
        plt.show()

    def _heuristic(self, a: Point, b: Point) -> int:
        """
        Simple heuristic for A* path planning algorithm, determining Euclidean distance to goal
        
        Args:
         - a (Point): Current position in grid
         - b (Point): Goal position in grid
        """
        return ((a[0] - b[0])**2 + (a[1] - b[1])**2) ** 0.5

    def plan(self, map: Map) -> Path | None:
        """
        Run the A* search algorithm on the input Grid,
        start -> goal and return a Path object.
        """
        S = map.start_pos
        G = map.goal_pos
        came_from = {}
        g_costs = {S: 0}
        explored = set()
        operation_count = 0

        open_set = [(self._heuristic(S, G), 0, S)]

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
            solution: list[Point] = []
            while current_pos != S:
                solution.append(current_pos)
                current_pos = came_from[current_pos]
            solution.append(S)
            solution.reverse()

            ideal_path = Path(waypoints=solution, cost=g_costs[G])
            if self.debug:
                self._display_path(map, ideal_path, explored=explored)
            return ideal_path, explored, operation_count
        else:
            print("Path planning could not find a solution.")
            return None, explored, operation_count

            

