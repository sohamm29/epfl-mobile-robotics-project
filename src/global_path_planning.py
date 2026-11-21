from heapq import heappop, heappush
from data_types import Map, Path, Point, State

import matplotlib.pyplot as plt
import numpy as np
from loguru import logger
from map_generation import Map, Point
from matplotlib.patches import Patch

TURNING_WEIGHT = 1.0 / 180.0

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

    def _display_path(self, map: Map, path: Path, explored: set[Point] | None = None) -> None:
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
        ax.scatter(map.start_state[0][1], map.start_state[0][0], c="blue", s=80, label="Start", zorder=3)
        ax.scatter(map.goal_state[0][1], map.goal_state[0][0], c="red", s=80, label="Goal", zorder=3)

        ax.set_title("Occupancy Grid w/ Path")

        handles, labels = ax.get_legend_handles_labels()
        handles.append(Patch(facecolor="black", edgecolor="black", label="Obstacle"))
        ax.legend(handles=handles, loc="upper left", bbox_to_anchor=(1.05, 1), borderaxespad=0.0)

        plt.tight_layout()
        plt.show()

    def _angle_diff(self, a: float, b: float) -> float:
        """
        Find the absolute, minimum difference between two angles (degrees), wrapping to stay in the range [-180, 180]
        """
        delta = a - b
        if abs(delta) > 180.0:
            return min(abs(delta % 180.0), abs(-delta % 180.0))
        else:
            return abs(delta)

    def _heuristic(self, a: State, b: State) -> int:
        """
        Simple heuristic for A* path planning algorithm, determining Octile distance to the goal

        Args:
         - a (Point): Current position in grid
         - b (Point): Goal position in grid
        """
        dx = abs(a[0][0] - b[0][0])
        dy = abs(a[0][1] - b[0][1])
        if a[1] and b[1]:
            turning_cost = TURNING_WEIGHT * self._angle_diff(a[1], b[1])
        else:
            turning_cost = 0.0
        return (dx + dy) + (np.sqrt(2) - 2) * min(dx, dy) + turning_cost

    def plan(self, map: Map) -> tuple[Path | None, set[Point], int]:
        """
        Run the A* search algorithm on the input Grid,
        start -> goal and return a Path object.
        """
        S: State = map.start_state
        G: State = map.goal_state
        came_from = {}
        g_costs = {S[0]: 0}
        explored: set[Point] = set()
        operation_count = 0

        open_set = [(self._heuristic(S, G), 0, S)]
        current_state: State = S

        if map.grid is None:
            logger.info("Grid object within Map is None.")
            return None, explored, operation_count

        while open_set:
            current_f_cost, current_g_cost, current_state = heappop(open_set)
            explored.add(current_state[0])

            if current_state[0] == G[0]:
                break

            neighbours = [  # Scan all surrounding positions, including diagonally
                ((current_state[0][0] + 1, current_state[0][1]), 90),  # Right
                ((current_state[0][0] - 1, current_state[0][1]), -90),  # Left
                ((current_state[0][0], current_state[0][1] + 1), 180),  # Down
                ((current_state[0][0], current_state[0][1] - 1), 0),  # Up
                ((current_state[0][0] + 1, current_state[0][1] + 1), 135),  # Down-Right
                ((current_state[0][0] - 1, current_state[0][1] + 1), -135),  # Down-Left
                ((current_state[0][0] + 1, current_state[0][1] - 1), 45),  # Up-Right
                ((current_state[0][0] - 1, current_state[0][1] - 1), -45),  # Up-Left
            ]

            for neighbour in neighbours:
                # Check the position is within bounds
                if 0 <= neighbour[0][0] < map.dimensions[0] and 0 <= neighbour[0][1] < map.dimensions[1]:
                    # Check the position is not an obstacle
                    if map.grid[neighbour[0][0]][neighbour[0][1]] != 1:
                        if neighbour[1] and current_state[1]:
                            turning_cost = TURNING_WEIGHT * self._angle_diff(neighbour[1], current_state[1])
                        else:
                            turning_cost = 0.0
                        tentative_g_cost = current_g_cost + 1 + map.grid[neighbour[0][0], neighbour[0][1]] + turning_cost

                        if neighbour[0] not in g_costs or g_costs[neighbour[0]] > tentative_g_cost:
                            g_costs[neighbour[0]] = tentative_g_cost
                            came_from[neighbour] = current_state

                            operation_count += 1

                            f_cost = tentative_g_cost + self._heuristic(neighbour, G)
                            heappush(open_set, (f_cost, tentative_g_cost, neighbour))
        
        if current_state[0] == G[0]:
            solution: list[Point] = []
            while current_state[0] != S[0]:
                solution.append(current_state[0])
                current_state = came_from[current_state]
            solution.append(S[0])
            solution.reverse()

            ideal_path = Path(waypoints=solution, cost=g_costs[G[0]])
            if self.debug:
                self._display_path(map, ideal_path, explored=explored)
            return ideal_path, explored, operation_count
        else:
            logger.info("Path planning could not find a solution.")
            return None, explored, operation_count
