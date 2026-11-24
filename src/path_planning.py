from heapq import heappop, heappush
from time import perf_counter

import matplotlib.pyplot as plt
import numpy as np
from data_types import Map, Path, Point, State
from loguru import logger
from matplotlib.patches import Patch
from utils import angle_diff

from map_generation import OBSTACLE_WEIGHT

TURNING_WEIGHT = 5.0 / 180.0
MOVEMENT_WEIGHT = 1.0


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
        ax.imshow(display_grid, cmap="Greys", vmin=0.0, vmax=np.max(display_grid), origin="upper")

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
        ax.scatter(map.start_state.position[1], map.start_state.position[0], c="blue", s=80, label="Start", zorder=3)
        ax.scatter(map.goal_state.position[1], map.goal_state.position[0], c="red", s=80, label="Goal", zorder=3)

        ax.set_title("Occupancy Grid w/ Path")

        handles, labels = ax.get_legend_handles_labels()
        handles.append(Patch(facecolor="black", edgecolor="black", label="Obstacle"))
        ax.legend(handles=handles, loc="upper left", bbox_to_anchor=(1.05, 1), borderaxespad=0.0)

        plt.tight_layout()
        plt.show()

    def _heuristic(self, a: State, b: State) -> float:
        """
        Simple heuristic for A* path planning algorithm, determining Octile distance to the goal

        Args:
         - a (State): Current state in grid
         - b (State): Goal state in grid
        """
        dx = abs(a.position[0] - b.position[0])
        dy = abs(a.position[1] - b.position[1])
        if a.heading is not None and b.heading is not None:
            turning_cost = TURNING_WEIGHT * abs(angle_diff(a.heading, b.heading))
        else:
            turning_cost = 0.0
        return float((dx + dy) + (np.sqrt(2) - 2) * min(dx, dy) + turning_cost)

    def plan(self, map: Map, prune: bool = True, lookahead_distance: int = 5, max_cost: float = 0.0) -> tuple[Path | None, set[Point], int]:
        """
        Run the A* search algorithm on the input Grid,
        start -> goal and return a Path object.
        If prune is True it will post-process optimize the path. 
        """
        start_cell = (int(round(map.start_state.position[0])), int(round(map.start_state.position[1])))
        goal_cell = (int(round(map.goal_state.position[0])), int(round(map.goal_state.position[1])))
        S: State = State(position=start_cell, heading=map.start_state.heading)
        G: State = State(position=goal_cell, heading=map.goal_state.heading)
        came_from: dict[State, State] = {}
        g_costs: dict[Point, float] = {S.position: 0.0}
        explored: set[Point] = set()
        operation_count = 0

        open_set = [(self._heuristic(S, G), 0.0, operation_count, S)]
        current_state: State = S

        if map.grid is None:
            logger.info("Grid object within Map is None.")
            return None, explored, operation_count

        while open_set:
            current_f_cost, current_g_cost, _, current_state = heappop(open_set)
            explored.add(current_state.position)

            if current_state.position == G.position:
                break

            r, c = int(current_state.position[0]), int(current_state.position[1])
            neighbours = [  # Scan all surrounding positions, including diagonally
                # (0 deg = North/Up, 90 deg = East/Right)
                State(position=(r + 1, c), heading=180),  # Down (South)
                State(position=(r - 1, c), heading=0),  # Up (North)
                State(position=(r, c + 1), heading=90),  # Right (East)
                State(position=(r, c - 1), heading=-90),  # Left (West)
                State(position=(r + 1, c + 1), heading=135),  # Down-Right (South-East)
                State(position=(r - 1, c + 1), heading=45),  # Up-Right (North-East)
                State(position=(r + 1, c - 1), heading=-135),  # Down-Left (South-West)
                State(position=(r - 1, c - 1), heading=-45),  # Up-Left (North-West)
            ]

            for neighbour in neighbours:
                nr, nc = int(neighbour.position[0]), int(neighbour.position[1])
                # Check the position is within bounds
                if 0 <= nr < map.dimensions[0] and 0 <= nc < map.dimensions[1]:
                    # Check the position is not an obstacle
                    if map.grid[nr][nc] != OBSTACLE_WEIGHT:
                        if neighbour.heading is not None and current_state.heading is not None:
                            turning_cost = TURNING_WEIGHT * abs(angle_diff(neighbour.heading, current_state.heading))
                        else:
                            turning_cost = 0.0
                        tentative_g_cost = float(current_g_cost + MOVEMENT_WEIGHT + map.grid[nr, nc] + turning_cost)

                        if neighbour.position not in g_costs or g_costs[neighbour.position] > tentative_g_cost:
                            g_costs[neighbour.position] = tentative_g_cost
                            came_from[neighbour] = current_state

                            operation_count += 1

                            f_cost = tentative_g_cost + self._heuristic(neighbour, G)
                            heappush(open_set, (f_cost, tentative_g_cost, operation_count, neighbour))

        if current_state.position == G.position:
            solution: list[Point] = []
            while current_state.position != S.position:
                solution.append(current_state.position)
                current_state = came_from[current_state]
            solution.append(S.position)
            solution.reverse()

            ideal_path = Path(waypoints=solution, cost=g_costs[G.position])
            if self.debug:
                self._display_path(map, ideal_path, explored=explored)

            if prune:
                ideal_path = self._line_of_sight_pruning(map, ideal_path, lookahead_distance, max_cost)
            
            return ideal_path, explored, operation_count
        else:
            logger.info("Path planning could not find a solution.")
            return None, explored, operation_count

    def _bresenham_line(self, a: Point, b: Point) -> Path:
        """
        Given two Point coordinates, use the Bresenham algorithm to
        determine the straight-line Path from one to another

        Args:
         - a: starting coordinate
         - b: ending coordinate
        """
        line: list[Point] = []
        dx = abs(a[0] - b[0])
        sx = 1 if (a[0] < b[0]) else -1
        dy = -abs(a[1] - b[1])
        sy = 1 if (a[1] < b[1]) else -1
        error = dx + dy

        x, y = a[0], a[1]
        while True:
            line.append((x, y))
            e2 = 2 * error
            if e2 >= dy:
                if x == b[0]:
                    break
                error += dy
                x += sx
            if e2 <= dx:
                if y == b[1]:
                    break
                error += dx
                y += sy

        return Path(waypoints=line)

    def _line_of_sight_pruning(self, map: Map, path: Path, lookahead_distance: int, max_cost: float) -> Path:
        """
        Line-of-sight pruning determines what points in the Path can be skipped entirlely for a smaller Path -
        for each point we see what the farthest point that we can see without any obstacles is,
        and skip any points in between.

        lookahead_distance: looks "n" waypoints into the future.
        max_cost: minimum defined threshold to evaluate terrain as candidate
        """
        pruned_points: list[Point] = [path.waypoints[0]]

        i = 0
        start_time = perf_counter()
        while i < len(path.waypoints) - 1:
            waypoint = path.waypoints[i]

            best_j = i + 2 + lookahead_distance
            for j in range(i + 2, min(len(path.waypoints), i + 3 + lookahead_distance)):
                line = self._bresenham_line(waypoint, path.waypoints[j])
                collision = False
                for p in line.waypoints:
                    if map.grid is not None and map.grid[p[0]][p[1]] > max_cost:
                        collision = True
                        break

                if collision:
                    best_j = j - 1
                    break
                elif j == len(path.waypoints) - 1:
                    best_j = len(path.waypoints) - 1
                    break

            # Snap best_j to the maximum value as a precaution
            best_j = best_j if best_j <= (len(path.waypoints) - 1) else (len(path.waypoints) - 1)

            pruned_points.append(path.waypoints[best_j])
            i = best_j

        end_time = perf_counter()
        pruned_path = Path(waypoints=pruned_points, cost=path.cost)
        logger.info(f"Original Path Size = {len(path.waypoints)} Pruned Path Size = {len(pruned_path.waypoints)}")
        logger.info(f"Pruning Duration = {(end_time - start_time):.4f}s")
        logger.debug(f"Prunned Path= {pruned_path}")
        if self.debug:
            self._display_path(map, pruned_path)
        return self._convert_path_from_cells_to_points(pruned_path, map)


    def _convert_path_from_cells_to_points(self, path: Path, grid_map: Map) -> Path:
        """
        Path is currently operated as grid cell positions; converts it to absolute (x, y) coordinates.
        """
        res = grid_map.cell_resolution
        path.waypoints = [((x + 0.5) * res, (y + 0.5) * res) for x, y in path.waypoints]
        return path
