## General Path Planning pipeline:
1. A* path planning on the occupancy grid
    - Octile movement & heuristic to allow diagonals
    - Penalty for the change in heading from one movement to the next is included in the heuristic & cost function to encourage straight lines that the Thymio can follow faster, and more accurately
2. Line-of-sight Pruning
    - Using the Bresenham Line algorithm, see what points in the path can be skipped in the path following without colliding with an obstacle.


Parameters to consider playing around with to optimize performance:
- ```TURNING_WEIGHT```: Turning penalty relative to the penalty of the cells close to an obstacle, so that the robot chooses well how closely it passes by obstacles
- ```max_dist```: The maximum number of cells ahead the LOS_pruning can skip, to avoid attempts to aggressively turn the full path finding into one straight line
- ```max_cost```: The maximum acceptable cost for the Bresenham line to cut through and still be considered a viable path, depends on the magnitude of the cell diffusion penalties 