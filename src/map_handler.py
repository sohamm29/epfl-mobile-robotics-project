import cv2
import matplotlib.pyplot as plt
import numpy as np
from data_types import Map, Point, State
from loguru import logger
from matplotlib.patches import Patch
from utils import get_aruco_tag

SPACEBAR_KEY_VAL = 32
PX_PER_CELL = 20
OBSTACLE_DETECTION_NOISE_THRESHOLD = 50
DILATION_DECAY_RATE = 0.4
OBSTACLE_WEIGHT = 10


class MapHandler:
    def __init__(self, map_tag_ids: list[int], robot_tag_id: int, debug: bool = False):
        """
        Function called on class instantiation, i.e. with a = MapHandler().
        Also handles the interaction with the camera, and image processing.

        Args:
         - debug (bool): This value activates logging and debug
                         interface for deeper understanding and visualization.
         - robot_tag_id (int): ArUco tag ID mounted on the Thymio robot.
         - map_tag_ids (list[int]): ArUco tag ID's for map delimitation.
        """
        self._debug = debug
        self._map_tag_ids = map_tag_ids
        self._robot_tag_id = robot_tag_id
        self._map: Map = Map()
        self._transform_matrix: np.ndarray | None = None
        self._warped_size: tuple[int, int] = (0, 0)
        self._dilation_mask: np.ndarray | None = None
        self._dilation_margin: int = 0

    def _locate_aruco_tags(self, image: np.ndarray) -> list[Point]:
        """
        Within the image, locate the 4 corner aruco tags to get orientation,
        and spacing details of the map.
        """
        detected_tags = get_aruco_tag(image=image, tag_ids=self._map_tag_ids)

        if not detected_tags:
            logger.error(f"Unable to detect Map tags given= {self._map_tag_ids}")
            return []

        if len(detected_tags) < 4:
            logger.error(f"Detected tags ({[elem.tid for elem in detected_tags]}) less than 4 = {len(detected_tags)}")
            return []

        tags = {tag.tid: tag for tag in detected_tags}
        if not all(i in tags for i in range(4)):
            return []

        # Inner map corner of tag i (0:TL, 1:TR, 2:BR, 3:BL)
        return [
            tags[0].bottom_right_corner_pos,
            tags[1].bottom_left_corner_pos,
            tags[2].top_left_corner_pos,
            tags[3].top_right_corner_pos,
        ]

    def _preprocess_image(self, image: np.ndarray, aruco_tags_loc: list[Point]) -> np.ndarray:
        """
        Returns an image that is already warped, cropped to the map ROI, and blurred.
        """
        rows, cols = self._map.dimensions
        out_w, out_h = cols * PX_PER_CELL, rows * PX_PER_CELL
        self._warped_size = (out_w, out_h)
        src = np.array(aruco_tags_loc, dtype=np.float32)
        dst = np.array([[0, 0], [out_w, 0], [out_w, out_h], [0, out_h]], dtype=np.float32)

        self._transform_matrix = cv2.getPerspectiveTransform(src, dst)
        warped = cv2.warpPerspective(image, self._transform_matrix, (out_w, out_h))
        return cv2.GaussianBlur(warped, (5, 5), 0)

    def _generate_grid(
        self,
        map_width: float,
        map_height: float,
        cell_resolution: int = 5,
    ) -> None:
        """
        Given the desired resolution (in centimetres), build the grid layout.
        """
        cols = int(map_width // cell_resolution)
        rows = int(map_height // cell_resolution)
        self._map.cell_resolution = cell_resolution
        self._map.dimensions = (rows, cols)
        self._map.grid = np.zeros(self._map.dimensions, dtype=float)

    def _find_blob_center(
        self,
        image: np.ndarray,
        lower: tuple[int, int, int],
        upper: tuple[int, int, int],
    ) -> Point:
        """
        Locate the center of the largest color blob within the HSV range
        [lower, upper] and return its corresponding (row, col) grid cell.

        Args:
         - image (np.ndarray): Warped BGR map image.
         - lower (tuple[int, int, int]): Min (Hue, Saturation, Value) threshold
                                         (H: 0-179, S: 0-255, V: 0-255).
         - upper (tuple[int, int, int]): Max (Hue, Saturation, Value) threshold.
        """
        # Convert BGR to HSV to get brightness stability as BGR is affected more by lighting.
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, np.array(lower), np.array(upper))

        # Extract outer contours of all connected white regions in the mask
        # External as we do not care about inner holes
        # Chain approx so we can only keep a few points within straight segments
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return (0, 0)

        # Pick the largest blob by area and compute its image moments
        # (weighted sum of particle values to find mass center)
        largest_blob = max(contours, key=cv2.contourArea)
        m = cv2.moments(largest_blob)
        if m["m00"] == 0:
            return (0, 0)

        # Centroid in pixels: cx = m10 / m00, cy = m01 / m00, and convert to grid coord
        rows, cols = self._map.dimensions
        row = int(np.clip((m["m01"] / m["m00"]) // PX_PER_CELL, 0, rows - 1))
        col = int(np.clip((m["m10"] / m["m00"]) // PX_PER_CELL, 0, cols - 1))
        return (row, col)

    def _detect_obstacles(
        self,
        image: np.ndarray,
        dilation_margin: int = 0,
    ) -> None:
        """
        Update the Grid cells with obstacle (trees, rocks) location and optional
        dilation margin (in number of grid cells).
        """
        if self._map.grid is None:
            return
        rows, cols = self._map.dimensions
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

        green = cv2.inRange(hsv, np.array([35, 40, 30]), np.array([85, 255, 255]))
        grey = cv2.inRange(hsv, np.array([0, 0, 50]), np.array([179, 30, 200]))
        raw_mask = cv2.bitwise_or(green, grey)

        # mask is on original image, resize to col, rows and interpolate between 0 - 255, check
        # >threshold to avoid computer noise detecting false obstacles
        raw_grid = (cv2.resize(raw_mask, (cols, rows), interpolation=cv2.INTER_AREA) > OBSTACLE_DETECTION_NOISE_THRESHOLD).astype(
            np.uint8
        )

        # dilation, add a 1 near any cell that has a one
        self._dilation_margin = dilation_margin
        dilated_grid = raw_grid.astype(float) * OBSTACLE_WEIGHT

        if dilation_margin > 0:
            # distanceTransform measures distance to the nearest 0-pixel,
            # so invert raw_grid so obstacles = 0 and free space = 1
            free_space = (raw_grid == 0).astype(np.uint8)
            dist = cv2.distanceTransform(free_space, cv2.DIST_L2, 5)

            # Mask of cells strictly inside the dilation margin (excluding original obstacles)
            in_margin = (dist > 0) & (dist <= dilation_margin)

            # Exponential decay: tune decay_rate to control how fast the cost drops off
            decay_rate = DILATION_DECAY_RATE
            self._dilation_mask = np.zeros_like(dilated_grid)
            self._dilation_mask[in_margin] = OBSTACLE_WEIGHT * np.exp(-decay_rate * dist[in_margin])

            # Apply the exponential gradient to the grid
            dilated_grid[in_margin] = self._dilation_mask[in_margin]
        else:
            self._dilation_mask = np.zeros_like(dilated_grid)

        self._map.grid = dilated_grid

        if self._debug and dilation_margin > 0:
            logger.info(
                f"Obstacle dilation active: {dilation_margin} cell(s) "
                f"(+{int(np.sum(self._dilation_mask))} dilated grid cells)."
                f"Decay Factor = {DILATION_DECAY_RATE}"
            )

    def _post_process_grid(self) -> None:
        """
        Add a boundary around Grid and update dimensions to match this change.
        """
        if self._map.grid is None:
            return

        # Pad 1 cell on all 4 sides with cost 1
        self._map.grid = np.pad(self._map.grid, pad_width=1, mode="constant", constant_values=OBSTACLE_WEIGHT)
        self._map.dimensions = self._map.grid.shape

        # Shift start, goal, and dilation mask by +1 to match padded coordinates
        sr, sc = self._map.start_state.position
        gr, gc = self._map.goal_state.position

        if self._dilation_mask is not None:
            # update dilation to match padding added to map
            self._dilation_mask = np.pad(self._dilation_mask, pad_width=1, mode="constant", constant_values=0)

    def _display_map(self, raw_image: np.ndarray, warped_image: np.ndarray) -> None:
        """
        Display 3 subplots:
        1. Raw camera image
        2. Warped and preprocessed image
        3. Occupancy grid (showing obstacles and obstacle dilation margin)
        """
        if self._map.grid is None:
            return

        fig, axes = plt.subplots(1, 3, figsize=(15, 5))

        axes[0].imshow(cv2.cvtColor(raw_image, cv2.COLOR_BGR2RGB))
        axes[0].set_title("1. Raw Camera Image")

        axes[1].imshow(cv2.cvtColor(warped_image, cv2.COLOR_BGR2RGB))
        axes[1].set_title("2. Warped & Preprocessed Image")

        # Visual grid: 0 = free (white), 0.45 = dilation (gray), 1 = obstacle (black)
        display_grid = self._map.grid.astype(float)
        axes[2].imshow(display_grid, cmap="Greys", vmin=0.0, vmax=np.max(display_grid), origin="upper")
        rows, cols = self._map.dimensions
        axes[2].set_xticks(np.arange(cols))
        axes[2].set_yticks(np.arange(rows))
        axes[2].set_xticks(np.arange(-0.5, cols, 1), minor=True)
        axes[2].set_yticks(np.arange(-0.5, rows, 1), minor=True)
        axes[2].grid(which="minor", color="lightgray", linestyle="-", linewidth=0.75)
        axes[2].tick_params(which="minor", bottom=False, left=False)
        axes[2].scatter(
            self._map.start_state.position[1],
            self._map.start_state.position[0],
            c="blue",
            s=80,
            label="Start",
        )
        axes[2].scatter(
            self._map.goal_state.position[1],
            self._map.goal_state.position[0],
            c="red",
            s=80,
            label="Goal",
        )
        axes[2].set_title("3. Occupancy Grid")

        handles, labels = axes[2].get_legend_handles_labels()
        handles.append(Patch(facecolor="black", edgecolor="black", label="Obstacle"))
        if self._dilation_margin > 0:
            handles.append(
                Patch(
                    facecolor="gray",
                    edgecolor="black",
                    label=f"Obstacle Dilation (+{self._dilation_margin} cell)",
                )
            )
        axes[2].legend(handles=handles, loc="upper left", bbox_to_anchor=(1.05, 1), borderaxespad=0.0)

        plt.tight_layout()
        plt.show()

    def _detect_fire(self, image: np.ndarray) -> None:
        """
        Detect fire position within Map.
        Fire in HSV: red-orange Hue (0-15), high Saturation (160-255), Value (150-255).
        """
        pos = self._find_blob_center(image, (0, 160, 150), (15, 255, 255))
        self._map.goal_state = State(position=pos, heading=None)

    def _detect_start(self, robot_tag_id: int, frame: np.ndarray) -> None:
        """
        Detect robot start position within Map.
        Falls back to white blob detection, calls public get_robot_state() during generate().
        """
        robot_state = self.get_robot_state(robot_tag_id=robot_tag_id, frame=frame)
        if not robot_state:
            logger.error("Unable to detect Roobt State.")
            return
        self._map.start_state = robot_state

    def get_robot_state(
        self,
        robot_tag_id: int,
        frame: np.ndarray,
    ) -> State | None:
        """
        Continuously locate the robot's ArUco tag from a new camera frame (or image_path)
        and return its State in padded grid coordinates (row, col) and heading (degrees).
        Returns None if the camera is covered or the tag is not visible.
        """
        if self._transform_matrix is None or self._warped_size == (0, 0):
            logger.error("Map perspective transform is not initialized. Call generate() first.")
            return None

        warped = cv2.warpPerspective(frame, self._transform_matrix, self._warped_size)
        robot_tags = get_aruco_tag(image=warped, tag_ids=[robot_tag_id])

        if not robot_tags:
            pos = self._find_blob_center(frame, (0, 0, 200), (179, 55, 255))
            logger.warning("No Robot Tag found. Resorting to White Blob Filtering")
            return State(position=pos, heading=None)

        px_x, px_y = robot_tags[0].state.position
        rows, cols = self._map.dimensions
        row = int(np.clip(px_y // PX_PER_CELL, 0, rows - 1))
        col = int(np.clip(px_x // PX_PER_CELL, 0, cols - 1))
        # Convert warped pixel coordinates to padded grid coordinates (+1 for post-process padding)
        row = (px_y / PX_PER_CELL) + 1.0
        col = (px_x / PX_PER_CELL) + 1.0
        return State(position=(row, col), heading=robot_tags[0].state.heading)

    def generate(
        self,
        cell_resolution: int,
        map_width: float,
        map_height: float,
        frame: np.ndarray,
        dilation_margin: int = 0,
    ) -> Map | None:
        """
        Function to display map after generation.

        Args:
        - cell_resolution (int): Desired cell width/height (in cm)
        - map_width (float): Physical map width (in cm)
        - map_height (float): Physical map height (in cm)
        - dilation_margin (int): Obstacle dilation margin (in number of grid cells)
        - frame (np.ndarray): The image of the map from which we want to generate the map.
        """

        if frame is None:
            logger.error("None object passed for 'image_path'. Please ensure a valid object of type 'str' or 'Path' is passed.")
            return

        aruco_tags_loc = self._locate_aruco_tags(image=frame)
        logger.debug(aruco_tags_loc)
        if len(aruco_tags_loc) < 4:
            logger.error("Failed to detect all 4 ArUco markers.")
            return

        self._generate_grid(
            cell_resolution=cell_resolution,
            map_height=map_height,
            map_width=map_width,
        )

        preproc_and_warped_image = self._preprocess_image(image=frame, aruco_tags_loc=aruco_tags_loc)

        self._detect_obstacles(
            image=preproc_and_warped_image,
            dilation_margin=dilation_margin,
        )

        self._detect_fire(frame)
        self._detect_start(self._robot_tag_id, frame)
        self._post_process_grid()

        if self._debug:
            self._display_map(raw_image=frame, warped_image=preproc_and_warped_image)

        return self._map
