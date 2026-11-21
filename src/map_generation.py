from pathlib import Path
from data_types import Map, Point

import cv2
import matplotlib.pyplot as plt
import numpy as np
from loguru import logger
from matplotlib.patches import Patch

SPACEBAR_KEY_VAL = 32
PX_PER_CELL = 20
OBSTACLE_DETECTION_NOISE_THRESHOLD = 50

class MapGenerator:
    def __init__(self, debug: bool = False):
        """
        Function called on class instantiation, i.e. with a = MapGenerator().
        Also handles the interaction with the camera, and image processing.

        Args:
         - debug (bool): This value activates logging and debug
                         interface for deeper understanding and visualization.
        """
        self._debug = debug
        self._map: Map = Map()
        self._transform_matrix: np.ndarray | None = None
        self._dilation_mask: np.ndarray | None = None
        self._dilation_margin: int = 0

    def _locate_aruco_tags(self, image: np.ndarray) -> list[Point]:
        """
        Within the image, locate the aruco tags to get orientation,
        and spacing details of the map.
        """
        aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_6X6_1000)
        detector = cv2.aruco.ArucoDetector(aruco_dict, cv2.aruco.DetectorParameters())

        # we place tags 0 -> 3 clockwise from top left
        corners, ids, _ = detector.detectMarkers(image)
        if ids is None or len(ids) < 4:
            return []

        tags = {int(i): c[0] for c, i in zip(corners, ids.flatten(), strict=False)}
        # Inner map corner of tag i (0:TL, 1:TR, 2:BR, 3:BL) is corner (i + 2) % 4
        return [tuple(tags[i][(i + 2) % 4].astype(int)) for i in range(4)]

    def _preprocess_image(self, image: np.ndarray, aruco_tags_loc: list[Point]) -> np.ndarray:
        """
        Returns an image that is already warped, cropped to the map ROI, and blurred.
        """
        rows, cols = self._map.dimensions
        out_w, out_h = cols * PX_PER_CELL, rows * PX_PER_CELL
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
        raw_grid = (
            cv2.resize(raw_mask, (cols, rows), interpolation=cv2.INTER_AREA)
            > OBSTACLE_DETECTION_NOISE_THRESHOLD
        ).astype(np.uint8)

        # dilation, add a 1 near any cell that has a one
        self._dilation_margin = dilation_margin
        dilated_grid = (
            cv2.dilate(raw_grid, np.ones((3, 3), np.uint8), iterations=dilation_margin)
            if dilation_margin > 0
            else raw_grid
        )
        dilated_grid = dilated_grid.astype(float)

        # get independent dilation mask, to display seperately on map.
        self._dilation_mask = (dilated_grid - raw_grid).astype(float)
        dilated_grid[self._dilation_mask > 0] = 0.5
        self._map.grid = dilated_grid

        if self._debug and dilation_margin > 0:
            logger.info(
                f"Obstacle dilation active: {dilation_margin} cell(s) "
                f"(+{int(np.sum(self._dilation_mask))} dilated grid cells)."
            )

    def _detect_start(self, image: np.ndarray) -> None:
        """
        Detect position of Thymio Robot within the Map.
        White in HSV: any Hue (0-179), low Saturation (0-55), high Value (200-255).
        """
        pos = self._find_blob_center(image, (0, 0, 200), (179, 55, 255))
        heading = None
        self._map.start_state = (pos, heading)

    def _detect_goal(self, image: np.ndarray) -> None:
        """
        Detect goal (fire) position within Map.
        Fire in HSV: red-orange Hue (0-15), high Saturation (160-255), Value (150-255).
        """
        pos = self._find_blob_center(image, (0, 160, 150), (15, 255, 255))
        heading = None
        self._map.goal_state = (pos, heading)

    def _post_process_grid(self) -> None:
        """
        Add a boundary around Grid and update dimensions to match this change.
        """
        if self._map.grid is None:
            return

        # Pad 1 cell on all 4 sides with cost 1
        self._map.grid = np.pad(self._map.grid, pad_width=1, mode="constant", constant_values=1)
        self._map.dimensions = self._map.grid.shape

        # Shift start, goal, and dilation mask by +1 to match padded coordinates
        self._map.start_state = ((self._map.start_state[0][0] + 1, self._map.start_state[0][1] + 1), self._map.start_state[1])
        self._map.goal_state = ((self._map.goal_state[0][0] + 1, self._map.goal_state[0][1] + 1), self._map.goal_state[1])

        if self._dilation_mask is not None:
            self._dilation_mask = np.pad(self._dilation_mask, pad_width=1, mode="constant", constant_values=0)

    def _get_frame(self, image_path: str | Path | None = None) -> np.ndarray | None:
        """
        Load frame from disk if an image path is provided, otherwise capture from camera.
        """
        if image_path is not None:
            frame = cv2.imread(str(image_path))
            if frame is None:
                logger.error(f"Failed to load image from path: {image_path}")
            return frame

        cam = cv2.VideoCapture(0)
        check, frame = cam.read()
        cam.release()
        if not check:
            logger.error("Failed to get Camera Frame.")
            return None
        return frame

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
        if self._dilation_mask is not None and self._dilation_margin > 0:
            display_grid[self._dilation_mask > 0] = 0.45

        axes[2].imshow(display_grid, cmap="Greys", vmin=0.0, vmax=1.0, origin="upper")
        rows, cols = self._map.dimensions
        axes[2].set_xticks(np.arange(cols))
        axes[2].set_yticks(np.arange(rows))
        axes[2].set_xticks(np.arange(-0.5, cols, 1), minor=True)
        axes[2].set_yticks(np.arange(-0.5, rows, 1), minor=True)
        axes[2].grid(which="minor", color="lightgray", linestyle="-", linewidth=0.75)
        axes[2].tick_params(which="minor", bottom=False, left=False)
        axes[2].scatter(
            self._map.start_state[0][1],
            self._map.start_state[0][0],
            c="blue",
            s=80,
            label="Start",
        )
        axes[2].scatter(self._map.goal_state[0][1], self._map.goal_state[0][0], c="red", s=80, label="Goal")
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

    def generate(
        self,
        cell_resolution: int,
        map_width: float,
        map_height: float,
        dilation_margin: int = 0,
        image_path: str | Path | None = None,
    ) -> Map | None:
        """
        Function to display map after generation.

        Args:
        - cell_resolution (int): Desired cell width/height (in cm)
        - map_width (float): Physical map width (in cm)
        - map_height (float): Physical map height (in cm)
        - dilation_margin (int): Obstacle dilation margin (in number of grid cells)
        - image_path (str | Path | None): Optional path to an image file for offline testing
        """
        frame = self._get_frame(image_path=image_path)
        if frame is None:
            return None

        aruco_tags_loc = self._locate_aruco_tags(image=frame)
        if len(aruco_tags_loc) < 4:
            logger.error("Failed to detect all 4 ArUco markers.")
            return None

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
        self._detect_start(image=preproc_and_warped_image)
        self._detect_goal(image=preproc_and_warped_image)

        self._post_process_grid()

        if self._debug:
            self._display_map(raw_image=frame, warped_image=preproc_and_warped_image)

        return self._map
