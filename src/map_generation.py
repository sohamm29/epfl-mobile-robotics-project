import numpy as np
import cv2
from loguru import logger
from dataclasses import dataclass, field

# Type alias or structure representing the occupancy grid from MapGenerator
type Grid = np.ndarray | None
type Point = tuple[int, int]


@dataclass
class Map:
    """
    Represents our global Map object, and all potentionally required elements within it. 
    """
    grid: Grid = None
    start_pos: Point = (0,0)
    goal_pos: Point = (0,0)

SPACEBAR_KEY_VAL = 32

class MapGenerator:
    def __init__(self, debug: bool = False):
        """
        Function called on class instantiation, i.e. with a = MapGenerator().
        Also handles the interaction with the camera, and image processing.

        Args:
         - debug (bool): This value activates logging and debug
                         interface for deeper understanding and visualization.
        """
        self._map: Map = Map()
        self._debug = debug

    def _locate_april_tags(self, image: np.ndarray) -> list[Point]:
        """
        Within the image, locate the april tags to get orientation,
        and spacing details of the map.
        """
        return list()

    def _generate_grid(self, april_tags_loc: list[Point], cell_resolution: int = 5) -> None:
        """
        Given the desired resolution (in centimetres), we can build a grid layout using information from the april tags.
        """
        pass

    def _detect_obstacles(self, image: np.ndarray) -> None:
        """
        Update the Grid cells with obstacle location.
        """
        pass

    def _detect_start(self, image: np.ndarray) -> None:
        """
        Detect start position within Map.
        """
        pass

    def _detect_goal(self, image: np.ndarray) -> None:
        """
        Detect goal position within Map.
        """
        pass

    def _display_map(self) -> None:
        """
        Function to display output Grid that is generated.
        """
        pass

    def generate(self, cell_resolution: int) -> Map | None:
        """
        Function to display map after generation.

        Args: 
        - cell_resolution (int): Desired cell width/height (in cm)
        """
        cam = cv2.VideoCapture(0)

        check, frame = cam.read()
        if not check:
            logger.info("Failed to get Camera Frame.")
            return 

        april_tags_loc = self._locate_april_tags(image=frame)
        self._generate_grid(april_tags_loc=april_tags_loc, cell_resolution=cell_resolution)
        self._detect_obstacles(image=frame)
        self._detect_start(image=frame)
        self._detect_goal(image=frame)

        if self._debug == True:
            self._display_map()

        return self._map
