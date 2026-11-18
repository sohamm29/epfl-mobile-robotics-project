from loguru import logger


class MapGenerator():

    def __init__(self, debug: bool = True):
        """
        Function called on class instantiation, i.e. with a = MapGenerator(). 
        Also handles the interaction with the camera, and image processing.

        Args:
         - debug (bool): This value activates logging and debug interface for deeper understanding and visualization.
        """
        pass

    def _locate_april_tags(self):
        """Within the image, locate the april tags to get orientation, and spacing details of the map."""
        pass

    def _generate_grid(self, cell_resolution_x: int, cell_resolution_y: int) -> Grid:
        """Given the desired resolution, we can build a grid layout."""
        pass

    def _detect_obstacles(self, image) -> Grid:
        """Update the Grid cells with obstacle location."""
        pass

    def _display_map(self) -> None:
        """Function to display output Grid that is generated."""
        pass

    def generate(self) -> Grid:
        """Function to display map after generation."""
        pass
