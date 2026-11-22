from data_types import SensorData, State  

class PoseEstimator():
    """
    Responsible for estimating pose of Robot
    """

    def __init__(self, debug: bool = False) -> None:
        """
        Instantiate class for Pose Estimation
        """
        self._debug = debug


    def _display(self) -> None:
        """
        Display the probability distribution of states.
        """
        raise NotImplementedError("TBD")


    def is_kidnapped(self, current_state: State, data: SensorData) -> bool:
        """
        Returns True if detected that robot has been kidnapped (i.e. lifted off the ground.)
        """
        raise NotImplementedError("TBD")
    
    def move(self, left_speed: int, right_speed: int, dt: float) -> None:
        """
        Updated probability with motion rules updates.
        """
        raise NotImplementedError("TBD")

    def sense(self, data: SensorData) -> None:
        """
        Update probability with sensed data.
        """
        raise NotImplementedError("TBD")

    def get_estimate(self) -> State:
        """
        Returns an estimate of the state of where the Robot is.
        If debug flag is true, displays the Probability distribution Grid
        """
        raise NotImplementedError("TBD")