from data_types import Command, SensorData, State


class LocalAvoider:
    """
    PID Controller used to follow the path.
    """

    def __init__(self, debug: bool = False) -> None:
        """
        Instantiate all the K_p, K_i and K_d Gains and Anti-Windup and Derivative Filtering for Robustness.
        """
        self._debug = debug

    def should_activate(self, data: SensorData) -> bool:
        """
        Returns true if avoidance should be active, and false if no longer necessary
        """
        raise NotImplementedError("TBD")

    def avoid(self, current_state: State, data: SensorData) -> Command:
        """
        Function is responsible of issuing command given current State and Sensor Readings.
        """
        raise NotImplementedError("TBD")
