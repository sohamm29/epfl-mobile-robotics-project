from data_types import State, Command, Path
from loguru import logger

class PIDController():
    """
    PID Controller used to follow the path.
    """

    def __init__(self, debug: bool = False) -> None:
        """
        Instantiate Path Follower with dbeug flag.
        """
        self._debug = debug

    def setup(self, 
              path: Path, 
              ) -> None:
        """
        Set the path and gains, not in init in case reset is required, and to reuse same class object.
        """
        self._path = path

    def reset_gains(self) -> None:
        """
        Restore param to new values and defaults, useful in cases where Path Following has to be paused.
        """
        raise NotImplementedError("TBD")
     
    def next(self, current_state: State, dt: float) -> tuple[Command, bool] | None:
        """
        Generate new commands given the current_state, calculates target state and rReturns the command and True if goal reached.
        """
        if not self._path:
            logger.error("Call set_path() before iterating.")
            return None
        return Command(), True