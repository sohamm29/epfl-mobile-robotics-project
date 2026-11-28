from pathlib import Path

import numpy as np
import yaml
from loguru import logger
from utils import angle_clamp


from data_types import SensorData, State

CONFIG_PATH = Path(__file__).with_name("config.yaml")


def wrap_angle(a: np.ndarray) -> np.ndarray:
    """Wrap angles in degrees to (-180, 180]."""
    return 180.0 - (180.0 - a) % 360.0


class PoseEstimator:
    """
    Particle filter estimating the robot pose [x (cm), y (cm), heading (deg)].

    Frame (same as the camera / map, see utils.get_aruco_tag):
      - x points right (east), y points DOWN (south), like image coordinates
      - heading in degrees, 0 = north (up), clockwise positive, range (-180, 180]
    """

    def __init__(
        self,
        debug: bool = False,
        x_pos: float = 0.0,
        y_pos: float = 0.0,
        heading: float = 0.0,
        seed: int | None = None,
    ) -> None:
        """
        Instantiate the filter with all particles around the given start pose.
        """
        self._debug: bool = debug
        with open(CONFIG_PATH) as f:
            self._config: dict = yaml.safe_load(f)
        self._pf: dict = self._config["particle_filter"]
        self._n: int = self._pf["n_particles"]
        self._rng = np.random.default_rng(seed)
        self._lifted_steps: int = 0
        self.reset(State(position=(x_pos, y_pos), heading=heading))

    @property
    def particles(self) -> np.ndarray:
        """(N, 3) array of particles [x, y, heading] (read-only copy, for plots and tests)."""
        return self._particles.copy()

    def reset(self, state: State) -> None:
        """
        Re-initialise the particle cloud around a known pose (start, or after kidnapping).
        """
        x, y = state.position
        pos_std, head_std = self._pf["init_pos_std_cm"], self._pf["init_heading_std_deg"]
        self._particles = np.column_stack(
            [
                self._rng.normal(x, pos_std, self._n),
                self._rng.normal(y, pos_std, self._n),
                wrap_angle(self._rng.normal(state.heading, head_std, self._n)),
            ]
        )
        self._weights = np.full(self._n, 1.0 / self._n)
        self._draw_scales()
        logger.debug(f"Particle filter reset around {state}")

    def _draw_scales(self) -> None:
        """
        Give each particle its own calibration error (factor on v and w).
        Matches the run-to-run error measured in the odometry tests (sandbox/calibration/odometry).
        """
        self._v_scale = self._rng.normal(1.0, self._pf["v_scale_std"], self._n)
        self._w_scale = self._rng.normal(1.0, self._pf["w_scale_std"], self._n)

    def is_kidnapped(self, current_state: State, data: SensorData) -> bool:
        """
        Returns True while the robot is lifted off the ground: both ground sensors see (almost) nothing
        for a few consecutive steps (so driving over a dark spot of the map doesn't trigger it).
        """
        cfg = self._config["kidnapping"]
        lifted = max(data.prox_ground) < cfg["ground_threshold"]
        self._lifted_steps = self._lifted_steps + 1 if lifted else 0
        return self._lifted_steps >= cfg["min_lifted_steps"]

    def _estimate_w(self, left_speed: int, right_speed: int) -> float:
        """
        Estimate the angular velocity of the robot (deg/s) using calibrated values
        """
        return (left_speed - right_speed) * self._config["calibration"]["w_coef"]

    def _estimate_v(self, left_speed: int, right_speed: int) -> float:
        """
        Estimate the linear velocity of the robot (cm/s) using calibrated values
        """
        return 0.5 * (left_speed + right_speed) * self._config["calibration"]["v_coef"]

    def move(self, left_speed: int, right_speed: int, dt: float) -> None:
        """
        Prediction step: move every particle with the odometry model and its own noise.
        """
        v = self._estimate_v(left_speed, right_speed) * self._v_scale  # (N,) cm/s
        w = self._estimate_w(left_speed, right_speed) * self._w_scale  # (N,) deg/s
        x, y, h = self._particles.T

        # Heading drift when driving (motor imbalance): random walk growing with distance travelled
        drift = self._rng.normal(0.0, self._pf["drift_deg_per_sqrt_cm"] * np.sqrt(np.abs(v) * dt))

        # Midpoint method: use the heading halfway through the step (dx = +sin, dy = -cos, see frame)
        mid = np.radians(h + 0.5 * w * dt)
        self._particles = np.column_stack(
            [
                x + v * dt * np.sin(mid),
                y - v * dt * np.cos(mid),
                wrap_angle(h + w * dt + drift),
            ]
        )

    def _camera_log_likelihood(self, cam: State) -> np.ndarray:
        """
        Measurement model: log p(camera pose | particle) for every particle, as an (N,) array.
        Any constant offset is fine (it cancels when the weights are normalised).

        Placeholder: independent Gaussian noise on position and heading (std from config.yaml).
        To be replaced/refined by the camera model.

        TODO: cam must be in this filter's frame (cm, x/y). MapGenerator.get_robot_state
        currently returns grid cells (row, col), so a conversion is still needed.
        """
        dx = self._particles[:, 0] - cam.position[0]
        dy = self._particles[:, 1] - cam.position[1]
        dh = wrap_angle(self._particles[:, 2] - cam.heading)
        return -0.5 * ((dx**2 + dy**2) / self._pf["camera_pos_std_cm"] ** 2 + dh**2 / self._pf["camera_heading_std_deg"] ** 2)

    def sense(self, data: SensorData) -> None:
        """
        Correction step (Bayes): new weight = old weight x likelihood of the camera pose, then resample if needed.
        Does nothing when the camera is hidden (pose_from_camera is None): the cloud keeps spreading.
        """
        cam = data.pose_from_camera
        if cam is None:
            return

        log_w = self._camera_log_likelihood(cam)
        weights = self._weights * np.exp(log_w - log_w.max())  # subtract max to avoid underflow
        self._weights = weights / weights.sum()

        # Resample only when the weights are uneven (effective sample size below N/2)
        if 1.0 / np.sum(self._weights**2) < self._n / 2:
            self._resample()

    def _resample(self) -> None:
        """
        Systematic resampling, plus a small jitter so duplicated particles don't stay identical.
        """
        # N evenly spaced pointers in [0, 1), shifted by one random offset
        pointers = (self._rng.random() + np.arange(self._n)) / self._n

        # CDF of the weights: particle i owns the interval (cdf[i-1], cdf[i]]
        cdf = np.cumsum(self._weights)
        cdf[-1] = 1.0  # guard against rounding (sum can be 0.9999...)

        # Particle each pointer falls into: heavy particles are picked several times
        idx = np.searchsorted(cdf, pointers)

        pos_std, head_std = self._pf["resample_pos_std_cm"], self._pf["resample_heading_std_deg"]
        self._particles = self._particles[idx] + self._rng.normal(0.0, [pos_std, pos_std, head_std], (self._n, 3))
        self._particles[:, 2] = wrap_angle(self._particles[:, 2])
        self._weights = np.full(self._n, 1.0 / self._n)
        self._draw_scales()  # fresh scales keep the spread realistic for the next camera outage

    def get_estimate(self) -> State:
        """
        Returns the weighted mean pose (circular mean for the heading).
        """
        x, y = self._weights @ self._particles[:, :2]
        h = np.radians(self._particles[:, 2])
        heading = np.degrees(np.arctan2(self._weights @ np.sin(h), self._weights @ np.cos(h)))
        return State(position=(float(x), float(y)), heading=float(heading))

    def _display(self) -> None:
        """
        Plot the particle cloud and the estimate (debug only).
        """
        import matplotlib.pyplot as plt

        est = self.get_estimate()
        _, ax = plt.subplots(figsize=(5, 5))
        ax.scatter(self._particles[:, 0], self._particles[:, 1], s=2, alpha=0.3, label="particles")
        ax.plot(*est.position, "r+", markersize=12, label="estimate")
        ax.invert_yaxis()  # y points down, like the map
        ax.set_aspect("equal")
        ax.set_xlabel("x (cm)")
        ax.set_ylabel("y (cm)")
        ax.legend()
        plt.show()
