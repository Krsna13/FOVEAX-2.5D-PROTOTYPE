"""FOVEAX Phase 8A — 3D multi-object tracking.

Tracking-by-detection with a constant-velocity Kalman filter and
Hungarian assignment (scipy.optimize.linear_sum_assignment) when SciPy
is available, falling back to a greedy closest-pair association otherwise.

Classes
-------
TrackState
    Single tracked object state (position, velocity, covariance, metadata).

MultiObjectTracker
    Update-by-detections tracker implementing the AB3DMOT-style pipeline.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from src.perception.object_detector import Detection3D


# ---------------------------------------------------------------------------
# Tunable defaults
# ---------------------------------------------------------------------------

_TRACKER_DEFAULTS = {
    "process_noise_std": 1.0,        # m/s^2  (acceleration noise)
    "measure_noise_std": 0.5,        # m      (position measurement noise)
    "gating_threshold_m": 3.0,       # Mahalanobis / Euclidean gate
    "max_missed_frames": 5,          # frames before a track is deleted
    "min_hits_for_dynamic": 2,       # observations before speed is considered
    "dynamic_speed_threshold_mps": 0.5,  # m/s above which track is "dynamic"
    "min_detection_confidence": 0.0, # ignore detections below this confidence
    "matching_class_compatible": True,  # only match same class name
}

# Real-world classes that are physically incapable of independent motion --
# never eligible for the "Dynamic" flag regardless of measured speed.
#
# Root-caused (long-playback investigation, 200+ real RELLIS-3D frames):
# this pipeline has no ego-motion compensation (no odometry -- see
# src/perception/ego_motion_estimate.py's module docstring). Over a long
# real playback the vehicle's own real motion (translation and/or
# rotation) is significant, and every real static object's raw detection
# position drifts through the ego-relative sensor frame as a direct,
# sustained, non-noisy side effect -- confirmed by tracing a track with
# 198 hits across 200 frames (thoroughly Kalman-converged, not early-frame
# noise) still showing sustained ~1.1-1.3 m/s "motion", and by an
# independent real ICP displacement measurement confirming nonzero real
# ego motion over the same window. This is NOT the earlier AABB-centroid
# bug (already fixed and independently reverified here) and NOT primarily
# Kalman-convergence noise (a min-hits threshold would not fix a
# 198-hit track) or excessive track churn (this happens on long-lived
# tracks too).
#
# Full ego-motion compensation (real SLAM/odometry) is out of scope for
# this fix. Instead: real-world domain knowledge says vegetation,
# man-made structures, and terrain surfaces cannot move at 1+ m/s (wind-
# blown vegetation sway is on the order of cm/s), so the "Dynamic"
# classification is suppressed for these classes regardless of the
# underlying numeric contamination source. Genuinely mobile real classes
# (VEHICLE, PEDESTRIAN) and unclassified/wildcard detections (where the
# real class is unknown, so motion can't be ruled out) remain eligible.
_STATIC_ONLY_CLASSES = frozenset({
    "DRIVABLE_GROUND", "ROUGH_TERRAIN", "VEGETATION", "BUILDING_WALL",
    "SOLID_OBSTACLE",
})


def _class_can_be_dynamic(class_name: str) -> bool:
    """True unless class_name is a real-world class that cannot move."""
    return class_name not in _STATIC_ONLY_CLASSES


# ---------------------------------------------------------------------------
# TrackState
# ---------------------------------------------------------------------------

@dataclass
class TrackState:
    """State of a single tracked object.

    Attributes
    ----------
    track_id : int
        Unique, persistent track identifier.
    class_name : str
        Object class (e.g. "vehicle", "pedestrian", "cyclist", "unknown_obstacle").
    state : np.ndarray, shape (6,), float64
        State vector [x, y, z, vx, vy, vz] in the tracker's coordinate frame.
    covariance : np.ndarray, shape (6, 6), float64
        State covariance matrix.
    size_lwh : np.ndarray, shape (3,), float32
        Estimated 3-D box size [length, width, height] in metres.
    yaw_rad : float
        Estimated heading in radians (updated from detections when possible).
    age_frames : int
        Total number of frames this track has existed.
    hits : int
        Number of frames in which a detection was associated.
    missed_frames : int
        Consecutive frames without an associated detection.
    confidence : float
        Aggregated track confidence in [0, 1].
    source : str
        Detections source name used to initialise / update this track.
    dynamic : bool
        True when estimated speed exceeds the dynamic threshold and the track
        has at least ``min_hits_for_dynamic`` observations.
    last_timestamp_s : float | None
        Timestamp of the last update (used to compute dt).
    """

    track_id: int
    class_name: str
    state: np.ndarray
    covariance: np.ndarray
    size_lwh: np.ndarray
    yaw_rad: float
    age_frames: int = 0
    hits: int = 0
    missed_frames: int = 0
    confidence: float = 0.0
    source: str = ""
    dynamic: bool = False
    last_timestamp_s: float | None = None
    # Real, per-frame geometric features computed from this detection's own
    # member points (src/perception/terrain_features.py), e.g.
    # {"slope_deg": float, "ground_clearance_m": float, "is_overhang": bool,
    # "is_probable_rock": bool}. Empty when the detector didn't attach any --
    # never fabricated on the tracker's side. Refreshed from the matched
    # detection every frame (not carried over stale from a prior match).
    metadata: dict = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.state.shape != (6,):
            raise ValueError(f"state must have shape (6,), got {self.state.shape}.")
        if self.state.dtype != np.float64:
            raise ValueError(
                f"state must be float64, got {self.state.dtype}."
            )
        if self.covariance.shape != (6, 6):
            raise ValueError(
                f"covariance must have shape (6, 6), got {self.covariance.shape}."
            )
        if self.covariance.dtype != np.float64:
            raise ValueError(
                f"covariance must be float64, got {self.covariance.dtype}."
            )


# ---------------------------------------------------------------------------
# Kalman filter helpers
# ---------------------------------------------------------------------------

def _build_state_transition(dt: float) -> np.ndarray:
    """Constant-velocity state transition matrix for [x, y, z, vx, vy, vz].

    Parameters
    ----------
    dt : float
        Time delta in seconds.

    Returns
    -------
    np.ndarray, shape (6, 6), float64
    """
    F = np.eye(6, dtype=np.float64)
    F[0, 3] = dt
    F[1, 4] = dt
    F[2, 5] = dt
    return F


def _build_process_noise_covariance(dt: float, accel_std: float) -> np.ndarray:
    """Discrete-time process noise covariance for a CV model.

    Q = integral_0^dt F(t) * G * G^T * q * F(t)^T dt
    where G = [0, 0, 0, 1, 1, 1]^T and q = accel_std^2.

    For the CV model this yields the standard form:

        Q[i, i] = q * dt^2           for position rows (i=0,1,2)
        Q[i, i] = q * dt^2           for velocity rows (i=3,4,5)
        Q[i, j] = q * dt             for (pos_i, vel_j) pairs with i=j-3

    At dt=0 all entries are zero (no process noise for a frozen step).

    Parameters
    ----------
    dt : float
        Time delta in seconds.
    accel_std : float
        Acceleration noise standard deviation (m/s^2).

    Returns
    -------
    np.ndarray, shape (6, 6), float64
    """
    q = accel_std * accel_std
    Q = np.zeros((6, 6), dtype=np.float64)
    if dt <= 0.0:
        return Q
    # Position variance
    Q[0, 0] = q * dt * dt
    Q[1, 1] = q * dt * dt
    Q[2, 2] = q * dt * dt
    # Velocity variance
    Q[3, 3] = q * dt * dt
    Q[4, 4] = q * dt * dt
    Q[5, 5] = q * dt * dt
    # Cross terms (position-velocity)
    Q[0, 3] = q * dt
    Q[1, 4] = q * dt
    Q[2, 5] = q * dt
    Q[3, 0] = q * dt
    Q[4, 1] = q * dt
    Q[5, 2] = q * dt
    return Q


def _build_measurement_noise_covariance(meas_std: float) -> np.ndarray:
    """Measurement noise covariance for position-only measurements.

    R = diag(meas_std^2, meas_std^2, meas_std^2, 0, 0, 0)  (only x,y,z measured)
    but our Kalman filter uses only the 3x3 position block in the update step.

    Returns the full 6x6 form for consistency, with zeros on velocity rows.
    """
    R = np.zeros((6, 6), dtype=np.float64)
    R[0, 0] = meas_std * meas_std
    R[1, 1] = meas_std * meas_std
    R[2, 2] = meas_std * meas_std
    return R


def _predict_kf(state: np.ndarray, covariance: np.ndarray, dt: float,
                accel_std: float) -> tuple[np.ndarray, np.ndarray]:
    """Predict one Kalman filter step (time update).

    Parameters
    ----------
    state : np.ndarray, shape (6,), float64
        [x, y, z, vx, vy, vz].
    covariance : np.ndarray, shape (6, 6), float64
    dt : float
        Time delta in seconds.
    accel_std : float
        Acceleration process-noise stddev (m/s^2).

    Returns
    -------
    new_state, new_covariance
    """
    F = _build_state_transition(dt)
    Q = _build_process_noise_covariance(dt, accel_std)

    new_state = F @ state
    new_cov = F @ covariance @ F.T + Q
    return new_state, new_cov


def _update_kf(
    state: np.ndarray,
    covariance: np.ndarray,
    measurement: np.ndarray,
    meas_std: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Kalman filter measurement update for position-only measurements.

    The measurement vector is [x, y, z] (3-D).  The Kalman gain is computed
    using only the position block of the state/covariance.

    Parameters
    ----------
    state : np.ndarray, shape (6,), float64
    covariance : np.ndarray, shape (6, 6), float64
    measurement : np.ndarray, shape (3,), float64
        Measured position [x, y, z].
    meas_std : float
        Measurement noise standard deviation (metres).

    Returns
    -------
    new_state, new_covariance
    """
    # Measurement matrix: projects state to [x, y, z].
    H = np.zeros((3, 6), dtype=np.float64)
    H[0, 0] = 1.0
    H[1, 1] = 1.0
    H[2, 2] = 1.0

    R = np.zeros((3, 3), dtype=np.float64)
    R[0, 0] = meas_std * meas_std
    R[1, 1] = meas_std * meas_std
    R[2, 2] = meas_std * meas_std

    # Innovation covariance: S = H * P * H^T + R
    P = covariance
    S = H @ P @ H.T + R

    # Kalman gain: K = P * H^T * S^{-1}
    try:
        S_inv = np.linalg.inv(S)
    except np.linalg.LinAlgError:
        # Degenerate case: fall back to pseudo-inverse.
        S_inv = np.linalg.pinv(S)

    K = P @ H.T @ S_inv

    # Innovation: y = z - H * x
    innovation = measurement - H @ state

    new_state = state + K @ innovation

    # Covariance update: P = (I - K * H) * P
    I = np.eye(6, dtype=np.float64)
    new_cov = (I - K @ H) @ P

    # Ensure symmetry and positivity (numerical safety).
    new_cov = (new_cov + new_cov.T) / 2.0
    try:
        _ = np.linalg.cholesky(new_cov)
    except np.linalg.LinAlgError:
        # Add a small jitter on the diagonal if not positive definite.
        new_cov += np.eye(6, dtype=np.float64) * 1e-6

    return new_state, new_cov


# ---------------------------------------------------------------------------
# Association helpers
# ---------------------------------------------------------------------------

def _class_compatible(a: str, b: str, compatible: bool) -> bool:
    """Check whether two class names are compatible for matching."""
    if not compatible:
        return True
    # Same class OR one of them is an unclassified / mock wildcard.
    if a == b:
        return True
    wildcards = {"unknown_obstacle", "unclassified", "UNKNOWN"}
    if a in wildcards or b in wildcards:
        return True
    return False


def _euclidean_cost_matrix(
    tracks: list[TrackState],
    detections: list[Detection3D],
) -> np.ndarray:
    """Build a Euclidean distance cost matrix for track-detection association.

    cost[i, j] = ||track.state[:3] - detection.center_xyz||_2

    Parameters
    ----------
    tracks : list[TrackState]
    detections : list[Detection3D]

    Returns
    -------
    np.ndarray, shape (n_tracks, n_detections), float64
    """
    n_t = len(tracks)
    n_d = len(detections)
    cost = np.full((n_t, n_d), np.inf, dtype=np.float64)
    if n_t == 0 or n_d == 0:
        return cost

    track_centers = np.array([t.state[:3] for t in tracks], dtype=np.float64)
    det_centers = np.array([d.center_xyz.astype(np.float64) for d in detections],
                            dtype=np.float64)

    # Expand and compute pairwise distances.
    diff = track_centers[:, None, :] - det_centers[None, :, :]
    cost = np.linalg.norm(diff, axis=2)
    return cost


def _gated_cost_matrix(
    tracks: list[TrackState],
    detections: list[Detection3D],
    gating_threshold_m: float,
    class_compatible: bool,
) -> np.ndarray:
    """Build a gated Euclidean cost matrix.

    Costs above the gating threshold (or between incompatible classes) are set
    to infinity so they are never matched by the Hungarian algorithm.

    Parameters
    ----------
    tracks : list[TrackState]
    detections : list[Detection3D]
    gating_threshold_m : float
    class_compatible : bool

    Returns
    -------
    np.ndarray, shape (n_tracks, n_detections), float64
    """
    cost = _euclidean_cost_matrix(tracks, detections)
    n_t, n_d = cost.shape

    # Apply class compatibility mask.
    if class_compatible:
        for i, t in enumerate(tracks):
            for j, d in enumerate(detections):
                if not _class_compatible(t.class_name, d.class_name, True):
                    cost[i, j] = np.inf

    # Apply gating threshold.
    cost[cost > gating_threshold_m] = np.inf
    return cost


def _hungarian_assign(cost_matrix: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Solve the assignment problem with the Hungarian algorithm.

    Parameters
    ----------
    cost_matrix : np.ndarray, shape (n_tracks, n_detections), float64
        Gated cost matrix (inf = forbidden assignment).

    Returns
    -------
    row_ind, col_ind : np.ndarray
        Indices of matched pairs.
    unmatched_tracks : np.ndarray
        Track indices not matched.
    unmatched_detections : np.ndarray
        Detection indices not matched.
    """
    try:
        from scipy.optimize import linear_sum_assignment
        use_scipy = True
    except ImportError:
        use_scipy = False

    n_t, n_d = cost_matrix.shape

    if n_t == 0 or n_d == 0:
        return (
            np.array([], dtype=int),
            np.array([], dtype=int),
            np.arange(n_t, dtype=int),
            np.arange(n_d, dtype=int),
        )

    if use_scipy:
        # scipy's linear_sum_assignment handles inf entries gracefully,
        # but we must ensure no finite-cost assignment crosses an inf.
        # Replace inf with a large finite number to avoid warnings from
        # scipy on some versions.
        finite_cost = cost_matrix.copy()
        finite_cost[np.isinf(finite_cost)] = 1e12
        row_ind, col_ind = linear_sum_assignment(finite_cost)
        # Filter out assignments that were originally forbidden.
        valid = ~np.isinf(cost_matrix[row_ind, col_ind])
        row_ind = row_ind[valid]
        col_ind = col_ind[valid]
    else:
        # Greedy nearest-neighbour fallback (deterministic: pick smallest
        # cost pair, remove both, repeat).
        row_ind = []
        col_ind = []
        used_rows = set()
        used_cols = set()
        c = cost_matrix.copy()
        for _ in range(min(n_t, n_d)):
            # Find the smallest valid cost.
            flat_idx = np.argmin(c.reshape(-1))
            i = flat_idx // n_d
            j = flat_idx % n_d
            if np.isinf(c[i, j]):
                break
            row_ind.append(i)
            col_ind.append(j)
            used_rows.add(i)
            used_cols.add(j)
            c[i, :] = np.inf
            c[:, j] = np.inf
        row_ind = np.array(row_ind, dtype=int)
        col_ind = np.array(col_ind, dtype=int)

    matched_rows = set(row_ind.tolist())
    matched_cols = set(col_ind.tolist())
    unmatched_tracks = np.array(
        [i for i in range(n_t) if i not in matched_rows],
        dtype=int,
    )
    unmatched_detections = np.array(
        [j for j in range(n_d) if j not in matched_cols],
        dtype=int,
    )
    return row_ind, col_ind, unmatched_tracks, unmatched_detections


# ---------------------------------------------------------------------------
# Track confidence computation
# ---------------------------------------------------------------------------

def _compute_track_confidence(
    detection_confidence: float,
    age_frames: int,
    hits: int,
    missed_frames: int,
    max_missed_frames: int,
) -> float:
    """Aggregate track confidence from detection confidence and track history.

    Confidence rises with hits and falls with missed frames.  A new track
    starts at the detection confidence, then is adjusted by a maturity factor
    that increases with age and hits.

    Parameters
    ----------
    detection_confidence : float
        Confidence of the detection that was just associated (0 if no detection).
    age_frames : int
    hits : int
    missed_frames : int
    max_missed_frames : int

    Returns
    -------
    float in [0, 1]
    """
    # Base confidence from the latest detection.
    base = detection_confidence

    # Maturity bonus: more hits and older age increase confidence.
    # Saturates at 5 hits / 10 frames.
    maturity = float(np.clip((hits / 5.0 + age_frames / 10.0) * 0.1, 0.0, 0.2))

    # Penalty for missed frames: linear drop towards 0 after max_missed_frames.
    if max_missed_frames > 0:
        missed_ratio = missed_frames / max_missed_frames
        missed_penalty = float(np.clip(missed_ratio * 0.5, 0.0, 0.5))
    else:
        missed_penalty = 0.0

    confidence = float(np.clip(base + maturity - missed_penalty, 0.0, 1.0))
    return confidence


# ---------------------------------------------------------------------------
# MultiObjectTracker
# ---------------------------------------------------------------------------

class MultiObjectTracker:
    """Multi-object tracker implementing tracking-by-detection.

    Each call to :meth:`update` performs:

        1. Predict: advance all existing tracks by elapsed time using a
           constant-velocity Kalman filter.
        2. Associate: match current detections to predicted tracks using
           Euclidean centre distance (gated, Hungarian via SciPy when available,
           greedy fallback otherwise).
        3. Update: apply Kalman measurement update to matched tracks; create
           new tracks for unmatched detections.
        4. Delete: remove tracks that have been missed too many frames.
        5. Classify dynamic: mark tracks as dynamic when estimated speed
           exceeds the threshold and the track has enough observations.

    Output is sorted deterministically by track_id.

    Parameters
    ----------
    process_noise_std : float
        Acceleration noise standard deviation (m/s^2).
    measure_noise_std : float
        Measurement (position) noise standard deviation (metres).
    gating_threshold_m : float
        Maximum allowed Euclidean distance between a track and a detection for
        them to be considered a valid match (metres).
    max_missed_frames : int
        Number of consecutive missed frames before a track is deleted.
    min_hits_for_dynamic : int
        Minimum number of hits before a track's speed is considered for the
        dynamic flag.
    dynamic_speed_threshold_mps : float
        Speed (m/s) above which a track is marked as dynamic.
    min_detection_confidence : float
        Detections with confidence below this value are ignored.
    seed : int | None
        Seed for any randomness (not used by the deterministic tracker, but
        accepted for interface consistency).
    """

    _next_track_id: int = 0
    _tracks: list[TrackState] = field(default_factory=list)

    def __init__(
        self,
        process_noise_std: float = _TRACKER_DEFAULTS["process_noise_std"],
        measure_noise_std: float = _TRACKER_DEFAULTS["measure_noise_std"],
        gating_threshold_m: float = _TRACKER_DEFAULTS["gating_threshold_m"],
        max_missed_frames: int = _TRACKER_DEFAULTS["max_missed_frames"],
        min_hits_for_dynamic: int = _TRACKER_DEFAULTS["min_hits_for_dynamic"],
        dynamic_speed_threshold_mps: float = _TRACKER_DEFAULTS[
            "dynamic_speed_threshold_mps"
        ],
        min_detection_confidence: float = _TRACKER_DEFAULTS[
            "min_detection_confidence"
        ],
        seed: int | None = None,
    ) -> None:
        if process_noise_std <= 0.0:
            raise ValueError("process_noise_std must be > 0.")
        if measure_noise_std <= 0.0:
            raise ValueError("measure_noise_std must be > 0.")
        if gating_threshold_m <= 0.0:
            raise ValueError("gating_threshold_m must be > 0.")
        if max_missed_frames < 0:
            raise ValueError("max_missed_frames must be >= 0.")
        if min_hits_for_dynamic < 1:
            raise ValueError("min_hits_for_dynamic must be >= 1.")
        if dynamic_speed_threshold_mps < 0.0:
            raise ValueError("dynamic_speed_threshold_mps must be >= 0.")
        if not (0.0 <= min_detection_confidence <= 1.0):
            raise ValueError("min_detection_confidence must be in [0, 1].")

        self.process_noise_std = process_noise_std
        self.measure_noise_std = measure_noise_std
        self.gating_threshold_m = gating_threshold_m
        self.max_missed_frames = max_missed_frames
        self.min_hits_for_dynamic = min_hits_for_dynamic
        self.dynamic_speed_threshold_mps = dynamic_speed_threshold_mps
        self.min_detection_confidence = min_detection_confidence

        if seed is not None:
            import random
            random.seed(seed)

        self._next_track_id = 0
        self._tracks: list[TrackState] = []

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def tracks(self) -> list[TrackState]:
        """Return a copy of the current active tracks, sorted by track_id."""
        return sorted(self._tracks, key=lambda t: t.track_id)

    @property
    def active_track_ids(self) -> list[int]:
        """Return the track IDs of all currently active tracks."""
        return [t.track_id for t in self.tracks]

    # ------------------------------------------------------------------
    # Public update
    # ------------------------------------------------------------------

    def update(
        self,
        detections: list[Detection3D],
        timestamp_s: float,
    ) -> list[TrackState]:
        """Process one frame of detections and return updated tracks.

        Parameters
        ----------
        detections : list[Detection3D]
            Detections for the current frame.
        timestamp_s : float
            Sensor timestamp in seconds.  Must be >= the previous timestamp
            (monotonic).  The first call may pass any value; subsequent calls
            must not decrease.

        Returns
        -------
        list[TrackState]
            All active tracks after this update, sorted deterministically by
            track_id.
        """
        # Validate timestamp monotonicity.
        if self._tracks and self._tracks[-1].last_timestamp_s is not None:
            last_ts = self._tracks[-1].last_timestamp_s
            if timestamp_s < last_ts:
                warnings.warn(
                    f"Timestamp decreased from {last_ts:.6f} to {timestamp_s:.6f}. "
                    "dt may be negative; tracker behaviour is undefined for "
                    "non-monotonic timestamps.",
                    RuntimeWarning,
                    stacklevel=2,
                )

        # Filter out low-confidence detections.
        filtered = [
            d for d in detections
            if d.confidence >= self.min_detection_confidence
        ]

        # 1. Predict existing tracks.
        for track in self._tracks:
            if track.last_timestamp_s is not None:
                dt = timestamp_s - track.last_timestamp_s
                if dt > 0.0:
                    track.state, track.covariance = _predict_kf(
                        track.state,
                        track.covariance,
                        dt,
                        self.process_noise_std,
                    )
                elif dt < 0.0:
                    dt = 0.0  # clamp negative dt to zero
                # else dt == 0: no prediction needed
            track.last_timestamp_s = timestamp_s

        # 2. Associate detections to predicted tracks.
        cost_matrix = _gated_cost_matrix(
            self._tracks,
            filtered,
            self.gating_threshold_m,
            class_compatible=_TRACKER_DEFAULTS["matching_class_compatible"],
        )
        matched_rows, matched_cols, unmatched_tracks, unmatched_detections = (
            _hungarian_assign(cost_matrix)
        )

        # 3. Update matched tracks.
        for row, col in zip(matched_rows, matched_cols):
            track = self._tracks[row]
            det = filtered[col]

            # Kalman measurement update.
            measurement = det.center_xyz.astype(np.float64)
            track.state, track.covariance = _update_kf(
                track.state,
                track.covariance,
                measurement,
                self.measure_noise_std,
            )

            # Update size and yaw from the detection (running estimate).
            # Simple moving average: blend previous estimate with new detection.
            alpha = 0.3  # weight for new observation
            track.size_lwh = (
                (1.0 - alpha) * track.size_lwh + alpha * det.size_lwh
            ).astype(np.float32)
            # Yaw: blend using circular interpolation.
            track.yaw_rad = _blend_yaw(track.yaw_rad, det.yaw_rad, alpha)

            track.hits += 1
            track.age_frames += 1
            track.confidence = _compute_track_confidence(
                detection_confidence=det.confidence,
                age_frames=track.age_frames,
                hits=track.hits,
                missed_frames=0,
                max_missed_frames=self.max_missed_frames,
            )
            track.source = det.source
            # Refreshed from this frame's real detection, not accumulated --
            # a feature real last frame but absent this frame (e.g. the
            # cluster no longer has enough member points for a plane fit)
            # must not keep showing.
            track.metadata = dict(getattr(det, "metadata", None) or {})

            # Upgrade class_name if track was unclassified or det has a more specific class.
            wildcards = {"unknown_obstacle", "unclassified", "UNKNOWN"}
            if det.class_name not in wildcards:
                track.class_name = det.class_name
            elif track.class_name in wildcards and det.class_name:
                track.class_name = det.class_name

            # Dynamic flag: only after enough observations, and only for
            # classes physically capable of independent motion (see
            # _STATIC_ONLY_CLASSES).
            if track.hits >= self.min_hits_for_dynamic and _class_can_be_dynamic(track.class_name):
                speed = np.linalg.norm(track.state[3:6])
                track.dynamic = speed >= self.dynamic_speed_threshold_mps
            else:
                track.dynamic = False

            # Reset missed counter on a successful match.
            track.missed_frames = 0

        # 4. Handle unmatched tracks (missed frames).
        for row in unmatched_tracks:
            track = self._tracks[row]
            track.missed_frames += 1
            track.age_frames += 1
            # No real detection this frame to derive geometric features
            # from -- clear rather than keep showing a stale value.
            track.metadata = {}
            track.confidence = _compute_track_confidence(
                detection_confidence=0.0,
                age_frames=track.age_frames,
                hits=track.hits,
                missed_frames=track.missed_frames,
                max_missed_frames=self.max_missed_frames,
            )
            # Re-evaluate dynamic: if speed is still high and we have
            # observations, keep the flag.  Otherwise clear it. Never
            # eligible for classes physically incapable of motion.
            if track.hits >= self.min_hits_for_dynamic and _class_can_be_dynamic(track.class_name):
                speed = np.linalg.norm(track.state[3:6])
                track.dynamic = speed >= self.dynamic_speed_threshold_mps
            else:
                track.dynamic = False

        # 5. Create new tracks for unmatched detections.
        for col in unmatched_detections:
            det = filtered[col]
            initial_state = np.zeros(6, dtype=np.float64)
            initial_state[0] = det.center_xyz[0]
            initial_state[1] = det.center_xyz[1]
            initial_state[2] = det.center_xyz[2]
            initial_state[3] = 0.0
            initial_state[4] = 0.0
            initial_state[5] = 0.0

            initial_cov = np.eye(6, dtype=np.float64) * 1.0

            track = TrackState(
                track_id=self._next_track_id,
                class_name=det.class_name,
                state=initial_state,
                covariance=initial_cov,
                size_lwh=det.size_lwh.copy(),
                yaw_rad=det.yaw_rad,
                age_frames=1,
                hits=1,
                missed_frames=0,
                confidence=_compute_track_confidence(
                    detection_confidence=det.confidence,
                    age_frames=1,
                    hits=1,
                    missed_frames=0,
                    max_missed_frames=self.max_missed_frames,
                ),
                source=det.source,
                dynamic=False,
                last_timestamp_s=timestamp_s,
                metadata=dict(getattr(det, "metadata", None) or {}),
            )
            self._next_track_id += 1
            self._tracks.append(track)

        # 6. Delete stale tracks.
        self._tracks = [
            t for t in self._tracks
            if t.missed_frames < self.max_missed_frames
        ]

        # 7. Return sorted tracks.
        return self.tracks

    # ------------------------------------------------------------------
    # Reset / management
    # ------------------------------------------------------------------

    def reset(self) -> None:
        """Clear all tracks and reset the track ID counter."""
        self._tracks.clear()
        self._next_track_id = 0

    def remove_track(self, track_id: int) -> bool:
        """Remove a specific track by ID.

        Parameters
        ----------
        track_id : int

        Returns
        -------
        bool
            True if a track was found and removed.
        """
        for i, t in enumerate(self._tracks):
            if t.track_id == track_id:
                self._tracks.pop(i)
                return True
        return False

    def get_track(self, track_id: int) -> TrackState | None:
        """Return the track with the given ID, or None."""
        for t in self._tracks:
            if t.track_id == track_id:
                return t
        return None


# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------

def _blend_yaw(prev: float, new: float, alpha: float) -> float:
    """Blend two yaw angles (radians) with circular interpolation.

    Parameters
    ----------
    prev : float
    new : float
    alpha : float in [0, 1]

    Returns
    -------
    float
        Interpolated yaw in (-pi, pi].
    """
    # Normalised angular difference in (-pi, pi].
    raw_diff = (new - prev) % (2.0 * np.pi)
    if raw_diff > np.pi:
        diff = raw_diff - 2.0 * np.pi
    else:
        diff = raw_diff
    blended = prev + alpha * diff
    return (blended + np.pi) % (2.0 * np.pi) - np.pi
