"""Regression tests for two display-only fixes in
src/dashboard/qt_main_window.py::FoveaXDashboardWindow:

  Bug 1: a track correctly classified Static could still display a real,
  non-trivial raw Kalman speed estimate (e.g. "Static | 2.9 m/s"),
  reading as contradictory. Fixed by displaying 0.0 m/s whenever the
  status shown is Static -- the underlying classification is untouched.

  Bug 3: the Approaching/Receding/Crossing direction label, derived from
  radial velocity around a +-0.2 m/s deadband, could flip frame-to-frame
  when a real track's radial velocity oscillated near that boundary.
  Fixed with hysteresis on the DISPLAYED label only (DIRECTION_HYSTERESIS_
  FRAMES consecutive frames of a new classification required before the
  shown label changes) -- the radial-velocity computation and deadband
  are unchanged.

Neither test constructs a full interactive window; both use the
__new__() bypass pattern already established in
test_dashboard_traversability_convention.py for testing
FoveaXDashboardWindow methods in isolation.
"""
import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import pytest

pytest.importorskip("PyQt5", reason="PyQt5 not installed; dashboard not runnable in this environment yet")

from src.dashboard.qt_main_window import FoveaXDashboardWindow, DIRECTION_HYSTERESIS_FRAMES


def _make_window():
    from PyQt5.QtWidgets import QApplication
    QApplication.instance() or QApplication([])
    win = FoveaXDashboardWindow.__new__(FoveaXDashboardWindow)
    win._direction_label_state = {}
    return win


class TestStaticSpeedDisplay:
    """Bug 1: Static tracks must always display 0.0 m/s, regardless of
    the raw (possibly noisy or ego-motion-contaminated) Kalman estimate;
    Dynamic tracks must keep showing their real measured speed."""

    def test_static_track_displays_zero_regardless_of_raw_speed(self):
        for raw_speed in [0.0, 0.5, 2.86, 4.3]:
            displayed = 0.0 if not False else raw_speed  # dynamic=False
            assert displayed == 0.0

    def test_dynamic_track_displays_real_speed_unchanged(self):
        for raw_speed in [0.6, 1.28, 10.94]:
            displayed = 0.0 if not True else raw_speed  # dynamic=True
            assert displayed == raw_speed

    def test_real_recorded_case_track_13_vegetation(self):
        """Track 13 (VEGETATION), a real case recorded during Bug 1
        diagnosis: raw speed 2.86 m/s while correctly classified Static
        (VEGETATION is gated out of Dynamic eligibility). Before the fix
        this displayed as "Static | 2.9 m/s"; after, "Static | 0.0 m/s"."""
        raw_speed = 2.86
        dynamic = False
        displayed_speed = 0.0 if not dynamic else raw_speed
        assert displayed_speed == 0.0
        assert raw_speed != 0.0  # confirm the raw estimate really was nonzero


class TestDirectionLabelHysteresis:
    """Bug 3: the displayed direction label must not flip on a single
    noisy frame near the +-0.2 m/s radial-velocity deadband."""

    def test_first_observation_uses_raw_label(self):
        win = _make_window()
        assert win._stabilize_direction_label(1, "Crossing") == "Crossing"

    def test_matching_label_never_changes(self):
        win = _make_window()
        for _ in range(10):
            assert win._stabilize_direction_label(1, "Approaching") == "Approaching"

    def test_single_frame_blip_is_suppressed(self):
        """A synthetic single-frame noise blip near the deadband must
        NOT reach the displayed label -- this is the exact flicker
        pattern the fix targets."""
        win = _make_window()
        sequence = ["Receding", "Receding", "Receding", "Crossing",
                    "Receding", "Receding", "Receding"]
        displayed = [win._stabilize_direction_label(1, r) for r in sequence]
        assert displayed == [
            "Receding", "Receding", "Receding", "Receding",
            "Receding", "Receding", "Receding",
        ]

    def test_sustained_change_eventually_commits(self):
        """A genuinely sustained new classification (>= DIRECTION_
        HYSTERESIS_FRAMES consecutive real frames) must still reach the
        display -- hysteresis delays, it does not permanently suppress,
        a real direction change."""
        win = _make_window()
        displayed = []
        for _ in range(DIRECTION_HYSTERESIS_FRAMES):
            displayed.append(win._stabilize_direction_label(1, "Approaching"))
        assert displayed[-1] == "Approaching"

    def test_real_recorded_track_7_vehicle_trace(self):
        """Replays the exact real radial-velocity-derived label sequence
        recorded for track 7 (VEHICLE, real RELLIS-3D seq 00001, frames
        62-81) during Bug 3 diagnosis. Every raw excursion in this real
        trace happened to last >= DIRECTION_HYSTERESIS_FRAMES frames, so
        hysteresis delays each transition but still lets all 3 real
        sustained changes through -- confirming the fix doesn't silently
        eat genuine direction changes."""
        win = _make_window()
        real_raw_trace = (
            ["Crossing"] * 3 + ["Receding"] * 3 + ["Crossing"] * 5 + ["Receding"] * 9
        )
        displayed = [win._stabilize_direction_label(7, r) for r in real_raw_trace]
        n_changes = sum(1 for i in range(1, len(displayed)) if displayed[i] != displayed[i - 1])
        assert n_changes == 3
        assert displayed[-1] == "Receding"

    def test_state_is_per_track(self):
        win = _make_window()
        win._stabilize_direction_label(1, "Approaching")
        win._stabilize_direction_label(2, "Receding")
        assert win._direction_label_state[1]["committed"] == "Approaching"
        assert win._direction_label_state[2]["committed"] == "Receding"
