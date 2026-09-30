"""Regression test: the real nearby-object alert cards (AlertCard, red
border) and the nearest-dynamic-object threat toast must both be laid out
in the left panel -- the combined ADAPTIVE RESOLUTION ZONES (spec + live
objects) box, then the threat toast, then the alert cards -- not floated
as fixed-position overlays on top of the 3D view.

Those floating overlays used to fight the embedded Open3D view's real
native child windows for z-order (see the "hiding behind" fix history in
src/dashboard/qt_main_window.py's alert_container/threat_toast comments),
and could show only a sliver of their red border with content clipped
underneath. Embedding them in the panel's own layout removes that failure
mode entirely, since neither overlaps any native child window anymore.

Needs a real, fully __init__'d FoveaXDashboardWindow (not the __new__()
bypass used elsewhere in this test suite) because the check is about real
Qt layout geometry, not an isolated method's return value.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

pytest.importorskip("PyQt5", reason="PyQt5 not installed; dashboard not runnable in this environment yet")

from src.dashboard.qt_main_window import FoveaXDashboardWindow
from src.tracking.multi_object_tracker import TrackState


@pytest.fixture
def window():
    from PyQt5.QtWidgets import QApplication

    app = QApplication.instance() or QApplication([])
    win = FoveaXDashboardWindow(embed_3d=False)
    win.resize(1600, 1000)
    win.show()
    app.processEvents()
    yield win
    win.close()
    app.processEvents()


def _approaching_pedestrian(track_id=920, dist=15.0):
    return TrackState(
        track_id=track_id,
        class_name="PEDESTRIAN",
        state=np.array([dist, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float64),
        covariance=np.eye(6, dtype=np.float64),
        size_lwh=np.array([0.5, 0.5, 1.8], dtype=np.float64),
        yaw_rad=0.0,
        dynamic=True,
    )


def _left_panel_children(window):
    """(widget, title-or-classname, y, height) for each item in the left
    panel's own QVBoxLayout, in layout order."""
    panel = window.alert_container.parentWidget()
    layout = panel.layout()
    rows = []
    for i in range(layout.count()):
        w = layout.itemAt(i).widget()
        if w is not None:
            title = w.title() if hasattr(w, "title") else type(w).__name__
            rows.append((w, title, w.y(), w.height()))
    return rows


class TestAlertCardIsInLeftPanel:
    def test_alert_container_lives_in_the_left_panel_not_central_widget(self, window):
        """It must be a normal layout child of the left panel, not a
        floating overlay parented directly on central_widget."""
        assert window.alert_container.parentWidget() is not window.centralWidget()

    def test_threat_toast_lives_in_the_left_panel_not_central_widget(self, window):
        assert window.threat_toast.parentWidget() is window.alert_container.parentWidget()

    def test_order_is_zones_then_threat_toast_then_alert_cards(self, window):
        rows = _left_panel_children(window)
        titles = [t for _, t, _, _ in rows]
        assert "ADAPTIVE RESOLUTION ZONES (spec + live objects)" in titles
        zone_idx = titles.index("ADAPTIVE RESOLUTION ZONES (spec + live objects)")
        assert rows[zone_idx + 1][0] is window.threat_toast
        assert rows[zone_idx + 2][0] is window.alert_container

    def test_threat_toast_sits_immediately_below_the_zone_panel(self, window):
        # A hidden widget isn't laid out yet (y stays 0) -- give it real
        # content first, matching how it actually appears on screen.
        window._update_alerts([(15.0, _approaching_pedestrian(), 1.3, "Approaching", "Near (10-25m)")])
        window.lbl_toast_text.setText("DYNAMIC OBJECT: PEDESTRIAN #1 at 10.0 m, 1.0 m/s")
        window.threat_toast.show()
        rows = _left_panel_children(window)
        by_title = {t: (y, h) for _, t, y, h in rows}
        zone_y, zone_h = by_title["ADAPTIVE RESOLUTION ZONES (spec + live objects)"]
        assert window.threat_toast.y() >= zone_y + zone_h

    def test_alert_container_sits_below_the_threat_toast(self, window):
        window._update_alerts([(15.0, _approaching_pedestrian(), 1.3, "Approaching", "Near (10-25m)")])
        window.lbl_toast_text.setText("DYNAMIC OBJECT: PEDESTRIAN #1 at 10.0 m, 1.0 m/s")
        window.threat_toast.show()
        assert window.alert_container.y() >= window.threat_toast.y() + window.threat_toast.height()


class TestThreatToastVisibility:
    def test_shown_with_real_text_when_a_dynamic_object_is_tracked(self, window):
        from src.dashboard.dashboard_state import FrameState

        t = _approaching_pedestrian(920, 40.4)
        window.update_ui(FrameState(points=np.zeros((0, 4), np.float32), tracks=[t]))
        assert not window.threat_toast.isHidden()
        assert "PEDESTRIAN" in window.lbl_toast_text.text()
        assert "#920" in window.lbl_toast_text.text()

    def test_hidden_when_no_dynamic_object_tracked(self, window):
        from src.dashboard.dashboard_state import FrameState

        static_track = _approaching_pedestrian(1, 10.0)
        static_track.dynamic = False
        window.update_ui(FrameState(points=np.zeros((0, 4), np.float32), tracks=[static_track]))
        assert window.threat_toast.isHidden()


class TestAlertCardVisibility:
    def test_shown_with_real_content_when_an_alert_exists(self, window):
        window._update_alerts([(15.0, _approaching_pedestrian(), 1.3, "Approaching", "Near (10-25m)")])
        assert not window.alert_container.isHidden()
        assert window.alert_layout.count() == 1

    def test_hidden_with_no_reserved_space_when_no_alerts(self, window):
        window._update_alerts([(15.0, _approaching_pedestrian(), 1.3, "Approaching", "Near (10-25m)")])
        window._update_alerts([])
        assert window.alert_container.isHidden()

    def test_multiple_real_alerts_all_rendered(self, window):
        alerts = [
            (15.0, _approaching_pedestrian(1, 15.0), 1.3, "Approaching", "Near (10-25m)"),
            (28.0, _approaching_pedestrian(2, 28.0), 0.8, "Approaching", "Middle (25-50m)"),
        ]
        window._update_alerts(alerts)
        assert window.alert_layout.count() == 2

    def test_card_width_fits_panel_not_hardcoded_320(self, window):
        """The old floating-overlay AlertCard forced setFixedWidth(320),
        sized for its old screen position -- now embedded in the (usually
        narrower) left panel column, it must size to that column instead."""
        window._update_alerts([(15.0, _approaching_pedestrian(), 1.3, "Approaching", "Near (10-25m)")])
        card = window.alert_layout.itemAt(0).widget()
        panel_width = window.alert_container.parentWidget().width()
        assert card.width() <= panel_width + 5
