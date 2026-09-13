"""Tests for 3D box class labels (src/dashboard/track_labels.py and their
use in open3d_viewer.py / qt_main_window.py).

Real-data counterpart (renders real RELLIS-3D frames and checks label anchors
against rendered box pixels): tests/verify_track_labels_real_data.py.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.dashboard.track_labels import (
    LABEL_Z_OFFSET_M,
    TrackLabel,
    box_top_center,
    build_track_labels,
    display_class_for_track,
    distance_zone_for_track,
    orthonormalize_extrinsic,
    project_to_screen,
    track_distance_m,
    track_label_text,
    track_status,
    zone_object_counts,
)
from src.tracking.multi_object_tracker import TrackState


def _track(track_id, class_name, xyz, lwh=(4.0, 2.0, 1.5), yaw=0.0, dynamic=False, metadata=None):
    return TrackState(
        track_id=track_id,
        class_name=class_name,
        state=np.array([*xyz, 0.0, 0.0, 0.0], dtype=np.float64),
        covariance=np.eye(6, dtype=np.float64),
        size_lwh=np.array(lwh, dtype=np.float64),
        yaw_rad=yaw,
        dynamic=dynamic,
        metadata=metadata or {},
    )


# Camera at (0,0,-10) looking down +Z with +X right, +Y down.
_K = np.array([[500.0, 0, 320.0], [0, 500.0, 240.0], [0, 0, 1]])
_EXT = np.eye(4)
_EXT[2, 3] = 10.0


class TestLabelText:
    @pytest.mark.parametrize(
        "class_name,dynamic,xyz,expected",
        [
            ("VEHICLE", True, (3.0, 4.0, 0.0), "VEHICLE (Dynamic) - 5.0m"),
            ("PEDESTRIAN", False, (0.0, 0.0, 0.0), "PEDESTRIAN (Static) - 0.0m"),
            ("VEGETATION", False, (6.0, 8.0, 2.0), "VEGETATION (Static) - 10.0m"),
            ("unknown_obstacle", True, (1.0, 0.0, 0.0), "unknown_obstacle (Dynamic) - 1.0m"),
        ],
    )
    def test_text_comes_from_track_fields(self, class_name, dynamic, xyz, expected):
        assert track_label_text(_track(1, class_name, xyz, dynamic=dynamic)) == expected

    def test_text_follows_track_mutation(self):
        """No cached or default text: changing the track changes the label."""
        t = _track(3, "unknown_obstacle", (0, 0, 0))
        assert track_label_text(t) == "unknown_obstacle (Static) - 0.0m"
        t.class_name = "VEHICLE"
        t.dynamic = True
        assert track_label_text(t) == "VEHICLE (Dynamic) - 0.0m"

    def test_status_matches_dynamic_flag(self):
        assert track_status(_track(1, "X", (0, 0, 0), dynamic=True)) == "Dynamic"
        assert track_status(_track(1, "X", (0, 0, 0), dynamic=False)) == "Static"

    def test_text_uses_display_class_not_raw_class_when_metadata_present(self):
        t = _track(1, "SOLID_OBSTACLE", (3.0, 0.0, 0.0), metadata={"is_overhang": True})
        assert track_label_text(t) == "Overhang (SOLID_OBSTACLE) (Static) - 3.0m"


class TestDistance:
    @pytest.mark.parametrize(
        "xyz,expected",
        [((3.0, 4.0, 0.0), 5.0), ((0.0, 0.0, 5.0), 0.0), ((-6.0, 8.0, 2.0), 10.0)],
    )
    def test_2d_radial_distance_ignores_z(self, xyz, expected):
        assert track_distance_m(_track(1, "X", xyz)) == pytest.approx(expected)

    def test_matches_raw_hypot_of_state(self):
        t = _track(1, "X", (7.0, -3.0, 1.5))
        assert track_distance_m(t) == pytest.approx(float(np.hypot(7.0, -3.0)))


class TestDisplayClass:
    """display_class_for_track must derive purely from track.metadata --
    populated only from real per-frame geometry in
    DataStreamerThread._attach_terrain_features -- never invented here."""

    def test_no_metadata_falls_back_to_real_class_name(self):
        t = _track(1, "VEGETATION", (0, 0, 0))
        assert display_class_for_track(t) == "VEGETATION"

    def test_overhang_takes_priority_and_keeps_underlying_class(self):
        t = _track(1, "SOLID_OBSTACLE", (0, 0, 0), metadata={
            "is_overhang": True, "slope_deg": 12.0, "is_probable_rock": True,
        })
        assert display_class_for_track(t) == "Overhang (SOLID_OBSTACLE)"

    def test_slope_shown_when_no_overhang(self):
        t = _track(1, "ROUGH_TERRAIN", (0, 0, 0), metadata={"slope_deg": 14.3})
        assert display_class_for_track(t) == "Slope (14°)"

    def test_rock_heuristic_shown_only_without_overhang_or_slope(self):
        t = _track(1, "unclassified", (0, 0, 0), metadata={"is_probable_rock": True})
        assert display_class_for_track(t) == "Rock (heuristic)"

    def test_false_flags_do_not_trigger_derived_labels(self):
        t = _track(1, "VEGETATION", (0, 0, 0), metadata={
            "is_overhang": False, "is_probable_rock": False,
        })
        assert display_class_for_track(t) == "VEGETATION"

    def test_slope_zero_is_a_real_value_not_falsy_skip(self):
        """slope_deg=0.0 (perfectly flat) must still be shown, not treated
        as absent because it's falsy."""
        t = _track(1, "DRIVABLE_GROUND", (0, 0, 0), metadata={"slope_deg": 0.0})
        assert display_class_for_track(t) == "Slope (0°)"


class TestAnchor:
    def test_top_center_is_center_plus_half_height_plus_offset(self):
        t = _track(1, "VEHICLE", (3.0, -2.0, 0.5), lwh=(4.0, 2.0, 1.6))
        np.testing.assert_allclose(box_top_center(t), [3.0, -2.0, 0.5 + 0.8 + LABEL_Z_OFFSET_M])

    @pytest.mark.parametrize("yaw", [0.0, 0.7, -2.1, np.pi])
    def test_matches_drawn_lineset_top_face(self, yaw):
        pytest.importorskip("open3d")
        from src.dashboard.open3d_viewer import FoveaX3DViewer

        t = _track(1, "VEHICLE", (5.0, 1.0, -0.3), lwh=(3.7, 1.9, 1.4), yaw=yaw)
        lineset = FoveaX3DViewer._create_box_lineset(
            None, size_lwh=t.size_lwh, center_xyz=t.state[:3], yaw_rad=t.yaw_rad, color=[1, 0, 0]
        )
        corners = np.asarray(lineset.points)
        top = corners[4:8]
        assert np.allclose(top[:, 2], corners[:, 2].max())
        np.testing.assert_allclose(box_top_center(t, z_offset_m=0.0), top.mean(axis=0), atol=1e-9)


class TestProjection:
    def test_known_pixel(self):
        u, v, vis = project_to_screen(np.array([[1.0, 0.5, 0.0]]), _K, _EXT, 640, 480)
        assert u[0] == pytest.approx(320 + 500 * 1.0 / 10)
        assert v[0] == pytest.approx(240 + 500 * 0.5 / 10)
        assert vis[0]

    def test_behind_camera_not_visible(self):
        _, _, vis = project_to_screen(np.array([[0.0, 0.0, -20.0]]), _K, _EXT, 640, 480)
        assert not vis[0]

    def test_off_screen_not_visible(self):
        _, _, vis = project_to_screen(np.array([[100.0, 0.0, 0.0]]), _K, _EXT, 640, 480)
        assert not vis[0]

    def test_orthonormalize_fixes_non_perpendicular_up(self):
        """The matrix Open3D returns for the dashboard's perspective preset
        (front=[-.5,-.5,.5], up=[0,0,1]) -- rows 1 and 2 are not orthogonal."""
        raw = np.array([
            [0.70710678, -0.70710678, 0.0, 0.0],
            [0.0, 0.0, -1.0, 28.0],
            [0.57735027, 0.57735027, -0.57735027, 48.49742261],
            [0, 0, 0, 1],
        ])
        fixed = orthonormalize_extrinsic(raw)
        r = fixed[:3, :3]
        np.testing.assert_allclose(r @ r.T, np.eye(3), atol=1e-9)
        eye_raw = -np.linalg.solve(raw[:3, :3], raw[:3, 3])
        eye_fixed = -r.T @ fixed[:3, 3]
        np.testing.assert_allclose(eye_fixed, eye_raw, atol=1e-9)
        # Image "down" must point toward world -Z for an up=+Z camera.
        assert r[1, 2] < 0

    def test_orthonormal_input_unchanged(self):
        np.testing.assert_allclose(orthonormalize_extrinsic(_EXT), _EXT, atol=1e-12)


class TestBuildTrackLabels:
    def test_one_label_per_track_in_order_with_matching_text(self):
        tracks = [
            _track(7, "VEHICLE", (0.0, 0.0, 0.0), dynamic=True),
            _track(2, "VEGETATION", (1.0, 1.0, 0.0)),
            _track(11, "PEDESTRIAN", (-1.0, 0.5, 0.0), dynamic=True),
        ]
        labels = build_track_labels(tracks, _K, _EXT, 640, 480)
        assert [l.track_id for l in labels] == [7, 2, 11]
        assert [l.text for l in labels] == [track_label_text(t) for t in tracks]
        assert [l.dynamic for l in labels] == [True, False, True]

    def test_empty(self):
        assert build_track_labels([], _K, _EXT, 640, 480) == []

    def test_label_position_is_projected_anchor(self):
        t = _track(1, "VEHICLE", (1.0, -0.5, 0.0), lwh=(4, 2, 2))
        (lab,) = build_track_labels([t], _K, _EXT, 640, 480)
        u, v, _ = project_to_screen(box_top_center(t)[None], _K, _EXT, 640, 480)
        assert (lab.u, lab.v) == (pytest.approx(u[0]), pytest.approx(v[0]))


def _hidden_viewer(width=640, height=480):
    o3d = pytest.importorskip("open3d")
    from src.dashboard.open3d_viewer import FoveaX3DViewer

    try:
        return o3d, FoveaX3DViewer(width=width, height=height, visible=False)
    except Exception as exc:  # no OpenGL context available
        pytest.skip(f"Open3D window unavailable: {exc}")


class TestOpen3DIntegration:
    def test_projection_matches_rendered_pixels_on_dashboard_presets(self):
        from scipy import ndimage

        o3d, viewer = _hidden_viewer()
        try:
            vis = viewer.vis
            vis.remove_geometry(viewer.axis)
            vis.get_render_option().background_color = np.zeros(3)
            extent = o3d.geometry.LineSet(
                points=o3d.utility.Vector3dVector([[-20, -20, 0], [20, 20, 3]]),
                lines=o3d.utility.Vector2iVector([[0, 1]]),
            )
            extent.colors = o3d.utility.Vector3dVector([[0, 0, 0]])
            vis.add_geometry(extent)
            targets = np.array([[6.0, -4.0, 2.0], [-8.0, 5.0, 1.0]])
            for p in targets:
                s = o3d.geometry.TriangleMesh.create_sphere(radius=0.35)
                s.translate(p)
                s.paint_uniform_color([1, 1, 1])
                vis.add_geometry(s)

            for preset in ("reset", "view_perspective", "view_top", "view_side"):
                viewer.handle_camera_command(preset)
                for _ in range(3):
                    vis.poll_events()
                    vis.update_renderer()
                img = np.asarray(vis.capture_screen_float_buffer(do_render=True))
                params = viewer.view_control.convert_to_pinhole_camera_parameters()
                u, v, ok = project_to_screen(
                    targets, params.intrinsic.intrinsic_matrix, params.extrinsic, 640, 480
                )
                labels, n = ndimage.label(img.mean(axis=2) > 0.2)
                cents = np.array(ndimage.center_of_mass(labels > 0, labels, range(1, n + 1)))[:, ::-1]
                for i in range(len(targets)):
                    if not ok[i]:
                        continue
                    err = np.hypot(cents[:, 0] - u[i], cents[:, 1] - v[i]).min()
                    assert err < 3.0, f"{preset}: target {i} projected {err:.2f}px from render"
        finally:
            viewer.destroy()

    def test_viewer_labels_track_drawn_boxes(self):
        from src.dashboard.dashboard_state import FrameState

        _, viewer = _hidden_viewer()
        try:
            rng = np.random.default_rng(0)
            pts = np.hstack([rng.uniform(-15, 15, (500, 3)), np.zeros((500, 1))]).astype(np.float32)
            tracks = [
                _track(4, "VEHICLE", (5.0, 2.0, 0.0), dynamic=True),
                _track(9, "PEDESTRIAN", (-3.0, 4.0, 0.0)),
            ]
            viewer.update(FrameState(points=pts, tracks=tracks))
            labels = viewer.compute_track_labels()
            assert {l.track_id for l in labels} == set(viewer.track_boxes) == {4, 9}
            assert {l.track_id: l.text for l in labels} == {
                4: "VEHICLE (Dynamic) - 5.4m", 9: "PEDESTRIAN (Static) - 5.0m"
            }

            # A track that disappears loses both its box and its label.
            viewer.update(FrameState(points=pts, tracks=tracks[:1]))
            labels = viewer.compute_track_labels()
            assert [l.track_id for l in labels] == [4] and set(viewer.track_boxes) == {4}

            # An empty frame draws nothing new, so labels stay with the drawn boxes.
            viewer.update(FrameState(points=np.zeros((0, 4), np.float32), tracks=[]))
            assert [l.track_id for l in viewer.compute_track_labels()] == [4]
        finally:
            viewer.destroy()


class TestQtOverlay:
    @pytest.fixture
    def widget(self):
        pytest.importorskip("PyQt5")
        from PyQt5.QtCore import Qt
        from PyQt5.QtWidgets import QApplication

        from src.dashboard.qt_main_window import Open3DEmbedWidget

        app = QApplication.instance() or QApplication([])
        w = Open3DEmbedWidget()
        w.setAttribute(Qt.WA_DontShowOnScreen, True)
        w.resize(640, 480)
        w.show()
        yield w
        w.close()
        app.processEvents()

    @staticmethod
    def _label(track_id, text, u=320.0, v=240.0, visible=True, dynamic=False):
        return TrackLabel(track_id, text, u, v, visible, dynamic, 640, 480)

    def test_every_visible_label_shown_with_exact_text(self, widget):
        widget.set_track_labels([
            self._label(1, "VEHICLE (Dynamic)", dynamic=True),
            self._label(2, "VEGETATION (Static)", u=100, v=100),
        ])
        assert widget.visible_label_texts() == {1: "VEHICLE (Dynamic)", 2: "VEGETATION (Static)"}

    def test_label_sits_centered_above_anchor(self, widget):
        widget.set_track_labels([self._label(1, "VEHICLE (Static)", u=300, v=200)])
        lbl = widget._label_widgets[1]
        assert abs((lbl.x() + lbl.width() / 2) - 300) <= 1
        assert lbl.y() + lbl.height() <= 200

    def test_off_screen_label_hidden(self, widget):
        widget.set_track_labels([self._label(1, "VEHICLE (Static)", visible=False)])
        assert widget.visible_label_texts() == {}

    def test_removed_track_label_is_removed(self, widget):
        widget.set_track_labels([self._label(1, "A (Static)"), self._label(2, "B (Static)")])
        widget.set_track_labels([self._label(2, "B (Static)")])
        assert set(widget._label_widgets) == {2}
        assert widget.visible_label_texts() == {2: "B (Static)"}

    def test_text_updates_when_track_status_changes(self, widget):
        widget.set_track_labels([self._label(5, "VEHICLE (Static)")])
        widget.set_track_labels([self._label(5, "VEHICLE (Dynamic)", dynamic=True)])
        assert widget.visible_label_texts() == {5: "VEHICLE (Dynamic)"}

    def test_scales_to_panel_when_framebuffer_size_differs(self, widget):
        lab = TrackLabel(1, "X (Static)", 640.0, 480.0, True, False, 1280, 960)
        widget.set_track_labels([lab])
        lbl = widget._label_widgets[1]
        assert abs((lbl.x() + lbl.width() / 2) - 320) <= 1


class TestDistanceZones:
    """Boundaries must match src/10b_eval_distance_metrics.py's real
    Near/Mid/Far buckets exactly (0-15m, 15-35m, 35-100m), and the
    OBJECTS PER ZONE panel must read the same real distance
    (track_distance_m) the 3D label and table already display."""

    @pytest.mark.parametrize(
        "dist,expected_zone",
        [
            (0.0, "Near"), (14.99, "Near"),
            (15.0, "Middle"), (34.99, "Middle"),
            (35.0, "Far"), (100.0, "Far"),
            (100.01, None), (500.0, None),
        ],
    )
    def test_zone_boundaries_match_eval_distance_metrics(self, dist, expected_zone):
        t = _track(1, "X", (dist, 0.0, 0.0))
        assert distance_zone_for_track(t) == expected_zone

    def test_zone_uses_real_track_distance_not_a_separate_computation(self):
        t = _track(1, "X", (6.0, 8.0, 0.0))  # hypot = 10.0 -> Near
        assert track_distance_m(t) == pytest.approx(10.0)
        assert distance_zone_for_track(t) == "Near"

    def test_counts_real_tracks_per_zone(self):
        tracks = [
            _track(1, "VEGETATION", (5.0, 0.0, 0.0)),    # Near
            _track(2, "VEHICLE", (10.0, 0.0, 0.0)),      # Near
            _track(3, "PEDESTRIAN", (20.0, 0.0, 0.0)),   # Middle
            _track(4, "VEGETATION", (40.0, 0.0, 0.0)),   # Far
            _track(5, "VEGETATION", (200.0, 0.0, 0.0)),  # outside all zones
        ]
        assert zone_object_counts(tracks) == {"Near": 2, "Middle": 1, "Far": 1}

    def test_empty_zone_reports_zero_not_omitted(self):
        counts = zone_object_counts([_track(1, "X", (5.0, 0.0, 0.0))])
        assert counts == {"Near": 1, "Middle": 0, "Far": 0}

    def test_no_tracks_all_zero(self):
        assert zone_object_counts([]) == {"Near": 0, "Middle": 0, "Far": 0}
