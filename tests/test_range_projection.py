"""Unit tests for FOVEAX -- range_projection module.

Covers the spherical range-image projection and the reverse re-projection
used by SalsaNextPredictor. All expected pixel coordinates below are worked
out by hand from the upstream formula (see module docstring in
src/perception/range_projection.py), not read back from the implementation.

Runs without GPU, PyTorch, or the real checkpoint.

Run with:
    python -m pytest tests/test_range_projection.py -v
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.perception.range_projection import (
    project_points_to_range_image,
    reproject_labels_to_points,
)

# The real sensor block from the downloaded checkpoint's arch_cfg.yaml
# (models/salsanext/pretrained/pretrained/arch_cfg.yaml), quoted exactly.
_REAL_SENSOR_CFG = {
    "dataset": {
        "sensor": {
            "name": "HDL64",
            "type": "spherical",
            "fov_up": 3,
            "fov_down": -25,
            "img_prop": {"width": 2048, "height": 64},
            "img_means": [12.12, 10.88, 0.23, -1.04, 0.21],
            "img_stds": [12.32, 11.47, 6.91, 0.86, 0.16],
        }
    }
}

# A tiny config, easier to reason about by hand for exact-coordinate tests.
_TINY_CFG = {
    "dataset": {
        "sensor": {
            "fov_up": 3,
            "fov_down": -25,
            "img_prop": {"width": 4, "height": 4},
            "img_means": [0.0, 0.0, 0.0, 0.0, 0.0],
            "img_stds": [1.0, 1.0, 1.0, 1.0, 1.0],
        }
    }
}


def _pt(x: float, y: float, z: float, i: float = 0.5) -> list[float]:
    return [x, y, z, i]


class TestProjectionGeometry:
    def test_output_shapes_match_real_config(self) -> None:
        points = np.array([_pt(10.0, 0.0, 0.0)], dtype=np.float32)
        result = project_points_to_range_image(points, _REAL_SENSOR_CFG)

        assert result.image.shape == (5, 64, 2048)
        assert result.image.dtype == np.float32
        assert result.proj_x.shape == (1,)
        assert result.proj_y.shape == (1,)
        assert result.mask.shape == (64, 2048)

    def test_point_straight_ahead_lands_at_horizontal_centre(self) -> None:
        """A point at +x, y=0, z=0 must land at column W/2.

        By hand: yaw = -atan2(0, 10) = 0, so
        proj_x = 0.5*(0/pi + 1.0) = 0.5 -> 0.5 * W = W/2 = 1024.
        """
        points = np.array([_pt(10.0, 0.0, 0.0)], dtype=np.float32)
        result = project_points_to_range_image(points, _REAL_SENSOR_CFG)

        assert result.proj_x[0] == 1024

    def test_point_directly_behind_lands_at_column_zero(self) -> None:
        """A point at -x, y=0 gives yaw = -atan2(0,-10) = -pi.

        proj_x = 0.5*(-pi/pi + 1.0) = 0.0 -> column 0.
        """
        points = np.array([_pt(-10.0, 0.0, 0.0)], dtype=np.float32)
        result = project_points_to_range_image(points, _REAL_SENSOR_CFG)

        assert result.proj_x[0] == 0

    def test_yaw_sign_convention_is_negated(self) -> None:
        """Upstream uses yaw = -atan2(y, x), so +y maps LEFT of centre.

        For (0, 10, 0): yaw = -atan2(10, 0) = -pi/2, so
        proj_x = 0.5*(-0.5 + 1.0) = 0.25 -> 0.25 * 2048 = 512.
        A non-negated convention would give 1536 instead, so this test
        pins the sign.
        """
        points = np.array([_pt(0.0, 10.0, 0.0)], dtype=np.float32)
        result = project_points_to_range_image(points, _REAL_SENSOR_CFG)

        assert result.proj_x[0] == 512

    def test_row_for_horizontal_point(self) -> None:
        """z=0 -> pitch=0. proj_y = 1.0 - (0 + 25deg)/28deg = 0.107142...

        0.107142... * 64 = 6.857 -> floor -> row 6.
        """
        points = np.array([_pt(10.0, 0.0, 0.0)], dtype=np.float32)
        result = project_points_to_range_image(points, _REAL_SENSOR_CFG)

        fov_up = 3.0 / 180.0 * np.pi
        fov_down = -25.0 / 180.0 * np.pi
        fov = abs(fov_up) + abs(fov_down)
        expected_row = int(np.floor((1.0 - (0.0 + abs(fov_down)) / fov) * 64))

        assert expected_row == 6
        assert result.proj_y[0] == 6

    def test_top_of_fov_maps_to_row_zero(self) -> None:
        """A point at exactly +fov_up (3 deg elevation) maps to row 0."""
        elev = 3.0 / 180.0 * np.pi
        r = 10.0
        points = np.array(
            [_pt(r * np.cos(elev), 0.0, r * np.sin(elev))], dtype=np.float32
        )
        result = project_points_to_range_image(points, _REAL_SENSOR_CFG)

        assert result.proj_y[0] == 0

    def test_out_of_fov_point_is_clamped_not_dropped(self) -> None:
        """Upstream clamps rather than discarding out-of-FOV points."""
        # Far below fov_down (-25 deg): straight down.
        points = np.array([_pt(0.1, 0.0, -10.0)], dtype=np.float32)
        result = project_points_to_range_image(points, _REAL_SENSOR_CFG)

        assert result.proj_y[0] == 63  # clamped to H-1, not -1/dropped
        assert result.valid[0]

    def test_fov_override_applies_correct_projection(self) -> None:
        """Overriding FOV (e.g. for Ouster OS1-64) changes row calculation."""
        # 10 deg elevation point: clamped to row 0 under HDL-64E (fov_up=3),
        # but lands comfortably inside middle rows under Ouster (fov_up=17.02, fov_down=-16.44).
        elev = 10.0 / 180.0 * np.pi
        r = 10.0
        points = np.array(
            [_pt(r * np.cos(elev), 0.0, r * np.sin(elev))], dtype=np.float32
        )
        default_res = project_points_to_range_image(points, _REAL_SENSOR_CFG)
        assert default_res.proj_y[0] == 0  # clamped to top row under HDL-64E

        ouster_res = project_points_to_range_image(
            points, _REAL_SENSOR_CFG, fov_up=17.02, fov_down=-16.44
        )
        # pitch = 10 deg. fov = 17.02 + 16.44 = 33.46 deg.
        # proj_y = 1.0 - (10 + 16.44)/33.46 = 0.2098... -> * 64 = 13.42 -> row 13
        assert ouster_res.proj_y[0] == 13

    def test_fov_override_none_preserves_default(self) -> None:
        """Passing fov_up=None and fov_down=None gives identical result to omitted args."""
        points = np.array([_pt(10.0, 0.0, 0.0)], dtype=np.float32)
        res_default = project_points_to_range_image(points, _REAL_SENSOR_CFG)
        res_none = project_points_to_range_image(
            points, _REAL_SENSOR_CFG, fov_up=None, fov_down=None
        )
        assert np.array_equal(res_default.image, res_none.image)
        assert np.array_equal(res_default.proj_x, res_none.proj_x)
        assert np.array_equal(res_default.proj_y, res_none.proj_y)


class TestCollisionPolicy:
    def test_closest_point_wins_pixel(self) -> None:
        """Two points on the same ray: the nearer one must own the pixel.

        Upstream assigns in order of decreasing range, so the closest write
        lands last and wins.
        """
        near = _pt(5.0, 0.0, 0.0, 0.1)
        far = _pt(50.0, 0.0, 0.0, 0.9)
        points = np.array([far, near], dtype=np.float32)

        result = project_points_to_range_image(points, _REAL_SENSOR_CFG)

        # Both project to the same pixel.
        assert result.proj_x[0] == result.proj_x[1]
        assert result.proj_y[0] == result.proj_y[1]

        row, col = result.proj_y[0], result.proj_x[0]
        # proj_idx must name the NEAR point (index 1), not the far one.
        assert result.proj_idx[row, col] == 1

    def test_collision_order_independent_of_input_order(self) -> None:
        """Same result whether the near point comes first or second."""
        near = _pt(5.0, 0.0, 0.0, 0.1)
        far = _pt(50.0, 0.0, 0.0, 0.9)

        a = project_points_to_range_image(
            np.array([near, far], dtype=np.float32), _REAL_SENSOR_CFG
        )
        row, col = a.proj_y[0], a.proj_x[0]
        assert a.proj_idx[row, col] == 0  # near is index 0 here

        b = project_points_to_range_image(
            np.array([far, near], dtype=np.float32), _REAL_SENSOR_CFG
        )
        assert b.proj_idx[row, col] == 1  # near is index 1 here


class TestInvalidPoints:
    def test_zero_range_points_excluded_from_projection(self) -> None:
        """(0,0,0) no-return padding must not be projected (would be NaN)."""
        points = np.array(
            [_pt(0.0, 0.0, 0.0, 0.0), _pt(10.0, 0.0, 0.0, 0.5)], dtype=np.float32
        )
        with np.errstate(invalid="raise", divide="raise"):
            result = project_points_to_range_image(points, _REAL_SENSOR_CFG)

        assert not result.valid[0]
        assert result.valid[1]
        assert result.proj_x[0] == -1
        assert result.proj_y[0] == -1

    def test_all_zero_points_produce_empty_mask(self) -> None:
        points = np.zeros((5, 4), dtype=np.float32)
        result = project_points_to_range_image(points, _REAL_SENSOR_CFG)

        assert not result.valid.any()
        assert result.mask.sum() == 0

    def test_rejects_wrong_shape(self) -> None:
        with pytest.raises(ValueError, match=r"shape \(N, 4\)"):
            project_points_to_range_image(
                np.zeros((3, 3), dtype=np.float32), _REAL_SENSOR_CFG
            )


class TestNormalizationAndMask:
    def test_empty_pixels_are_zeroed_after_normalization(self) -> None:
        """Masked-out pixels must be exactly 0 across all 5 channels."""
        points = np.array([_pt(10.0, 0.0, 0.0)], dtype=np.float32)
        result = project_points_to_range_image(points, _REAL_SENSOR_CFG)

        empty = result.mask == 0
        assert empty.any()
        for channel in range(5):
            assert np.all(result.image[channel][empty] == 0.0)

    def test_normalization_uses_config_means_and_stds(self) -> None:
        """Filled pixel value must equal (raw - mean)/std for each channel."""
        # Use point index 1, since upstream's `proj_idx > 0` mask excludes
        # the pixel won by point index 0 (see DEVIATION note in the source).
        points = np.array(
            [_pt(-10.0, 0.0, 0.0), _pt(10.0, 0.0, 0.0, 0.5)], dtype=np.float32
        )
        result = project_points_to_range_image(points, _REAL_SENSOR_CFG)

        row, col = result.proj_y[1], result.proj_x[1]
        means = _REAL_SENSOR_CFG["dataset"]["sensor"]["img_means"]
        stds = _REAL_SENSOR_CFG["dataset"]["sensor"]["img_stds"]

        raw = [10.0, 10.0, 0.0, 0.0, 0.5]  # range, x, y, z, remission
        for channel, raw_value in enumerate(raw):
            expected = (raw_value - means[channel]) / stds[channel]
            assert result.image[channel, row, col] == pytest.approx(expected, abs=1e-5)

    def test_point_index_zero_pixel_is_masked_upstream_off_by_one(self) -> None:
        """Deliberately replicates upstream's `proj_idx > 0` off-by-one.

        Upstream masks with `> 0` rather than `>= 0`, so the pixel won by
        point index 0 is treated as empty. Replicated on purpose to keep
        inference preprocessing identical to what the checkpoint was
        trained with -- this test documents it so it isn't "fixed" by
        accident.
        """
        points = np.array([_pt(10.0, 0.0, 0.0)], dtype=np.float32)
        result = project_points_to_range_image(points, _REAL_SENSOR_CFG)

        row, col = result.proj_y[0], result.proj_x[0]
        assert result.proj_idx[row, col] == 0
        assert result.mask[row, col] == 0


class TestReverseProjection:
    def test_labels_assigned_from_owning_pixel(self) -> None:
        points = np.array(
            [_pt(10.0, 0.0, 0.0), _pt(0.0, 10.0, 0.0)], dtype=np.float32
        )
        result = project_points_to_range_image(points, _REAL_SENSOR_CFG)

        pixel_labels = np.zeros((64, 2048), dtype=np.int32)
        pixel_labels[result.proj_y[0], result.proj_x[0]] = 5
        pixel_labels[result.proj_y[1], result.proj_x[1]] = 9

        point_labels = reproject_labels_to_points(pixel_labels, result, invalid_label=0)

        assert point_labels[0] == 5
        assert point_labels[1] == 9

    def test_invalid_points_get_invalid_label(self) -> None:
        points = np.array(
            [_pt(0.0, 0.0, 0.0), _pt(10.0, 0.0, 0.0)], dtype=np.float32
        )
        result = project_points_to_range_image(points, _REAL_SENSOR_CFG)

        pixel_labels = np.full((64, 2048), 3, dtype=np.int32)
        point_labels = reproject_labels_to_points(
            pixel_labels, result, invalid_label=99
        )

        assert point_labels[0] == 99  # zero-range point
        assert point_labels[1] == 3

    def test_collided_point_inherits_winning_pixel_label(self) -> None:
        """The losing point of a collision reads the winner's pixel label.

        This is upstream's documented behaviour (`proj_argmax[p_y, p_x]`
        indexes by each point's own pixel, with KNN post-processing
        disabled in this checkpoint's config).
        """
        near = _pt(5.0, 0.0, 0.0)
        far = _pt(50.0, 0.0, 0.0)
        points = np.array([far, near], dtype=np.float32)
        result = project_points_to_range_image(points, _REAL_SENSOR_CFG)

        pixel_labels = np.zeros((64, 2048), dtype=np.int32)
        pixel_labels[result.proj_y[1], result.proj_x[1]] = 7

        point_labels = reproject_labels_to_points(pixel_labels, result, invalid_label=0)

        assert point_labels[1] == 7  # winner
        assert point_labels[0] == 7  # loser inherits the same pixel's label

    def test_float_values_are_not_truncated(self) -> None:
        """Confidence/uncertainty are floats -- dtype must be preserved."""
        points = np.array([_pt(10.0, 0.0, 0.0)], dtype=np.float32)
        result = project_points_to_range_image(points, _REAL_SENSOR_CFG)

        pixel_conf = np.full((64, 2048), 0.75, dtype=np.float32)
        point_conf = reproject_labels_to_points(
            pixel_conf, result, invalid_label=0.0
        )

        assert point_conf.dtype == np.float32
        assert point_conf[0] == pytest.approx(0.75)
