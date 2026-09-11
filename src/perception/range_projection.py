"""Spherical range-image projection for SalsaNext inference.

This is a faithful reimplementation of the official SalsaNext preprocessing,
transcribed directly from the cloned upstream repository (verified
2026-09-11) rather than from general knowledge of the architecture:

    external/SalsaNext/train/common/laserscan.py :: LaserScan.do_range_projection
    external/SalsaNext/train/tasks/semantic/dataset/kitti/parser.py :: __getitem__

Matching the *training-time* preprocessing exactly matters: the pretrained
checkpoint learned on images produced by that code, so any deviation in
angle convention, collision policy, fill value, or normalization order
shifts the input distribution away from what the model expects.

Deviations from upstream are documented inline and marked "DEVIATION".
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass(frozen=True)
class RangeProjection:
    """Result of projecting a point cloud into a spherical range image.

    Attributes
    ----------
    image : np.ndarray, shape (5, H, W), float32
        Normalized 5-channel image, channel order [range, x, y, z, remission],
        matching the upstream `img_means`/`img_stds` ordering. Invalid
        (unfilled) pixels are zeroed by the mask, as upstream does.
    proj_x : np.ndarray, shape (N,), int32
        Column index each input point projected to, in original point order.
        -1 for points excluded from projection (see `valid`).
    proj_y : np.ndarray, shape (N,), int32
        Row index each input point projected to, in original point order.
        -1 for excluded points.
    mask : np.ndarray, shape (H, W), int32
        1 where a point was projected, 0 otherwise (upstream's `proj_mask`).
    valid : np.ndarray, shape (N,), bool
        True for points that took part in the projection. False for
        zero-range points (see DEVIATION note in the projection function).
    proj_idx : np.ndarray, shape (H, W), int32
        For each pixel, the index of the point that won it (-1 if empty).
    """

    image: np.ndarray
    proj_x: np.ndarray
    proj_y: np.ndarray
    mask: np.ndarray
    valid: np.ndarray
    proj_idx: np.ndarray


def project_points_to_range_image(
    points: np.ndarray,
    arch_cfg: dict[str, Any],
    fov_up: float | None = None,
    fov_down: float | None = None,
) -> RangeProjection:
    """Project (N,4) [x,y,z,intensity] points into a normalized range image.

    Follows upstream `do_range_projection` exactly:
      - yaw   = -arctan2(y, x)                (note the negative sign)
      - pitch = arcsin(z / range)
      - proj_x = 0.5 * (yaw/pi + 1.0)  -> [0,1], scaled by W
      - proj_y = 1.0 - (pitch + |fov_down|)/fov -> [0,1], scaled by H
      - floor, then clamp into [0,W-1] / [0,H-1]
      - collision policy: points are assigned in order of *decreasing*
        range, so nearer points overwrite farther ones -- i.e. the closest
        point wins each pixel (upstream sorts with `np.argsort(depth)[::-1]`
        and assigns in that order).
      - unfilled pixels keep upstream's -1 fill value in the raw channels
        before normalization, and are then zeroed by the mask.

    Parameters
    ----------
    points : np.ndarray, shape (N, 4), float32
    arch_cfg : dict
        Parsed arch_cfg.yaml; reads dataset.sensor.{fov_up,fov_down,
        img_prop.{height,width},img_means,img_stds}.
    fov_up : float | None, optional
        Optional vertical FOV up in degrees (e.g. 17.02 for Ouster OS1-64).
        If None, uses arch_cfg['dataset']['sensor']['fov_up'].
    fov_down : float | None, optional
        Optional vertical FOV down in degrees (e.g. -16.44 for Ouster OS1-64).
        If None, uses arch_cfg['dataset']['sensor']['fov_down'].

    Returns
    -------
    RangeProjection
    """
    if points.ndim != 2 or points.shape[1] != 4:
        raise ValueError(f"points must have shape (N, 4), got {points.shape}.")

    sensor = arch_cfg["dataset"]["sensor"]
    proj_h = int(sensor["img_prop"]["height"])
    proj_w = int(sensor["img_prop"]["width"])
    means = np.asarray(sensor["img_means"], dtype=np.float32)
    stds = np.asarray(sensor["img_stds"], dtype=np.float32)

    raw_fov_up = float(fov_up if fov_up is not None else sensor["fov_up"])
    raw_fov_down = float(fov_down if fov_down is not None else sensor["fov_down"])
    fov_up = raw_fov_up / 180.0 * np.pi
    fov_down = raw_fov_down / 180.0 * np.pi
    fov = abs(fov_down) + abs(fov_up)

    n_points = points.shape[0]
    xyz_all = points[:, :3].astype(np.float32)
    remission_all = points[:, 3].astype(np.float32)
    depth_all = np.linalg.norm(xyz_all, 2, axis=1)

    # DEVIATION from upstream: upstream assumes every point is a real
    # return, because KITTI .bin files contain only real returns. RELLIS-3D's
    # Ouster scans are *organized* (a fixed 64x2048 grid) and pad no-return
    # beams with exactly (0,0,0), which would make arcsin(z/depth) a 0/0
    # NaN and corrupt the projection. Those points are excluded here and
    # reported via `valid`; the caller labels them UNKNOWN rather than
    # letting the model assign them a fabricated class.
    valid = depth_all > 0.0

    proj_x_all = np.full(n_points, -1, dtype=np.int32)
    proj_y_all = np.full(n_points, -1, dtype=np.int32)

    proj_range = np.full((proj_h, proj_w), -1, dtype=np.float32)
    proj_xyz = np.full((proj_h, proj_w, 3), -1, dtype=np.float32)
    proj_remission = np.full((proj_h, proj_w), -1, dtype=np.float32)
    proj_idx = np.full((proj_h, proj_w), -1, dtype=np.int32)

    valid_indices = np.flatnonzero(valid)
    if valid_indices.size > 0:
        xyz = xyz_all[valid_indices]
        remission = remission_all[valid_indices]
        depth = depth_all[valid_indices]

        scan_x, scan_y, scan_z = xyz[:, 0], xyz[:, 1], xyz[:, 2]

        yaw = -np.arctan2(scan_y, scan_x)
        pitch = np.arcsin(scan_z / depth)

        proj_x = 0.5 * (yaw / np.pi + 1.0)
        proj_y = 1.0 - (pitch + abs(fov_down)) / fov

        proj_x *= proj_w
        proj_y *= proj_h

        proj_x = np.floor(proj_x)
        proj_x = np.minimum(proj_w - 1, proj_x)
        proj_x = np.maximum(0, proj_x).astype(np.int32)

        proj_y = np.floor(proj_y)
        proj_y = np.minimum(proj_h - 1, proj_y)
        proj_y = np.maximum(0, proj_y).astype(np.int32)

        proj_x_all[valid_indices] = proj_x
        proj_y_all[valid_indices] = proj_y

        # Assign in order of decreasing range so the closest point wins each
        # pixel (upstream: `order = np.argsort(depth)[::-1]`).
        order = np.argsort(depth)[::-1]
        proj_range[proj_y[order], proj_x[order]] = depth[order]
        proj_xyz[proj_y[order], proj_x[order]] = xyz[order]
        proj_remission[proj_y[order], proj_x[order]] = remission[order]
        proj_idx[proj_y[order], proj_x[order]] = valid_indices[order]

    # Upstream uses `(proj_idx > 0)`, not `>= 0`. That is an off-by-one --
    # it also masks out the pixel won by point index 0 -- but it is what the
    # checkpoint was trained with, so it is replicated deliberately rather
    # than "fixed", to keep inference preprocessing identical to training.
    mask = (proj_idx > 0).astype(np.int32)

    image = np.concatenate(
        [
            proj_range[None, ...],
            proj_xyz.transpose(2, 0, 1),
            proj_remission[None, ...],
        ],
        axis=0,
    ).astype(np.float32)

    # Normalize, then zero invalid pixels -- upstream applies the mask
    # *after* normalization, which matters: an unfilled pixel does not
    # normalize to zero on its own.
    image = (image - means[:, None, None]) / stds[:, None, None]
    image = image * mask.astype(np.float32)

    return RangeProjection(
        image=image.astype(np.float32),
        proj_x=proj_x_all,
        proj_y=proj_y_all,
        mask=mask,
        valid=valid,
        proj_idx=proj_idx,
    )


def reproject_labels_to_points(
    pixel_labels: np.ndarray,
    projection: RangeProjection,
    invalid_label: float,
) -> np.ndarray:
    """Map per-pixel predicted labels back to per-point labels.

    Follows upstream `user.py`'s non-KNN path exactly:
        unproj_argmax = proj_argmax[p_y, p_x]
    i.e. every point reads the label of the pixel it projected into --
    including points that lost a pixel collision, which therefore inherit
    the label of the closest point in that pixel. (The upstream KNN
    post-processing that would refine those is disabled in this
    checkpoint's config: `post.KNN.use: False`.)

    Points excluded from the projection entirely (zero-range padding) get
    `invalid_label` rather than a fabricated class.

    Parameters
    ----------
    pixel_labels : np.ndarray, shape (H, W)
        Per-pixel values to scatter back to points. Despite the name this
        also carries per-pixel float quantities (confidence, uncertainty);
        the output dtype follows `pixel_labels`, so float inputs are not
        truncated to int.
    projection : RangeProjection
    invalid_label : int | float
        Value assigned to points with no valid pixel.

    Returns
    -------
    np.ndarray, shape (N,), same dtype as `pixel_labels`
    """
    n_points = projection.proj_x.shape[0]
    point_labels = np.full(n_points, invalid_label, dtype=pixel_labels.dtype)

    valid = projection.valid
    if valid.any():
        point_labels[valid] = pixel_labels[
            projection.proj_y[valid], projection.proj_x[valid]
        ]
    return point_labels
