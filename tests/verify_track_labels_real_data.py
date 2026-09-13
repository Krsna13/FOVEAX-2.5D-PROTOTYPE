"""Real-data verification of 3D box labels (run manually, not by pytest).

Streams real RELLIS-3D frames through the real tracker into a hidden
FoveaX3DViewer, then for each checked frame and camera preset verifies:

  1. exactly one label per drawn box (same track IDs, same order as tracks);
  2. label text equals the track's own class_name / dynamic flag, rebuilt here
     independently of track_labels.track_label_text;
  3. the label anchor is the drawn LineSet's real top-face center;
  4. the projected top corners of every drawn box land on box-colored pixels
     in Open3D's actual rendered frame -- i.e. the projection labels use
     matches what the renderer draws.

    python tests/verify_track_labels_real_data.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

try:
    import torch  # noqa: F401  (torch before PyQt5 on Windows)
except ImportError:
    pass

from src.dashboard.data_streamer import DataStreamerThread
from src.dashboard.open3d_viewer import FoveaX3DViewer
from src.dashboard.track_labels import (
    LABEL_Z_OFFSET_M,
    display_class_for_track,
    project_to_screen,
    track_distance_m,
)

W, H = 1280, 720
FRAMES = 60
CHECK_EVERY = 10
PRESETS = ("default", "view_perspective", "view_top", "view_side")
BOX_COLORS = {True: np.array([0.0, 0.5, 1.0]), False: np.array([0.6, 0.2, 0.8])}


def main() -> int:
    streamer = DataStreamerThread(source="rellis3d", sequence="00001", num_frames=FRAMES, rate_hz=14.0)
    streamer._setup_sources()
    viewer = FoveaX3DViewer(width=W, height=H, visible=False)
    vc = viewer.view_control

    totals = dict(frames=0, views=0, tracks=0, labels=0, text_ok=0, anchor_ok=0,
                  corners_checked=0, corners_hit=0)
    failures: list[str] = []
    first_view_done = False

    for idx in range(FRAMES):
        state = streamer.process_frame(idx, timestamp_s=idx / 14.0)
        if idx % CHECK_EVERY:
            continue
        viewer.update(state)
        if not first_view_done:
            viewer.vis.reset_view_point(True)
            first_view_done = True
        totals["frames"] += 1

        for preset in PRESETS:
            if preset != "default":
                viewer.handle_camera_command(preset)
            viewer.vis.remove_geometry(viewer.pcd, reset_bounding_box=False)
            for _ in range(2):
                viewer.vis.poll_events()
                viewer.vis.update_renderer()
            img = np.asarray(viewer.vis.capture_screen_float_buffer(do_render=True))
            viewer.vis.add_geometry(viewer.pcd, reset_bounding_box=False)

            labels = viewer.compute_track_labels()
            params = vc.convert_to_pinhole_camera_parameters()
            k, ext = params.intrinsic.intrinsic_matrix, params.extrinsic
            totals["views"] += 1
            totals["tracks"] += len(state.tracks)
            totals["labels"] += len(labels)

            ids_tracks = [t.track_id for t in state.tracks]
            ids_labels = [l.track_id for l in labels]
            if ids_labels != ids_tracks or set(ids_labels) != set(viewer.track_boxes):
                failures.append(f"frame {idx} {preset}: label ids {ids_labels} != track ids {ids_tracks}")

            for t, lab in zip(state.tracks, labels):
                status = "Dynamic" if t.dynamic else "Static"
                expected_dist = track_distance_m(t)
                expected = (
                    f"{display_class_for_track(t)} ({status}) - {expected_dist:.1f}m"
                )
                if lab.text == expected:
                    totals["text_ok"] += 1
                else:
                    failures.append(f"frame {idx} track {t.track_id}: text {lab.text!r} != {expected!r}")

                real_dist = float(np.hypot(t.state[0], t.state[1]))
                if abs(real_dist - expected_dist) > 1e-9:
                    failures.append(
                        f"frame {idx} track {t.track_id}: track_distance_m {expected_dist} "
                        f"!= real hypot(x,y) {real_dist}"
                    )

                top = np.asarray(viewer.track_boxes[t.track_id].points)[4:8]
                anchor = top.mean(axis=0) + np.array([0.0, 0.0, LABEL_Z_OFFSET_M])
                au, av, _ = project_to_screen(anchor[None], k, ext, W, H)
                if abs(au[0] - lab.u) < 1e-6 and abs(av[0] - lab.v) < 1e-6:
                    totals["anchor_ok"] += 1
                else:
                    failures.append(f"frame {idx} track {t.track_id}: anchor mismatch")

                cu, cv, cvis = project_to_screen(top, k, ext, W, H)
                color = BOX_COLORS[bool(t.dynamic)]
                for u, v, ok in zip(cu, cv, cvis):
                    if not ok or u < 3 or v < 3 or u > W - 4 or v > H - 4:
                        continue
                    totals["corners_checked"] += 1
                    patch = img[int(v) - 3:int(v) + 4, int(u) - 3:int(u) + 4]
                    if (np.abs(patch - color).max(axis=2) < 0.12).any():
                        totals["corners_hit"] += 1

    print("=== Track label verification on real RELLIS-3D (seq 00001) ===")
    for key, val in totals.items():
        print(f"  {key:16s} {val}")
    rate = totals["corners_hit"] / max(1, totals["corners_checked"])
    print(f"  corner hit rate  {rate * 100:.2f}%")
    print(f"  failures         {len(failures)}")
    for f in failures[:20]:
        print("   ", f)
    viewer.destroy()
    return 0 if not failures and rate > 0.95 else 1


if __name__ == "__main__":
    raise SystemExit(main())
