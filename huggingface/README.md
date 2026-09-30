---
title: FOVEAX 2.5D LiDAR Dashboard
sdk: docker
app_port: 7860
pinned: false
short_description: Real-time foveated 2.5D LiDAR mapping dashboard (PyQt5 + Open3D)
---

# FOVEAX 2.5D LiDAR Dashboard

The real FOVEAX PyQt5 + Open3D dashboard, streamed to your browser from a
virtual desktop. It replays 100 real RELLIS-3D off-road LiDAR frames
(131,072 points each) through the FOVEAX pipeline:

- 2.5D elevation and traversability maps
- foveated resolution zones: 0-10 m at 5 cm, 10-30 m at 20 cm, 30-100 m at 50 cm
- Kalman-tracked objects, hazards and a forward-corridor profile

Left of the desktop is the Qt panel window, right is the Open3D 3D view. Use
PAUSE and STEP to explore a frame.

Notes: this Space runs on CPU with software OpenGL, so it is slower than the
desktop app, and everyone who opens it shares one session. It uses the
dataset's ground-truth labels and does not run the SalsaNext network.

Source code and the downloadable Windows app:
https://github.com/Krsna13/FOVEAX-2.5D-PROTOTYPE
