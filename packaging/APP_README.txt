FOVEAX 2.5D - Real-Time LiDAR Dashboard (Windows)

Double-click FOVEAX.exe.

What you see: a replay of real RELLIS-3D off-road LiDAR frames (Ouster OS1-64,
131,072 points per frame) processed live by the FOVEAX pipeline: 2.5D elevation
and traversability maps, foveated resolution zones (0-10 m 5 cm, 10-30 m 20 cm,
30-100 m 50 cm), semantic classes from the dataset's ground-truth labels,
Kalman-tracked objects, hazards and a forward-corridor profile.

Notes
- This demo build uses ground-truth labels. It does not run the SalsaNext
  neural network (that needs the full source install with PyTorch).
- Only the bundled sample sequence (00001) is included, so the environment
  buttons for other RELLIS-3D sequences fall back to a synthetic sample.
- Needs Windows 10/11 and a GPU/driver with OpenGL 3.3 for the 3D view.
- Session telemetry is written to outputs\ next to FOVEAX.exe.

Source, full documentation and the SalsaNext models:
https://github.com/Krsna13/FOVEAX-2.5D-PROTOTYPE
