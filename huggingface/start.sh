#!/bin/bash
# Runs inside the container: virtual screen -> window manager -> VNC -> browser
# bridge on :7860, then the dashboard. Both dashboard windows (Qt panels and the
# Open3D 3D view) live on one wide virtual screen so noVNC shows them together.
set -u
export DISPLAY=:99
cd /home/user/app

Xvfb :99 -screen 0 3200x1240x24 -nolisten tcp &
for _ in $(seq 1 50); do [ -S /tmp/.X11-unix/X99 ] && break; sleep 0.2; done
fluxbox >/dev/null 2>&1 &
x11vnc -display :99 -forever -shared -nopw -localhost -rfbport 5900 -quiet >/dev/null 2>&1 &
websockify --web=/usr/share/novnc 7860 localhost:5900 &

# Fewer frames than the full dataset: the Space only has the bundled sample.
python src/13_realtime_dashboard.py --source rellis3d --sequence 00001 --frames 100 --rate-hz 10 &
APP_PID=$!

# Lay the two windows out side by side once they exist (Qt panels on the left,
# 3D view on the right); the app maximizes the Qt window on start.
(
  for _ in $(seq 1 180); do
    if wmctrl -l | grep -q "2D & Metrics Dashboard" && wmctrl -l | grep -q "3D Dashboard"; then
      wmctrl -r "2D & Metrics Dashboard" -b remove,maximized_vert,maximized_horz
      wmctrl -r "2D & Metrics Dashboard" -e 0,0,0,1920,1200
      wmctrl -r "3D Dashboard" -e 0,1930,0,1260,720
      break
    fi
    sleep 1
  done
) &

# If the dashboard exits, stop the container so the Space restarts it.
wait $APP_PID
