# FOVEAX 2.5D dashboard for a Hugging Face Docker Space.
#
# A Space cannot show a native PyQt5 window, so this image runs the real PyQt5
# + Open3D dashboard on a virtual X display (Xvfb) and streams that display to
# the browser with noVNC on port 7860 (the port Spaces expect).
#
# The image is CPU-only: the dashboard replays recorded RELLIS-3D frames with
# ground-truth labels, so no PyTorch/GPU is needed. Rendering uses Mesa
# software OpenGL, so expect it to be slower than on a desktop.
#
# Space repo layout (only these files are needed at its root):
#   Dockerfile, start.sh, fetch_data.py, README.md   <- from this folder
# The application source is cloned from GitHub at build time.
FROM python:3.11-slim-bookworm

ARG REPO_URL=https://github.com/Krsna13/FOVEAX-2.5D-PROTOTYPE.git
ARG REPO_REF=main
# Zip that contains FOVEAX/data/rellis3d/... (the Windows release bundle).
# If it cannot be downloaded the dashboard falls back to a synthetic sample.
ARG SAMPLE_DATA_URL=https://github.com/Krsna13/FOVEAX-2.5D-PROTOTYPE/releases/latest/download/FOVEAX-windows.zip

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

# X server, VNC + browser bridge, a window manager (needed to position the two
# windows), and the shared libraries PyQt5 and Open3D load at runtime.
RUN apt-get update && apt-get install -y --no-install-recommends \
        xvfb x11vnc novnc websockify fluxbox wmctrl \
        libgl1 libglu1-mesa libegl1 libgl1-mesa-dri libglx-mesa0 \
        libxkbcommon-x11-0 libxcb-icccm4 libxcb-image0 libxcb-keysyms1 \
        libxcb-randr0 libxcb-render-util0 libxcb-shape0 libxcb-xinerama0 libxcb-xkb1 \
        libxrender1 libxi6 libxrandr2 libxinerama1 libxcursor1 libsm6 libice6 \
        libfontconfig1 libdbus-1-3 libgomp1 fonts-dejavu-core \
        git ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Opening the Space root should open the desktop, not the noVNC file listing.
RUN printf '%s' '<!doctype html><meta charset="utf-8"><meta http-equiv="refresh" content="0;url=vnc.html?autoconnect=true&resize=scale&reconnect=true"><title>FOVEAX</title>' \
        > /usr/share/novnc/index.html

# Spaces run the container as a non-root user with uid 1000.
RUN useradd -m -u 1000 user
USER user
ENV HOME=/home/user PATH=/home/user/.local/bin:$PATH
WORKDIR /home/user/app

RUN git clone --depth 1 --branch ${REPO_REF} ${REPO_URL} . \
    && pip install --user -r requirements-app.txt

COPY --chown=user fetch_data.py start.sh /home/user/
RUN chmod +x /home/user/start.sh \
    && python /home/user/fetch_data.py "${SAMPLE_DATA_URL}" /home/user/app/data

ENV DISPLAY=:99 \
    QT_QPA_PLATFORM=xcb \
    QT_X11_NO_MITSHM=1 \
    LIBGL_ALWAYS_SOFTWARE=1 \
    MESA_GL_VERSION_OVERRIDE=3.3 \
    MESA_GLSL_VERSION_OVERRIDE=330

EXPOSE 7860
CMD ["/home/user/start.sh"]
