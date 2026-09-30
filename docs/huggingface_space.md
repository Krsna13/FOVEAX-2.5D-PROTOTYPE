# Deploying the dashboard as a Hugging Face Space

A Space cannot display a native PyQt5 window, so the Docker image in
`huggingface/` runs the real dashboard on a virtual display (Xvfb) and
streams it to the browser with noVNC on port 7860.

## Deploy

1. Create a Space at https://huggingface.co/new-space, choose **Docker**
   (Blank template) and CPU basic hardware.
2. Copy the four files from `huggingface/` to the **root of the Space repo**:
   `Dockerfile`, `start.sh`, `fetch_data.py`, `README.md`.
   ```bash
   git clone https://huggingface.co/spaces/<you>/<space-name>
   cp huggingface/{Dockerfile,start.sh,fetch_data.py,README.md} <space-name>/
   cd <space-name> && git add . && git commit -m "Add FOVEAX dashboard" && git push
   ```
3. The Space builds automatically. Open its URL; the desktop connects by itself.

## What the image does

- Clones this GitHub repo at build time (`REPO_URL`, `REPO_REF` build args).
- Installs `requirements-app.txt` (no PyTorch, so the image stays small).
- Downloads the 100-frame RELLIS-3D sample from the latest GitHub Release
  (`SAMPLE_DATA_URL` build arg; the release zip must be published). If the
  download fails the dashboard falls back to a synthetic sample.
- Starts Xvfb, fluxbox, x11vnc and noVNC, then the dashboard, and places the
  Qt window and the Open3D window side by side.

Set build args under the Space's **Settings -> Variables** (or edit the
`ARG` lines) to point at a fork, a tag, or a different data zip.

## Limits

- CPU only and software OpenGL: slower than the Windows app.
- One shared session: everyone who opens the Space controls the same window.
- Only sequence 00001 (100 frames); no SalsaNext (needs PyTorch and a GPU).
- The Space's app code comes from GitHub `main` at build time, not from the
  Space repo, so push to GitHub first, then restart the Space to rebuild.
