"""Download the bundled RELLIS-3D sample into <dest>/rellis3d/...

Usage: python fetch_data.py <zip-url> <dest-dir>

The Windows release zip holds FOVEAX/data/rellis3d/Rellis-3D/00001/... . Only
that data folder is extracted. Any failure is reported but not fatal: without
the data the dashboard falls back to its built-in synthetic sample, so the
Space still starts.
"""
import shutil
import sys
import tempfile
import urllib.request
import zipfile
from pathlib import Path

PREFIX = "FOVEAX/data/"


def main() -> int:
    if len(sys.argv) != 3:
        print(__doc__)
        return 2
    url, dest = sys.argv[1], Path(sys.argv[2])
    try:
        with tempfile.TemporaryDirectory() as tmp:
            zpath = Path(tmp) / "sample.zip"
            print(f"[fetch_data] downloading {url}", flush=True)
            with urllib.request.urlopen(url, timeout=120) as r, open(zpath, "wb") as f:
                shutil.copyfileobj(r, f)
            dest.mkdir(parents=True, exist_ok=True)
            n = 0
            with zipfile.ZipFile(zpath) as z:
                for member in z.namelist():
                    if member.startswith(PREFIX) and not member.endswith("/"):
                        target = dest / member[len(PREFIX):]
                        target.parent.mkdir(parents=True, exist_ok=True)
                        with z.open(member) as src, open(target, "wb") as out:
                            shutil.copyfileobj(src, out)
                        n += 1
        if n == 0:
            print("[fetch_data] WARNING: zip had no data/ folder; using synthetic sample.")
        else:
            print(f"[fetch_data] extracted {n} files into {dest}")
    except Exception as exc:  # noqa: BLE001 - never fail the image build over sample data
        print(f"[fetch_data] WARNING: could not fetch sample data ({exc}); "
              "the dashboard will use its synthetic sample.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
