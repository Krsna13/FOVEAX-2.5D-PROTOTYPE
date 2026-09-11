# OpenPCDet PointPillars Setup

This guide explains how to set up the OpenPCDet toolbox to run a pretrained PointPillars 3-D object detection model for FOVEAX Phase 8B.

**Important Note:** The FOVEAX Phase 8B adapter connects to OpenPCDet natively but does *not* automatically download code or pretrained weights. You must perform these steps manually.

## 1. Prerequisites
- **Python**: 3.8+ (preferably matches your current virtual environment).
- **PyTorch**: 1.8+ with CUDA support is highly recommended. (CPU fallback is allowed but inference will be slow).
- **CUDA Toolkit**: Required for installing and running `spconv` (a core OpenPCDet dependency) efficiently.

## 2. Pinned Version & Clone Instructions
OpenPCDet's `main` branch frequently undergoes API and box post-processing modifications. To guarantee reproducibility and avoid breaking changes in NMS or model configuration schemas, FOVEAX pins OpenPCDet to the **`v0.5.2` release** (commit `b6fbf07fa0d6ca391037d02855a6e704f5478731`), the latest tagged release that actually exists upstream as of this writing.

> Earlier revisions of this document referenced a `v0.6.0` release and a specific commit hash that do
> not exist in the upstream repository (verified directly against `git tag` and `git log` on a fresh
> clone — the highest real tag is `v0.5.2`). If you see references to `v0.6.0` elsewhere, they are
> incorrect; use `v0.5.2` as pinned here.

Clone the repository and checkout the pinned tag:
```bash
git clone https://github.com/open-mmlab/OpenPCDet.git external/OpenPCDet
cd external/OpenPCDet
git checkout v0.5.2
```

## 3. Install Dependencies
Navigate into the OpenPCDet directory and install the necessary dependencies as per the official documentation:
```bash
cd external/OpenPCDet
pip install -r requirements.txt
pip install -e .
```
*(You may need to install `spconv` specific to your CUDA version as instructed on the OpenPCDet README).*

## 4. Obtain the Pretrained Checkpoint & License Terms
Download an official pretrained PointPillars checkpoint (for example, trained on **KITTI Detection**, **nuScenes**, or **Waymo**).
- Place the downloaded `.pth` file into the `models/openpcdet/` directory (e.g., `models/openpcdet/pointpillar_7728.pth`).

### Checkpoint License & Non-Commercial Terms
> [!WARNING]
> Most official pretrained weights for KITTI (and similar autonomous driving benchmarks) are released strictly under **Non-Commercial / Research-Only** terms (such as Creative Commons Attribution-NonCommercial-ShareAlike 3.0, CC BY-NC-SA 3.0). 
> - These models and checkpoints are intended solely for academic research and evaluation.
> - They **must not** be used in commercial deployments without appropriate licensing from the respective dataset authors and institutions.
> - Review the license associated with your chosen checkpoint prior to any redistribution.

## 5. Checkpoint Integrity & User-Supplied Checksum Verification
To guard against corrupted downloads or altered binary files, always verify the SHA-256 checksum of your downloaded checkpoint against the hash provided by your upstream source.

FOVEAX **does not auto-verify against hardcoded hashes**, because official mirrors and versions may differ depending on the specific model variant you choose. Users must verify their own files.

### Verification Instructions
**On Windows (PowerShell):**
```powershell
Get-FileHash -Algorithm SHA256 models/openpcdet/pointpillar_7728.pth
```
or via Windows Command Prompt:
```cmd
certutil -hashfile models/openpcdet/pointpillar_7728.pth SHA256
```

**On Linux / macOS:**
```bash
sha256sum models/openpcdet/pointpillar_7728.pth
```

Compare the displayed 64-character hexadecimal hash with the checksum published on the official OpenPCDet model zoo release page.

## 6. Configuration File
The PointPillars configuration file corresponding to the KITTI model is located at:
`external/OpenPCDet/tools/cfgs/kitti_models/pointpillar.yaml`
Ensure this file exists, as the FOVEAX adapter relies on it.

## 7. Dataset Compatibility, Range, and Taxonomy
- **Spatial Range (`point_cloud_range`)**: KITTI PointPillars models typically expect point coordinates within `x:[0, 70.4]`, `y:[-40.0, 40.0]`, `z:[-3.0, 1.0]`. If FOVEAX points fall outside this range, a warning will be logged.
- **Intensity Convention**: KITTI Velodyne intensity is expected to be a normalized float `[0.0, 1.0]`. If intensities are un-normalized (e.g. 0-255), a warning is logged.
- **Taxonomy Reconciliation**: Native detector classes (`Car`, `Pedestrian`, `Cyclist`) are automatically mapped to FOVEAX's Phase 6 taxonomy (`VEHICLE`, `PEDESTRIAN`, `SOLID_OBSTACLE`), while preserving native labels in metadata.

## 8. Run Validation
Once the repository is cloned, dependencies installed, and checkpoint placed, test the integration:
```bash
python src/11_object_detection_tracking.py --source sample --detector pointpillars \
    --openpcdet-repo external/OpenPCDet \
    --checkpoint models/openpcdet/pointpillar_7728.pth \
    --config external/OpenPCDet/tools/cfgs/kitti_models/pointpillar.yaml \
    --device auto
```

