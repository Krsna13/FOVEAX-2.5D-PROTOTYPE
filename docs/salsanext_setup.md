# SalsaNext Setup for FOVEAX Phase 7B

This document details how to install and configure the official SalsaNext repository for use as an AI inference backend in FOVEAX.

**Official Repository:** [https://github.com/TiagoCortinhal/SalsaNext](https://github.com/TiagoCortinhal/SalsaNext)

> **Important License Notice**
> By cloning and using the official SalsaNext repository and its pretrained weights, you must review and accept their respective licenses and terms of use.

## 1. Clone the Repository
SalsaNext is an external dependency. Clone it into the `external/` folder (do not modify its contents):

```bash
git clone https://github.com/TiagoCortinhal/SalsaNext.git external/SalsaNext
```

## 2. Required Dependencies
According to the SalsaNext official documentation, ensure your Python environment has the necessary packages. You generally need:
- `torch` (PyTorch, ideally with CUDA support)
- `torchvision`
- `numpy`
- `PyYAML`
- `tqdm`
- `numba` (often used in point cloud processing)

Install them in your active virtual environment. For example (for CUDA 12.1):
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
pip install pyyaml tqdm numba
```

## 3. Pretrained Weights

**Manual download required.** The official pretrained checkpoint is linked from the SalsaNext README's "Pretrained Model" section as a Google Drive share link (`https://drive.google.com/file/d/1utfzooTDAlV5M6XGvCE0-L-vbdLe_2rD/view?usp=sharing`), not a direct-fetch URL -- it must be downloaded via a browser, not `curl`/`wget`. The README gives no filename or size; confirmed actual size after download is ~103 MB (the weights file alone).

Confirmed real structure after downloading and extracting (verified 2026-09-11, not assumed):
```text
models/
└── salsanext/
    └── pretrained/
        └── pretrained/
            ├── arch_cfg.yaml
            ├── data_cfg.yaml
            └── SalsaNext          <- weights, no file extension
```
The zip's internal top-level folder is itself named `pretrained/`, so extracting it into `models/salsanext/` produces a `pretrained/pretrained/` double-nesting -- this is the real, verified layout, not a mistake to "fix" by flattening it. `arch_cfg.yaml` and `data_cfg.yaml` ship *inside* this folder alongside the weights; they are not part of the cloned `external/SalsaNext` repository.

## 3a. Version gap: known risk

SalsaNext's own pinned environment (`salsanext_cuda10.yml`) targets **Python 3.7.4, PyTorch 1.1.0, CUDA 10.0** -- roughly five years behind this project's actual stack (Python 3.11.16, PyTorch 2.14.0+cu130). This was flagged as a risk before attempting to load the checkpoint.

**Outcome, observed**: the version gap is **not yet a determined risk either way** -- a real load attempt against the actual downloaded checkpoint (2026-09-11) failed before ever reaching `torch.load()` on the weights file, so nothing has actually tested whether the old checkpoint's `state_dict` is compatible with PyTorch 2.14.0+cu130. The failure is a separate, unrelated bug: the official repo's own `train/tasks/semantic/modules/SalsaNext.py` does `import __init__ as booger`, which only resolves when `train/tasks/semantic/modules/` itself is on `sys.path` (the official `eval.sh` achieves this by `cd`-ing into `train/tasks/semantic/` before running, which Python's script-directory convention then adds to `sys.path[0]`). `salsanext_predictor.py`'s `_load_model()` currently only adds the repo root to `sys.path`, not that subdirectory, so the import fails with `ModuleNotFoundError: No module named '__init__'` -- confirmed via direct traceback, not guessed. This needs a fix to `_load_model()`'s `sys.path` setup before the version-gap question can actually be tested.

(Separately, `SalsaNext.py` also does `import imp`, a stdlib module deprecated and removed in Python 3.12+ -- harmless on this project's Python 3.11.16, but another data point that this repo predates the current toolchain by a wide margin.)

### 3b. `weights_only=False` is required (and why)

Once the `sys.path` issue above was fixed, `torch.load()` was reached and failed with:
```
_pickle.UnpicklingError: Weights only load failed.
	WeightsUnpickler error: Unsupported global: GLOBAL numpy.core.multiarray.scalar was not an allowed global by default.
```
PyTorch 2.6 changed `torch.load`'s `weights_only` default from `False` to `True`. This 2019-era checkpoint contains pickled numpy scalar types that are not in the new default safe-unpickling allowlist, so it cannot load under the new default.

`salsanext_predictor.py` therefore calls `torch.load(..., weights_only=False)`. **This re-enables the arbitrary-code-execution risk inherent to unpickling.** It is considered acceptable here on provenance grounds only: this specific checkpoint comes from the paper authors' own official Google Drive link, published in the SalsaNext repository's README (see section 3).

> **Warning for anyone substituting a different checkpoint**: `weights_only=False` will happily execute arbitrary code embedded in a malicious pickle. Do not point `--checkpoint` at a checkpoint from any other source without independently verifying its provenance first. If you cannot vouch for the source, prefer `torch.serialization.add_safe_globals([...])` with `weights_only=True` instead of loosening the default.

`tests/test_salsanext_predictor.py::test_torch_load_uses_weights_only_false` pins this behaviour so a refactor cannot silently revert to the failing default.

### 3c. Checkpoint keys carry a `module.` prefix (DataParallel)

The checkpoint's `state_dict` has 312 entries, **all** of them prefixed `module.` (e.g. `module.downCntx.conv1.weight`) -- the standard artifact of a model saved while wrapped in `nn.DataParallel`. Loading it directly into a bare `SalsaNext` instance fails with all 312 keys reported missing and all 312 reported unexpected.

Verified (2026-09-11) that this is **purely a prefix mismatch, not an architecture mismatch**: stripping the `module.` prefix and reloading yields `<All keys matched successfully>` under `strict=True`, with 0 missing and 0 unexpected keys. Also note the checkpoint is a full training checkpoint, not bare weights -- its top-level keys are `['epoch', 'state_dict', 'optimizer', 'info', 'scheduler']`.

## 4. Expected Dataset Format
SalsaNext expects the SemanticKITTI dataset in its standard format. For inference, you need the Velodyne `.bin` scans:
```text
data/semantic_kitti/dataset/sequences/00/velodyne/000000.bin
```

## 5. Official Inference & Output
FOVEAX wraps the SalsaNext model loading and inference logic via the adapter `src/perception/salsanext_predictor.py`.
The adapter reads the raw point cloud, processes it through the SalsaNext model, and returns the probabilities.

To run FOVEAX with SalsaNext:
```bash
python src/10_ai_semantic_2point5d_map.py \
    --source salsanext \
    --salsanext-repo external/SalsaNext \
    --checkpoint models/salsanext/pretrained/pretrained/SalsaNext \
    --config models/salsanext/pretrained/pretrained/arch_cfg.yaml \
    --device auto
```

**Note on `--config`**: this previously (incorrectly) pointed at `external/SalsaNext/train/tasks/semantic/config/arch_cfg.yaml` -- that file does not exist anywhere in the official repository. `arch_cfg.yaml` ships inside the downloaded pretrained-model folder itself (see section 3), alongside `data_cfg.yaml`, which `salsanext_predictor.py` now also loads automatically as a sibling of `--config` (needed to derive `num_classes`, which is absent from `arch_cfg.yaml` entirely -- see section 3a).
