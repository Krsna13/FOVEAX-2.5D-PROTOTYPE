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

Install them in your active virtual environment. This project's GPU (RTX 5050 Laptop, Blackwell `sm_120`) needs the **cu130** PyTorch build; older CUDA builds (cu121, cu126) install and report `torch.cuda.is_available() == True` but fail on the first real GPU op with `no kernel image is available`:
```bash
pip install torch==2.14.0 torchvision --index-url https://download.pytorch.org/whl/cu130
pip install pyyaml tqdm numba
```
On a different GPU, use the official PyTorch CUDA build that matches its compute capability, and confirm with a real op (e.g. a matmul on `cuda`), not just `is_available()`.

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

**Resolved (2026-09-11):** after the `sys.path` fix below plus 3b and 3c, the checkpoint loads under PyTorch 2.14.0+cu130 with 312/312 weights matched, and real inference reaches 94.98% on SemanticKITTI (`docs/validation_results.md` §2.1). The original investigation notes follow.

**Outcome, observed at the time**: the version gap is **not yet a determined risk either way** -- a real load attempt against the actual downloaded checkpoint (2026-09-11) failed before ever reaching `torch.load()` on the weights file, so nothing has actually tested whether the old checkpoint's `state_dict` is compatible with PyTorch 2.14.0+cu130. The failure is a separate, unrelated bug: the official repo's own `train/tasks/semantic/modules/SalsaNext.py` does `import __init__ as booger`, which only resolves when `train/tasks/semantic/modules/` itself is on `sys.path` (the official `eval.sh` achieves this by `cd`-ing into `train/tasks/semantic/` before running, which Python's script-directory convention then adds to `sys.path[0]`). `salsanext_predictor.py`'s `_load_model()` currently only adds the repo root to `sys.path`, not that subdirectory, so the import fails with `ModuleNotFoundError: No module named '__init__'` -- confirmed via direct traceback, not guessed. This needs a fix to `_load_model()`'s `sys.path` setup before the version-gap question can actually be tested.

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

## 6. Fine-tuned RELLIS-3D checkpoint

`src/14_finetune_salsanext.py` fine-tunes the pretrained checkpoint on RELLIS-3D with an 8-class FOVEAX head and writes to `models/salsanext/rellis3d_finetuned/` (`epoch_000.pt`...`epoch_009.pt`, `last.pt`, `best.pt`, `history.json`, plus copies of `arch_cfg.yaml`/`data_cfg.yaml`). Load it with `taxonomy="foveax"` (`--taxonomy foveax` on `src/10b_eval_distance_metrics.py`):

```bash
python src/10b_eval_distance_metrics.py --dataset-type rellis3d --sequence 00000 --frame 000000 \
    --taxonomy foveax \
    --salsanext-repo external/SalsaNext \
    --checkpoint models/salsanext/rellis3d_finetuned/best.pt \
    --config models/salsanext/rellis3d_finetuned/arch_cfg.yaml \
    --device auto
```

With `taxonomy="foveax"` the network's argmax is already a FOVEAX class ID, so `salsanext_predictor.py` skips `learning_map_inv` and `SEMANTICKITTI_TO_FOVEAX`, and points with no range-image pixel fall back to FOVEAX UNKNOWN (7), not class 0 (which is DRIVABLE_GROUND in this taxonomy). Results and the forgetting trade-off are in `docs/validation_results.md` §2.3 and `docs/rellis3d_integration.md`.
