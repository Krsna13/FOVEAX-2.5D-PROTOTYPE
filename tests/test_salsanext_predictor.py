"""Unit tests for the SalsaNext adapter.

Run with:
    python -m pytest tests/test_salsanext_predictor.py -v
"""

import sys
from pathlib import Path

import numpy as np
import pytest

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from src.perception.salsanext_predictor import SalsaNextPredictor
import src.perception.salsanext_predictor as snp
from unittest.mock import MagicMock

def setup_module(module):
    """Patch TORCH_AVAILABLE and torch for testing without PyTorch."""
    snp.TORCH_AVAILABLE = True
    snp.torch = MagicMock()
    snp.torch.cuda.is_available.return_value = False
    
    # Mock torch.device to just return the string it is given
    def mock_device(device_str):
        mock_dev = MagicMock()
        mock_dev.type = device_str
        return mock_dev
        
    snp.torch.device = mock_device


def test_missing_repo_raises_error(tmp_path: Path) -> None:
    """Missing repository directory should raise FileNotFoundError."""
    repo_path = tmp_path / "SalsaNext_Missing"
    checkpoint_path = tmp_path / "checkpoint.pth"
    config_path = tmp_path / "arch_cfg.yaml"
    
    # Touch checkpoint and config so they exist, isolating the repo failure
    checkpoint_path.touch()
    config_path.touch()

    with pytest.raises(FileNotFoundError, match="SalsaNext repository not found"):
        SalsaNextPredictor(
            repo_path=repo_path,
            checkpoint_path=checkpoint_path,
            config_path=config_path,
            strict=True,
        )


def test_missing_checkpoint_raises_error(tmp_path: Path) -> None:
    """Missing checkpoint should raise FileNotFoundError."""
    repo_path = tmp_path / "SalsaNext"
    repo_path.mkdir()
    checkpoint_path = tmp_path / "missing_checkpoint.pth"
    config_path = tmp_path / "arch_cfg.yaml"
    config_path.touch()

    with pytest.raises(FileNotFoundError, match="SalsaNext checkpoint not found"):
        SalsaNextPredictor(
            repo_path=repo_path,
            checkpoint_path=checkpoint_path,
            config_path=config_path,
            strict=True,
        )


def test_missing_config_raises_error(tmp_path: Path) -> None:
    """Missing config should raise FileNotFoundError."""
    repo_path = tmp_path / "SalsaNext"
    repo_path.mkdir()
    checkpoint_path = tmp_path / "checkpoint.pth"
    checkpoint_path.touch()
    config_path = tmp_path / "missing_arch_cfg.yaml"

    with pytest.raises(FileNotFoundError, match="SalsaNext config not found"):
        SalsaNextPredictor(
            repo_path=repo_path,
            checkpoint_path=checkpoint_path,
            config_path=config_path,
            strict=True,
        )


def _write_fake_salsanext_repo(repo_path: Path, track_state_dict: bool = False) -> None:
    """Create a minimal stand-in for the real SalsaNext repo's model module.

    Matches the real, verified import path
    (train.tasks.semantic.modules.SalsaNext.SalsaNext) so _load_model()'s
    dynamic import succeeds against a fake repo, without needing torch for
    the fake model class itself (its methods are trivial no-ops).

    Faithfully reproduces the real repo's `import __init__ as booger` quirk
    (verified present in the actual external/SalsaNext clone) at the top of
    SalsaNext.py -- this only resolves if train/tasks/semantic/modules/
    itself is on sys.path, so any test using this fake repo is a genuine
    regression test for that sys.path fix, not just a mock.

    Every call clears any cached `train`/`train.*` modules first: the
    dotted import name `train.tasks.semantic.modules.SalsaNext` is fixed
    regardless of which test's tmp_path repo produced it, so without this,
    Python's module cache would silently hand a *different* test's fake
    module (or an earlier version of this one, since track_state_dict
    varies across calls) to whichever test runs next in the same pytest
    process.

    track_state_dict: if True, the fake class records whatever dict
    load_state_dict() is actually called with (as self.received_state_dict)
    instead of discarding it -- used to assert on key names post-stripping,
    without needing a real nn.Module.
    """
    for mod_name in list(sys.modules):
        if mod_name == "train" or mod_name.startswith("train."):
            del sys.modules[mod_name]

    modules_dir = repo_path / "train" / "tasks" / "semantic" / "modules"
    modules_dir.mkdir(parents=True)
    (repo_path / "train" / "__init__.py").touch()
    (repo_path / "train" / "tasks" / "__init__.py").touch()
    (repo_path / "train" / "tasks" / "semantic" / "__init__.py").touch()
    (modules_dir / "__init__.py").touch()

    if track_state_dict:
        load_state_dict_body = (
            "    def __init__(self, nclasses):\n"
            "        self.nclasses = nclasses\n"
            "        self.received_state_dict = None\n"
            "    def load_state_dict(self, state_dict):\n"
            "        self.received_state_dict = state_dict\n"
        )
    else:
        load_state_dict_body = (
            "    def __init__(self, nclasses):\n"
            "        self.nclasses = nclasses\n"
            "    def load_state_dict(self, state_dict):\n"
            "        pass\n"
        )

    (modules_dir / "SalsaNext.py").write_text(
        "import __init__ as booger\n"
        "class SalsaNext:\n"
        f"{load_state_dict_body}"
        "    def to(self, device):\n"
        "        return self\n"
        "    def eval(self):\n"
        "        return self\n",
        encoding="utf-8",
    )


# A faithful copy of the real downloaded checkpoint's arch_cfg.yaml top-level
# structure (train/post/dataset.sensor), confirmed to NOT contain num_classes
# anywhere -- this is exactly why num_classes must come from data_cfg.yaml
# instead (see the comment in salsanext_predictor.py's _load_model).
_REAL_ARCH_CFG_YAML = """
train:
  loss: "xentropy"
  max_epochs: 150
post:
  KNN:
    use: False
dataset:
  labels: "kitti"
  scans: "kitti"
  max_points: 150000
  sensor:
    name: "HDL64"
    type: "spherical"
    fov_up: 3
    fov_down: -25
    img_prop:
      width: 2048
      height: 64
"""

# A faithful copy of the real downloaded checkpoint's data_cfg.yaml
# learning_map_inv structure (20 SemanticKITTI-reduced classes, 0-19).
_REAL_DATA_CFG_YAML = """
name: "kitti"
learning_map_inv:
  0: 0
  1: 10
  2: 11
  3: 15
  4: 18
  5: 20
  6: 30
  7: 31
  8: 32
  9: 40
  10: 44
  11: 48
  12: 49
  13: 50
  14: 51
  15: 70
  16: 71
  17: 72
  18: 80
  19: 81
"""


def test_num_classes_derived_from_data_cfg_learning_map_inv(tmp_path: Path) -> None:
    """_load_model must derive num_classes from data_cfg.yaml, not arch_cfg.yaml.

    Regression test for the fixed placeholder at the old line 132
    (arch_cfg["dataset"]["sensor"]["dataset"]["sensor"]["num_classes"]),
    which never matched any real arch_cfg.yaml structure. Verified against
    a real downloaded checkpoint (2026-09-11): num_classes is absent from
    arch_cfg.yaml entirely and must be len(data_cfg["learning_map_inv"]),
    matching the official SalsaNext/RangeNet++ parser.py.
    """
    repo_path = tmp_path / "SalsaNext"
    _write_fake_salsanext_repo(repo_path)

    config_path = tmp_path / "arch_cfg.yaml"
    config_path.write_text(_REAL_ARCH_CFG_YAML, encoding="utf-8")
    (tmp_path / "data_cfg.yaml").write_text(_REAL_DATA_CFG_YAML, encoding="utf-8")

    checkpoint_path = tmp_path / "SalsaNext_checkpoint"
    checkpoint_path.touch()

    predictor = SalsaNextPredictor(
        repo_path=repo_path,
        checkpoint_path=checkpoint_path,
        config_path=config_path,
        device="cpu",
        strict=True,
    )
    predictor._load_model()

    assert predictor.model is not None
    assert predictor.model.nclasses == 20


def test_sys_path_fix_resolves_booger_import(tmp_path: Path) -> None:
    """_load_model must add train/tasks/semantic/modules/ to sys.path.

    Regression test for the real ModuleNotFoundError: No module named
    '__init__' failure (confirmed against the actual external/SalsaNext
    clone, 2026-09-11) caused by SalsaNext.py's own `import __init__ as
    booger`. This test's fake repo reproduces that exact line (see
    _write_fake_salsanext_repo) via a real import, not a mock -- if the
    sys.path fix were removed from _load_model(), this test would fail
    with the same ModuleNotFoundError the real repo produced.
    """
    repo_path = tmp_path / "SalsaNext"
    _write_fake_salsanext_repo(repo_path)

    config_path = tmp_path / "arch_cfg.yaml"
    config_path.write_text(_REAL_ARCH_CFG_YAML, encoding="utf-8")
    (tmp_path / "data_cfg.yaml").write_text(_REAL_DATA_CFG_YAML, encoding="utf-8")

    checkpoint_path = tmp_path / "SalsaNext_checkpoint"
    checkpoint_path.touch()

    predictor = SalsaNextPredictor(
        repo_path=repo_path,
        checkpoint_path=checkpoint_path,
        config_path=config_path,
        device="cpu",
        strict=True,
    )
    # Must not raise ModuleNotFoundError / RuntimeError.
    predictor._load_model()
    assert predictor.model is not None

    modules_dir = str((repo_path / "train" / "tasks" / "semantic" / "modules").resolve())
    assert modules_dir in sys.path


def test_torch_load_uses_weights_only_false(tmp_path: Path) -> None:
    """torch.load must be called with weights_only=False.

    Regression test for the real _pickle.UnpicklingError (confirmed against
    the actual downloaded checkpoint, 2026-09-11): PyTorch 2.6 flipped
    torch.load's weights_only default to True, which rejects this legacy
    SalsaNext checkpoint's pickled numpy scalar types. A future refactor
    dropping this kwarg would silently reintroduce that load failure.
    """
    repo_path = tmp_path / "SalsaNext"
    _write_fake_salsanext_repo(repo_path)

    config_path = tmp_path / "arch_cfg.yaml"
    config_path.write_text(_REAL_ARCH_CFG_YAML, encoding="utf-8")
    (tmp_path / "data_cfg.yaml").write_text(_REAL_DATA_CFG_YAML, encoding="utf-8")

    checkpoint_path = tmp_path / "SalsaNext_checkpoint"
    checkpoint_path.touch()

    predictor = SalsaNextPredictor(
        repo_path=repo_path,
        checkpoint_path=checkpoint_path,
        config_path=config_path,
        device="cpu",
        strict=True,
    )

    snp.torch.load.reset_mock()
    predictor._load_model()

    assert snp.torch.load.called, "torch.load was never called"
    _, kwargs = snp.torch.load.call_args
    assert kwargs.get("weights_only") is False, (
        "torch.load must be called with weights_only=False for this legacy "
        f"checkpoint; got weights_only={kwargs.get('weights_only')!r}"
    )


def test_module_prefix_is_stripped_before_load_state_dict(tmp_path: Path) -> None:
    """DataParallel "module." prefix on checkpoint keys must be stripped.

    Regression test for the real RuntimeError (confirmed against the actual
    downloaded checkpoint, 2026-09-11): all 312 of its state_dict keys carry
    a "module." prefix (from being saved via nn.DataParallel), which a bare
    (non-DataParallel-wrapped) model reports as both missing and unexpected
    if loaded unmodified. Uses a small synthetic state_dict with a few
    "module."-prefixed keys, and a fake model (track_state_dict=True) that
    records exactly what it was called with -- this asserts the
    prefix-stripping happened before the call, not that PyTorch's own
    load_state_dict logic works (that's PyTorch's own test suite's job).
    """
    repo_path = tmp_path / "SalsaNext"
    _write_fake_salsanext_repo(repo_path, track_state_dict=True)

    config_path = tmp_path / "arch_cfg.yaml"
    config_path.write_text(_REAL_ARCH_CFG_YAML, encoding="utf-8")
    (tmp_path / "data_cfg.yaml").write_text(_REAL_DATA_CFG_YAML, encoding="utf-8")

    checkpoint_path = tmp_path / "SalsaNext_checkpoint"
    checkpoint_path.touch()

    predictor = SalsaNextPredictor(
        repo_path=repo_path,
        checkpoint_path=checkpoint_path,
        config_path=config_path,
        device="cpu",
        strict=True,
    )

    # A synthetic checkpoint whose state_dict keys are "module."-prefixed,
    # matching how a real nn.DataParallel-saved checkpoint looks.
    fake_state_dict = {
        "module.downCntx.conv1.weight": "weight_tensor_1",
        "module.downCntx.conv1.bias": "bias_tensor_1",
        "module.resBlock1.conv1.weight": "weight_tensor_2",
    }
    # snp.torch is a single module-level MagicMock shared by every test in
    # this file (set once in setup_module), so return_value set here would
    # leak into whichever test runs next -- reset it back afterward.
    snp.torch.load.return_value = {"state_dict": fake_state_dict}
    try:
        predictor._load_model()

        received_keys = set(predictor.model.received_state_dict.keys())
        assert received_keys == {
            "downCntx.conv1.weight",
            "downCntx.conv1.bias",
            "resBlock1.conv1.weight",
        }
        assert not any(k.startswith("module.") for k in received_keys)
    finally:
        snp.torch.load.reset_mock(return_value=True)


def test_missing_data_cfg_raises_error(tmp_path: Path) -> None:
    """data_cfg.yaml missing next to config_path must fail loudly, not silently."""
    repo_path = tmp_path / "SalsaNext"
    _write_fake_salsanext_repo(repo_path)

    config_path = tmp_path / "arch_cfg.yaml"
    config_path.write_text(_REAL_ARCH_CFG_YAML, encoding="utf-8")
    # Deliberately do NOT create data_cfg.yaml alongside it.

    checkpoint_path = tmp_path / "SalsaNext_checkpoint"
    checkpoint_path.touch()

    predictor = SalsaNextPredictor(
        repo_path=repo_path,
        checkpoint_path=checkpoint_path,
        config_path=config_path,
        device="cpu",
        strict=True,
    )
    with pytest.raises(RuntimeError, match="data_cfg.yaml not found"):
        predictor._load_model()


def test_device_resolver(tmp_path: Path) -> None:
    """Test device resolution to CPU when CUDA is not guaranteed or forced."""
    repo_path = tmp_path / "SalsaNext"
    repo_path.mkdir()
    checkpoint_path = tmp_path / "checkpoint.pth"
    checkpoint_path.touch()
    config_path = tmp_path / "arch_cfg.yaml"
    config_path.touch()

    # Even if torch is not installed or CUDA is unavailable, forcing 'cpu' should work.
    predictor = SalsaNextPredictor(
        repo_path=repo_path,
        checkpoint_path=checkpoint_path,
        config_path=config_path,
        device="cpu",
        strict=True,
    )
    assert predictor.device.type == "cpu"


def test_adaptation_options_stored_on_instance(tmp_path: Path) -> None:
    """Test that domain adaptation parameters (FOV and intensity) are stored on the instance."""
    repo_path = tmp_path / "SalsaNext"
    repo_path.mkdir()
    checkpoint_path = tmp_path / "checkpoint.pth"
    checkpoint_path.touch()
    config_path = tmp_path / "arch_cfg.yaml"
    config_path.touch()

    predictor = SalsaNextPredictor(
        repo_path=repo_path,
        checkpoint_path=checkpoint_path,
        config_path=config_path,
        device="cpu",
        strict=True,
        fov_up=17.02,
        fov_down=-16.44,
        rescale_intensity=True,
    )
    assert predictor.fov_up == 17.02
    assert predictor.fov_down == -16.44
    assert predictor.rescale_intensity is True

