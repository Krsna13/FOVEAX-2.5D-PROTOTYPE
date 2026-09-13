"""FOVEAX Phase 12 - Fine-tune SalsaNext on real RELLIS-3D data.

Starts from the official SemanticKITTI-pretrained SalsaNext checkpoint and
fine-tunes it on RELLIS-3D in the FOVEAX 8-class taxonomy.

Reused from upstream SalsaNext (imported from the cloned repo, not
reimplemented): the `SalsaNext` network, the `Lovasz_softmax` loss, the
`warmupLR` scheduler, the weighted-NLL objective, the class-weight formula
`w = 1/(content + epsilon_w)`, the SGD configuration, and upstream's
epoch->step learning-rate decay conversion.

Deliberate deviations from upstream, each for a real reason:

  * Output head is `NUM_FOVEAX_CLASSES` (8), not SemanticKITTI's 20. RELLIS-3D
    is labelled in its own ontology, and this project consumes FOVEAX classes
    everywhere downstream. The 310 backbone tensors transfer unchanged; only
    `logits.weight`/`logits.bias` are re-initialised, because a 20-class head
    cannot be reshaped into an 8-class one. This is a taxonomy change, not a
    from-scratch restart -- see the report in docs/validation_results.md.
  * `ignore_index` is 7 (FOVEAX UNKNOWN), not upstream's 0. FOVEAX class 0 is
    DRIVABLE_GROUND, a real supervised class; ignoring it would discard the
    most safety-relevant surface in the dataset.
  * No data augmentation. Training preprocessing is kept byte-identical to
    the inference path so the fine-tune transfers; see
    src/training/rellis3d_dataset.py.

Checkpoints are written every epoch and training is resumable (--resume).
"""

from __future__ import annotations

import argparse
import importlib
import json
import sys
import time
from pathlib import Path

import numpy as np
import yaml

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from src.perception.semantic_labels import FOVEAX_CLASSES, NUM_FOVEAX_CLASSES
from src.training.rellis3d_dataset import (
    IGNORE_INDEX,
    Rellis3DRangeDataset,
    build_splits,
    class_weights_from_counts,
    compute_class_frequencies,
)

DEFAULT_RELLIS_ROOT = Path("C:/dev/data/rellis3d")


def _import_upstream(repo_path: Path):
    """Import SalsaNext, Lovasz_softmax and warmupLR from the cloned repo.

    SalsaNext.py does `import __init__ as booger`, a bare import of the
    literal module name "__init__", so its own modules/ directory must be
    directly on sys.path -- the same requirement handled in
    src/perception/salsanext_predictor.py::_load_model.
    """
    repo = Path(repo_path).resolve()
    for p in (
        str(repo),
        str(repo / "train"),
        str(repo / "train" / "tasks" / "semantic" / "modules"),
    ):
        if p not in sys.path:
            sys.path.insert(0, p)
    # 'imp' was removed in Python 3.12+; SalsaNext.py has an unused import of
    # it. No-op on this project's pinned 3.11 interpreter.
    sys.modules.setdefault("imp", importlib)

    from train.common.warmupLR import warmupLR
    from train.tasks.semantic.modules.Lovasz_Softmax import Lovasz_softmax
    from train.tasks.semantic.modules.SalsaNext import SalsaNext

    return SalsaNext, Lovasz_softmax, warmupLR


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Fine-tune SalsaNext on RELLIS-3D in FOVEAX taxonomy."
    )
    p.add_argument("--rellis-root", type=Path, default=DEFAULT_RELLIS_ROOT)
    p.add_argument("--salsanext-repo", type=Path, default=Path("external/SalsaNext"))
    p.add_argument(
        "--pretrained",
        type=Path,
        default=Path("models/salsanext/pretrained/pretrained/SalsaNext"),
        help="SemanticKITTI-pretrained checkpoint used to initialise the backbone.",
    )
    p.add_argument(
        "--config",
        type=Path,
        default=Path("models/salsanext/pretrained/pretrained/arch_cfg.yaml"),
    )
    p.add_argument(
        "--out-dir", type=Path, default=Path("models/salsanext/rellis3d_finetuned")
    )
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--batch-size", type=int, default=2)
    p.add_argument("--lr", type=float, default=0.01)
    p.add_argument("--momentum", type=float, default=0.9)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--lr-decay", type=float, default=0.99, help="Per-epoch LR decay.")
    p.add_argument("--wup-epochs", type=float, default=1.0)
    p.add_argument("--train-stride", type=int, default=2)
    p.add_argument("--val-stride", type=int, default=10)
    p.add_argument("--workers", type=int, default=3)
    p.add_argument("--seed", type=int, default=1337)
    p.add_argument(
        "--resume",
        type=str,
        default=None,
        help="Checkpoint path, or 'auto' to resume from <out-dir>/last.pt if present.",
    )
    p.add_argument(
        "--max-steps-per-epoch",
        type=int,
        default=0,
        help="Debug: cap training steps per epoch (0 = no cap).",
    )
    return p.parse_args(argv)


def load_backbone(model, pretrained_path: Path, torch) -> tuple[int, list[str]]:
    """Transfer every pretrained tensor whose name and shape still match.

    weights_only=False is required for this legacy checkpoint (pickled numpy
    scalars outside torch's safe-unpickling allowlist). Same provenance
    justification as src/perception/salsanext_predictor.py: the paper authors'
    official Google Drive release.
    """
    ckpt = torch.load(pretrained_path, map_location="cpu", weights_only=False)
    sd = ckpt["state_dict"] if "state_dict" in ckpt else ckpt
    sd = {k.removeprefix("module."): v for k, v in sd.items()}

    own = model.state_dict()
    transferable = {k: v for k, v in sd.items() if k in own and own[k].shape == v.shape}
    skipped = sorted(k for k in sd if k not in transferable)
    model.load_state_dict(transferable, strict=False)
    return len(transferable), skipped


def evaluate(model, loader, device, torch) -> dict:
    """Per-pixel accuracy and per-class recall over supervised pixels."""
    model.eval()
    confusion = np.zeros((NUM_FOVEAX_CLASSES, NUM_FOVEAX_CLASSES), dtype=np.int64)
    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device, non_blocking=True)
            preds = model(images).argmax(dim=1).cpu().numpy()
            labels = labels.numpy()
            valid = labels != IGNORE_INDEX
            if not valid.any():
                continue
            np.add.at(confusion, (labels[valid], preds[valid]), 1)

    total = confusion.sum()
    correct = np.trace(confusion)
    per_class = {}
    for cid in range(NUM_FOVEAX_CLASSES):
        n = confusion[cid].sum()
        if n > 0:
            per_class[FOVEAX_CLASSES[cid]] = {
                "recall": float(confusion[cid, cid] / n),
                "n_pixels": int(n),
            }
    return {
        "accuracy": float(correct / total) if total else 0.0,
        "n_pixels": int(total),
        "per_class": per_class,
    }


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)

    import torch
    from torch.utils.data import DataLoader

    SalsaNext, Lovasz_softmax, warmupLR = _import_upstream(args.salsanext_repo)

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        print("WARNING: CUDA unavailable -- fine-tuning on CPU is impractically slow.")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    arch_cfg = yaml.safe_load(open(args.config))

    splits = build_splits(
        args.rellis_root, train_stride=args.train_stride, val_stride=args.val_stride
    )
    train_refs, val_refs = splits["train"], splits["val"]
    print(
        f"Split: train={len(train_refs)} val={len(val_refs)} "
        f"test={len(splits['test'])} (test sequence held out entirely)"
    )

    train_ds = Rellis3DRangeDataset(args.rellis_root, train_refs, arch_cfg)
    val_ds = Rellis3DRangeDataset(args.rellis_root, val_refs, arch_cfg)
    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.workers,
        pin_memory=True,
        drop_last=True,
        persistent_workers=args.workers > 0,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=True,
        persistent_workers=args.workers > 0,
    )

    # Class weights: upstream's w = 1/(content + epsilon_w), computed on this
    # project's real training split rather than SemanticKITTI's shipped
    # content table (different dataset, different distribution).
    freq_path = Path("outputs/phase12/class_frequencies.json")
    if freq_path.is_file():
        payload = json.load(open(freq_path))
        counts = np.array([c["pixels"] for c in payload["classes"]], dtype=np.int64)
        print(f"Loaded class frequencies from {freq_path} ({payload['n_frames']} frames)")
    else:
        print("Scanning training split for class frequencies...")
        counts = compute_class_frequencies(
            args.rellis_root, train_refs, NUM_FOVEAX_CLASSES, arch_cfg, progress=True
        )
        freq_path.parent.mkdir(parents=True, exist_ok=True)
        json.dump(
            {
                "n_frames": len(train_refs),
                "total_supervised_pixels": int(counts.sum()),
                "ignore_index": IGNORE_INDEX,
                "classes": [
                    {
                        "class_id": i,
                        "name": FOVEAX_CLASSES[i],
                        "pixels": int(counts[i]),
                        "fraction": float(counts[i] / counts.sum()),
                    }
                    for i in range(NUM_FOVEAX_CLASSES)
                ],
            },
            open(freq_path, "w"),
            indent=2,
        )
    weights = class_weights_from_counts(counts)
    print("Class weights: " + ", ".join(
        f"{FOVEAX_CLASSES[i]}={weights[i]:.2f}" for i in range(NUM_FOVEAX_CLASSES)
    ))

    model = SalsaNext(NUM_FOVEAX_CLASSES)
    n_transferred, skipped = load_backbone(model, args.pretrained, torch)
    print(f"Backbone transfer: {n_transferred} tensors loaded, re-initialised: {skipped}")
    model.to(device)

    # SalsaNext.forward already applies softmax, so the objective is NLL on
    # log-probabilities -- exactly upstream's formulation.
    criterion = torch.nn.NLLLoss(
        weight=torch.from_numpy(weights).to(device), ignore_index=IGNORE_INDEX
    ).to(device)
    lovasz = Lovasz_softmax(ignore=IGNORE_INDEX).to(device)
    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=args.lr,
        momentum=args.momentum,
        weight_decay=args.weight_decay,
    )

    steps_per_epoch = len(train_loader)
    if args.max_steps_per_epoch:
        steps_per_epoch = min(steps_per_epoch, args.max_steps_per_epoch)
    scheduler = warmupLR(
        optimizer=optimizer,
        lr=args.lr,
        warmup_steps=max(1, int(args.wup_epochs * steps_per_epoch)),
        momentum=args.momentum,
        decay=args.lr_decay ** (1 / max(1, steps_per_epoch)),
    )

    start_epoch = 0
    history: list[dict] = []
    best_acc = -1.0

    resume_path = None
    if args.resume == "auto":
        candidate = args.out_dir / "last.pt"
        resume_path = candidate if candidate.is_file() else None
        if resume_path is None:
            print("--resume auto: no last.pt found, starting fresh.")
    elif args.resume:
        resume_path = Path(args.resume)

    if resume_path is not None:
        ckpt = torch.load(resume_path, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["state_dict"])
        optimizer.load_state_dict(ckpt["optimizer"])
        start_epoch = ckpt["epoch"] + 1
        history = ckpt.get("history", [])
        best_acc = ckpt.get("best_acc", -1.0)
        # warmupLR wraps a CyclicLR whose state isn't captured by the base
        # scheduler's state_dict, so it is replayed forward instead. Stepping
        # only touches optimizer LR values -- no model compute.
        for _ in range(start_epoch * steps_per_epoch):
            scheduler.step()
        print(f"Resumed from {resume_path} at epoch {start_epoch}")

    print(
        f"Training: {args.epochs} epochs, batch={args.batch_size}, lr={args.lr}, "
        f"{steps_per_epoch} steps/epoch, device={device}"
    )

    for epoch in range(start_epoch, args.epochs):
        model.train()
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats()
        t_epoch = time.time()
        running = 0.0
        n_steps = 0

        for step, (images, labels) in enumerate(train_loader):
            if args.max_steps_per_epoch and step >= args.max_steps_per_epoch:
                break
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            output = model(images)
            loss = criterion(torch.log(output.clamp(min=1e-8)), labels) + lovasz(
                output, labels
            )

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()

            running += float(loss.item())
            n_steps += 1
            if n_steps % 100 == 0:
                print(
                    f"  epoch {epoch} step {n_steps}/{steps_per_epoch} "
                    f"loss={running / n_steps:.4f} "
                    f"lr={optimizer.param_groups[0]['lr']:.5f}",
                    flush=True,
                )

        train_time = time.time() - t_epoch
        peak_mb = (
            torch.cuda.max_memory_allocated() / 1024**2 if device.type == "cuda" else 0.0
        )

        t_val = time.time()
        metrics = evaluate(model, val_loader, device, torch)
        val_time = time.time() - t_val

        record = {
            "epoch": epoch,
            "train_loss": running / max(1, n_steps),
            "val_accuracy": metrics["accuracy"],
            "val_per_class": metrics["per_class"],
            "lr": optimizer.param_groups[0]["lr"],
            "train_seconds": train_time,
            "val_seconds": val_time,
            "peak_vram_mb": peak_mb,
        }
        history.append(record)
        print(
            f"EPOCH {epoch}: loss={record['train_loss']:.4f} "
            f"val_acc={metrics['accuracy'] * 100:.2f}% "
            f"train={train_time / 60:.1f}min val={val_time:.0f}s "
            f"peak_vram={peak_mb:.0f}MB",
            flush=True,
        )
        for name, stats in metrics["per_class"].items():
            print(f"    {name:<16} recall={stats['recall'] * 100:6.2f}%  n={stats['n_pixels']:,}")

        payload = {
            "state_dict": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "epoch": epoch,
            "history": history,
            "best_acc": max(best_acc, metrics["accuracy"]),
            "num_classes": NUM_FOVEAX_CLASSES,
            "taxonomy": "foveax",
            "args": {k: str(v) for k, v in vars(args).items()},
        }
        torch.save(payload, args.out_dir / f"epoch_{epoch:03d}.pt")
        torch.save(payload, args.out_dir / "last.pt")
        if metrics["accuracy"] > best_acc:
            best_acc = metrics["accuracy"]
            torch.save(payload, args.out_dir / "best.pt")
            print(f"    new best val accuracy -> {args.out_dir / 'best.pt'}")

        json.dump(history, open(args.out_dir / "history.json", "w"), indent=2)

    print(f"\nDone. Best val accuracy: {best_acc * 100:.2f}%")
    print(f"Checkpoints: {args.out_dir}/epoch_XXX.pt, last.pt, best.pt")


if __name__ == "__main__":
    main()
