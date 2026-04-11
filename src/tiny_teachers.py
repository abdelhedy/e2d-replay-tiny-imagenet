"""
TinyImageNet teacher loaders used by the CL E2D pipeline.

The zeyuanyin/tiny-imagenet checkpoints are torchvision ResNets adapted for
64x64 inputs:
  - conv1: 3x3, stride 1, padding 1
  - maxpool: identity
  - fc: 200 TinyImageNet classes
"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable, List, Sequence, Tuple

import torch
import torch.nn as nn
import torchvision.models as tv_models
from torch.serialization import safe_globals


def parse_teacher_checkpoints(raw: str | Sequence[str] | None) -> List[str]:
    """Parse comma-separated checkpoint paths while preserving explicit order."""
    if raw is None:
        return []
    if isinstance(raw, str):
        return [p.strip() for p in raw.split(",") if p.strip()]
    return [str(p).strip() for p in raw if str(p).strip()]


def _resolve_tiny_paths(train_data_path: str | Path) -> tuple[Path, Path]:
    """Return (tiny_root, train_dir) for either .../tiny-imagenet-200 or .../train."""
    path = Path(train_data_path)
    if path.name == "train":
        return path.parent, path
    return path, path / "train"


def tiny_avalanche_to_imagefolder_indices(
    train_data_path: str | Path,
    num_classes: int = 200,
) -> List[int] | None:
    """
    Map Avalanche TinyImageNet labels to ImageFolder/torchvision teacher labels.

    Avalanche's TinyImageNet dataset follows wnids.txt order. The zeyuanyin
    checkpoints are trained with torchvision's ImageFolder-style class order,
    which sorts class-folder names. For TinyImageNet those two orders differ.
    """
    tiny_root, train_dir = _resolve_tiny_paths(train_data_path)
    wnids_path = tiny_root / "wnids.txt"
    if not wnids_path.exists() or not train_dir.exists():
        return None

    avalanche_wnids = [
        line.strip()
        for line in wnids_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    imagefolder_wnids = sorted(
        p.name for p in train_dir.iterdir() if p.is_dir()
    )
    imagefolder_index = {
        wnid: idx for idx, wnid in enumerate(imagefolder_wnids)
    }

    missing = [wnid for wnid in avalanche_wnids if wnid not in imagefolder_index]
    if missing:
        raise ValueError(
            "TinyImageNet train folders are missing wnids from wnids.txt: "
            + ", ".join(missing[:10])
        )

    limit = min(int(num_classes), len(avalanche_wnids))
    return [imagefolder_index[wnid] for wnid in avalanche_wnids[:limit]]


def maybe_print_tiny_class_mapping(
    mapping: Sequence[int] | None,
    prefix: str,
    max_items: int = 10,
) -> None:
    """Small debug print for TinyImageNet label-space alignment."""
    if mapping is None:
        return
    preview = ", ".join(f"{i}->{mapping[i]}" for i in range(min(max_items, len(mapping))))
    identity = all(i == v for i, v in enumerate(mapping))
    status = "identity" if identity else "remapped"
    print(f"{prefix} Tiny class order: {status} ({preview})")


def infer_tiny_resnet_arch(path: str, checkpoint: dict | None = None) -> str:
    """Infer resnet18/resnet50 from checkpoint args first, then filename."""
    args = checkpoint.get("args") if isinstance(checkpoint, dict) else None
    model_name = getattr(args, "model", None)
    if model_name in {"resnet18", "resnet50"}:
        return model_name

    name = Path(path).name.lower()
    if "rn18" in name or "resnet18" in name:
        return "resnet18"
    if "rn50" in name or "resnet50" in name:
        return "resnet50"
    raise ValueError(
        f"Could not infer TinyImageNet teacher architecture from {path!r}. "
        "Use a filename containing rn18/resnet18 or rn50/resnet50."
    )


def build_tiny_resnet(arch: str, num_classes: int = 200) -> nn.Module:
    if arch not in {"resnet18", "resnet50"}:
        raise ValueError(f"Unsupported TinyImageNet teacher arch: {arch!r}")

    model = tv_models.__dict__[arch](weights=None)
    model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
    model.maxpool = nn.Identity()
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model


def _load_checkpoint(path: str) -> dict:
    # The checkpoints contain argparse.Namespace in metadata. Allowlisting that
    # keeps torch.load in weights_only mode instead of falling back to unsafe
    # arbitrary pickle loading.
    with safe_globals([argparse.Namespace]):
        checkpoint = torch.load(path, map_location="cpu")
    if not isinstance(checkpoint, dict):
        raise TypeError(f"Expected a dict checkpoint at {path!r}, got {type(checkpoint)!r}")
    return checkpoint


def _checkpoint_state_dict(checkpoint: dict) -> dict:
    state = checkpoint.get("model")
    if state is None:
        state = checkpoint.get("state_dict", checkpoint.get("model_state_dict", checkpoint))
    if not isinstance(state, dict):
        raise TypeError("Checkpoint does not contain a model state dict")

    # Be permissive with DataParallel checkpoints.
    return {
        (k[len("module.") :] if k.startswith("module.") else k): v
        for k, v in state.items()
    }


def load_tiny_resnet_teacher(
    checkpoint_path: str,
    device: torch.device | str | None = None,
    num_classes: int = 200,
) -> nn.Module:
    checkpoint = _load_checkpoint(checkpoint_path)
    arch = infer_tiny_resnet_arch(checkpoint_path, checkpoint)
    model = build_tiny_resnet(arch, num_classes=num_classes)
    state = _checkpoint_state_dict(checkpoint)
    model.load_state_dict(state, strict=True)

    model.teacher_name = f"{arch}:{Path(checkpoint_path).name}"  # debug/printing aid
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    if device is not None:
        model = model.to(device)
    return model


def load_tiny_resnet_student(
    checkpoint_path: str,
    num_classes: int,
    expected_arch: str | None = None,
) -> nn.Module:
    """
    Load a TinyImageNet ResNet checkpoint as a student backbone.

    The checkpoint head is 200-way TinyImageNet. We load it strictly first so
    architecture mismatches fail loudly, then replace only fc with the CL head.
    """
    checkpoint = _load_checkpoint(checkpoint_path)
    arch = infer_tiny_resnet_arch(checkpoint_path, checkpoint)
    if expected_arch is not None and arch != expected_arch:
        raise ValueError(
            f"Tiny student checkpoint is {arch}, but --student-arch is {expected_arch}."
        )

    model = build_tiny_resnet(arch, num_classes=200)
    state = _checkpoint_state_dict(checkpoint)
    model.load_state_dict(state, strict=True)
    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, num_classes)
    print(
        f"[Student] TinyImageNet checkpoint loaded: {arch}:{Path(checkpoint_path).name}. "
        f"Replaced 200-way fc with {num_classes}-way fc."
    )
    return model


def load_tiny_teacher_ensemble(
    checkpoint_paths: Iterable[str],
    device: torch.device | str | None = None,
    num_classes: int = 200,
) -> List[nn.Module]:
    paths = list(checkpoint_paths)
    if not paths:
        raise ValueError("No TinyImageNet teacher checkpoints were provided.")

    teachers = [
        load_tiny_resnet_teacher(path, device=device, num_classes=num_classes)
        for path in paths
    ]
    names = [getattr(t, "teacher_name", t.__class__.__name__) for t in teachers]
    print(f"[TinyTeachers] Loaded {len(teachers)} teacher(s): {names}")
    return teachers
