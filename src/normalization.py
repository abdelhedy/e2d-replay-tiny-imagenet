"""Shared normalization presets for the E2D TinyImageNet pipeline."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

from torchvision import transforms
from torchvision.transforms import InterpolationMode


@dataclass(frozen=True)
class NormalizationSpec:
    name: str
    mean: tuple[float, float, float]
    std: tuple[float, float, float]

    @property
    def is_identity(self) -> bool:
        return self.mean == (0.0, 0.0, 0.0) and self.std == (1.0, 1.0, 1.0)


NORMALIZATION_PRESETS: Dict[str, NormalizationSpec] = {
    "imagenet": NormalizationSpec(
        "imagenet",
        (0.485, 0.456, 0.406),
        (0.229, 0.224, 0.225),
    ),
    # Common TinyImageNet train-set statistics used in many public baselines.
    "tiny": NormalizationSpec(
        "tiny",
        (0.4802, 0.4481, 0.3975),
        (0.2302, 0.2265, 0.2262),
    ),
    # Avalanche SplitTinyImageNet default normalization.
    "avalanche": NormalizationSpec(
        "avalanche",
        (0.4914, 0.4822, 0.4465),
        (0.2023, 0.1994, 0.2010),
    ),
    "none": NormalizationSpec(
        "none",
        (0.0, 0.0, 0.0),
        (1.0, 1.0, 1.0),
    ),
}


def normalization_names() -> list[str]:
    return list(NORMALIZATION_PRESETS.keys())


def get_normalization(name: str | NormalizationSpec) -> NormalizationSpec:
    if isinstance(name, NormalizationSpec):
        return name
    key = str(name).lower()
    if key not in NORMALIZATION_PRESETS:
        choices = ", ".join(normalization_names())
        raise ValueError(f"Unknown normalization {name!r}. Choices: {choices}")
    return NORMALIZATION_PRESETS[key]


def describe_normalization(name: str | NormalizationSpec) -> str:
    spec = get_normalization(name)
    return f"{spec.name} mean={list(spec.mean)} std={list(spec.std)}"


def normalize_transform(name: str | NormalizationSpec) -> transforms.Normalize:
    spec = get_normalization(name)
    return transforms.Normalize(mean=spec.mean, std=spec.std)


def maybe_normalize(name: str | NormalizationSpec) -> transforms.Normalize:
    # Even "none" returns Normalize(0, 1). Keeping an explicit transform makes
    # diagnostics and pipeline shape identical across presets.
    return normalize_transform(name)


def make_tiny_train_transform(name: str | NormalizationSpec) -> transforms.Compose:
    return transforms.Compose([
        transforms.RandomCrop(64, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        maybe_normalize(name),
    ])


def make_tiny_eval_transform(name: str | NormalizationSpec) -> transforms.Compose:
    return transforms.Compose([
        transforms.Resize(64),
        transforms.CenterCrop(64),
        transforms.ToTensor(),
        maybe_normalize(name),
    ])


def make_replay_transform(
    image_size: int,
    name: str | NormalizationSpec,
) -> transforms.Compose:
    return transforms.Compose([
        transforms.Resize(image_size),
        transforms.RandomCrop(image_size, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        maybe_normalize(name),
    ])


def make_replay_eval_transform(
    image_size: int,
    name: str | NormalizationSpec,
) -> transforms.Compose:
    return transforms.Compose([
        transforms.Resize(image_size),
        transforms.CenterCrop(image_size),
        transforms.ToTensor(),
        maybe_normalize(name),
    ])


def make_relabel_transform(
    image_size: int,
    name: str | NormalizationSpec,
) -> transforms.Compose:
    return transforms.Compose([
        transforms.Resize(image_size),
        transforms.RandomResizedCrop(
            image_size,
            scale=(0.5, 1.0),
            interpolation=InterpolationMode.BILINEAR,
        ),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        maybe_normalize(name),
    ])


def as_float_lists(name: str | NormalizationSpec) -> tuple[list[float], list[float]]:
    spec = get_normalization(name)
    return list(spec.mean), list(spec.std)
