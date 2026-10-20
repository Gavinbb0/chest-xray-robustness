"""Dataset split and image degradation transformations."""

from __future__ import annotations

from io import BytesIO
from pathlib import Path
import random
import shutil

import numpy as np
import torch
from PIL import Image, ImageEnhance, ImageFilter
from torchvision import datasets, transforms

CLASSES = ("NORMAL", "PNEUMONIA")
MEAN = (0.485, 0.456, 0.406)
STD = (0.229, 0.224, 0.225)
EXTENSIONS = {".jpeg", ".jpg", ".png"}
LEVELS = {
    "blur": (0.5, 1.0, 2.0, 4.0),
    "noise": (0.02, 0.05, 0.10, 0.15),
    "contrast": (0.8, 0.6, 0.4, 0.2),
    "brightness": (0.85, 0.70, 0.50, 0.30),
    "jpeg": (80, 50, 25, 10),
}


def check_classes(dataset: datasets.ImageFolder) -> None:
    if dataset.classes != list(CLASSES):
        raise ValueError(f"Expected class folders {CLASSES}, got {dataset.classes}")


def image_dataset(folder: Path, transform) -> datasets.ImageFolder:
    if not folder.is_dir():
        raise FileNotFoundError(f"Missing dataset folder: {folder}")
    dataset = datasets.ImageFolder(folder, transform=transform)
    check_classes(dataset)
    return dataset


def clean_transform():
    return transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(MEAN, STD),
    ])


def raw_transform():
    return transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor()])


def _noise_pil(image: Image.Image, sigma: float) -> Image.Image:
    array = np.asarray(image.convert("RGB"), dtype=np.float32) / 255.0
    noise = np.random.normal(0, sigma, (*array.shape[:2], 1))
    return Image.fromarray((np.clip(array + noise, 0, 1) * 255).astype(np.uint8))


class RandomDegradation:
    """One of clean, blur, noise, contrast or brightness per training image."""

    def __call__(self, image: Image.Image) -> Image.Image:
        choice = random.choice(("clean", "blur", "noise", "contrast", "brightness"))
        if choice == "blur":
            return image.filter(ImageFilter.GaussianBlur(radius=random.uniform(0.5, 4.0)))
        if choice == "noise":
            return _noise_pil(image, random.uniform(0.02, 0.15))
        if choice == "contrast":
            return ImageEnhance.Contrast(image).enhance(random.uniform(0.2, 0.8))
        if choice == "brightness":
            return ImageEnhance.Brightness(image).enhance(random.uniform(0.3, 0.85))
        return image


def training_transform(robust: bool):
    operations = [transforms.Resize((224, 224))]
    if robust:
        operations.append(RandomDegradation())
    return transforms.Compose([
        *operations,
        transforms.ToTensor(),
        transforms.Normalize(MEAN, STD),
    ])


def degraded_transform(kind: str, level: int):
    """PIL degradations follow resize; noise is added to tensors in apply_noise."""
    if kind not in LEVELS or level not in range(1, 5):
        raise ValueError(f"Unknown degradation: {kind} level {level}")
    value = LEVELS[kind][level - 1]
    operations = [transforms.Resize((224, 224))]
    if kind == "blur":
        operations.append(transforms.Lambda(
            lambda image: image.filter(ImageFilter.GaussianBlur(radius=value))))
    elif kind == "contrast":
        operations.append(transforms.Lambda(
            lambda image: ImageEnhance.Contrast(image).enhance(value)))
    elif kind == "brightness":
        operations.append(transforms.Lambda(
            lambda image: ImageEnhance.Brightness(image).enhance(value)))
    elif kind == "jpeg":
        def compress(image):
            buffer = BytesIO()
            image.save(buffer, format="JPEG", quality=value)
            buffer.seek(0)
            with Image.open(buffer) as decoded:
                return decoded.convert("RGB").copy()
        operations.append(transforms.Lambda(compress))
    operations.append(transforms.ToTensor())
    return transforms.Compose(operations)


def apply_noise(images: torch.Tensor, sigma: float, generator: torch.Generator):
    # Same random offset in all three RGB channels, as in the original scripts.
    noise = torch.randn(
        (images.shape[0], 1, images.shape[2], images.shape[3]),
        generator=generator, dtype=images.dtype,
    ) * sigma
    return (images + noise).clamp(0, 1)


def prepare_split(source: Path, destination: Path, val_fraction: float, seed: int):
    """Copy source/train to train/val and source/test to test; never split test."""
    if not 0 < val_fraction < 1:
        raise ValueError("val_fraction must be between 0 and 1")
    if destination.exists() and any(destination.iterdir()):
        raise FileExistsError(f"Destination must be empty: {destination}")
    planned = []
    rng = random.Random(seed)
    for cls in CLASSES:
        train_dir, test_dir = source / "train" / cls, source / "test" / cls
        if not train_dir.is_dir() or not test_dir.is_dir():
            raise FileNotFoundError(f"Expected {train_dir} and {test_dir}")
        train_files = sorted(p for p in train_dir.iterdir() if p.suffix.lower() in EXTENSIONS)
        test_files = sorted(p for p in test_dir.iterdir() if p.suffix.lower() in EXTENSIONS)
        if len(train_files) < 2 or not test_files:
            raise ValueError(f"{cls}: need at least two train images and one test image")
        rng.shuffle(train_files)
        count = max(1, min(len(train_files) - 1, round(len(train_files) * val_fraction)))
        planned.extend((part, cls, item) for part, items in (
            ("train", train_files[count:]), ("val", train_files[:count]), ("test", test_files)
        ) for item in items)
    for part, cls, item in planned:
        target = destination / part / cls
        target.mkdir(parents=True, exist_ok=True)
        shutil.copy2(item, target / item.name)
    return {part: sum(1 for entry in planned if entry[0] == part)
            for part in ("train", "val", "test")}
