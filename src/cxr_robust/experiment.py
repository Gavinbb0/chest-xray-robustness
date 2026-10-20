"""Shared training, calibration and evaluation for both ResNet18 models."""

from __future__ import annotations

import csv
import json
from pathlib import Path
import random

import numpy as np
import torch
from sklearn.metrics import (
    accuracy_score, confusion_matrix, f1_score, recall_score, roc_auc_score, roc_curve,
)
from torch import nn
from torch.utils.data import DataLoader
from torchvision import models, transforms

from .data import (
    CLASSES, LEVELS, MEAN, STD, apply_noise, clean_transform, degraded_transform,
    image_dataset, raw_transform, training_transform,
)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def device_for(name: str) -> torch.device:
    if name == "auto":
        name = "cuda" if torch.cuda.is_available() else "cpu"
    if name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    return torch.device(name)


def build_model(pretrained: bool = False) -> nn.Module:
    weights = models.ResNet18_Weights.DEFAULT if pretrained else None
    model = models.resnet18(weights=weights)
    model.fc = nn.Linear(model.fc.in_features, 2)
    return model


def load_model(checkpoint: Path, device: torch.device) -> nn.Module:
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)
    model = build_model()
    # Source checkpoints are plain state_dict files; weights_only avoids pickle objects.
    model.load_state_dict(torch.load(checkpoint, map_location="cpu", weights_only=True))
    return model.to(device).eval()


def metadata_path(checkpoint: Path) -> Path:
    return checkpoint.with_suffix(".json")


def read_metadata(checkpoint: Path) -> dict:
    path = metadata_path(checkpoint)
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}


def write_metadata(checkpoint: Path, metadata: dict) -> None:
    metadata_path(checkpoint).write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def train_model(
    data_dir: Path, checkpoint: Path, robust: bool, epochs: int, batch_size: int,
    learning_rate: float, seed: int, device: torch.device, pretrained: bool,
) -> None:
    if epochs < 1 or batch_size < 1 or learning_rate <= 0:
        raise ValueError("epochs, batch_size and learning_rate must be positive")
    set_seed(seed)
    training = image_dataset(data_dir / "train", training_transform(robust))
    validation = image_dataset(data_dir / "val", clean_transform())
    train_loader = DataLoader(training, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(validation, batch_size=batch_size)
    model = build_model(pretrained).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.CrossEntropyLoss()
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    best_loss = float("inf")

    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            loss = criterion(model(images), labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * len(labels)
        model.eval()
        val_loss = 0.0
        correct = 0
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                output = model(images)
                val_loss += criterion(output, labels).item() * len(labels)
                correct += (output.argmax(1) == labels).sum().item()
        val_loss /= len(validation)
        print(f"Epoch {epoch+1:02d}/{epochs} | train loss {train_loss/len(training):.4f} "
              f"| val loss {val_loss:.4f} | val acc {correct/len(validation):.4f}")
        if val_loss < best_loss:
            best_loss = val_loss
            torch.save(model.state_dict(), checkpoint)
            write_metadata(checkpoint, {
                "variant": "robust" if robust else "baseline", "classes": list(CLASSES),
                "seed": seed, "epochs": epochs, "best_val_loss": best_loss,
                "threshold": None,
            })
    print(f"Saved best checkpoint: {checkpoint}")


def predict(
    model: nn.Module, loader: DataLoader, device: torch.device,
    noise_std: float = 0.0, seed: int = 42,
) -> tuple[list[int], list[float]]:
    labels_all, probs_all = [], []
    normalize = transforms.Normalize(MEAN, STD)
    generator = torch.Generator().manual_seed(seed)
    model.eval()
    with torch.no_grad():
        for images, labels in loader:
            if noise_std:
                images = apply_noise(images, noise_std, generator)
            output = model(normalize(images).to(device))
            labels_all.extend(labels.tolist())
            probs_all.extend(torch.softmax(output, dim=1)[:, 1].cpu().tolist())
    return labels_all, probs_all


def validation_threshold(
    data_dir: Path, checkpoint: Path, batch_size: int, device: torch.device,
) -> dict:
    model = load_model(checkpoint, device)
    loader = DataLoader(image_dataset(data_dir / "val", raw_transform()),
                        batch_size=batch_size)
    labels, probs = predict(model, loader, device)
    if len(set(labels)) != 2:
        raise ValueError("Validation set must contain both NORMAL and PNEUMONIA")
    fpr, tpr, thresholds = roc_curve(labels, probs)
    index = int(np.argmax(tpr - fpr))
    threshold = float(thresholds[index])
    metadata = read_metadata(checkpoint)
    metadata.update({"classes": list(CLASSES), "threshold": threshold,
                     "threshold_method": "validation Youden J"})
    write_metadata(checkpoint, metadata)
    result = {"threshold": threshold, "sensitivity": float(tpr[index]),
              "specificity": float(1 - fpr[index])}
    print(json.dumps(result, indent=2))
    return result


def get_threshold(checkpoint: Path, override: float | None) -> float:
    threshold = override if override is not None else read_metadata(checkpoint).get("threshold")
    if threshold is None:
        raise ValueError(f"No calibrated threshold for {checkpoint}. Run 'calibrate' "
                         "on validation data or pass --threshold.")
    if not 0 <= threshold <= 1:
        raise ValueError("threshold must be between 0 and 1")
    return float(threshold)


def metrics(labels: list[int], probs: list[float], threshold: float) -> dict:
    if len(set(labels)) != 2:
        raise ValueError("Evaluation set must contain both NORMAL and PNEUMONIA")
    predicted = [int(p >= threshold) for p in probs]
    tn, fp, fn, tp = confusion_matrix(labels, predicted, labels=[0, 1]).ravel()
    return {
        "AUROC": float(roc_auc_score(labels, probs)),
        "Accuracy": float(accuracy_score(labels, predicted)),
        "Sensitivity": float(recall_score(labels, predicted, zero_division=0)),
        "Specificity": float(tn / (tn + fp)),
        "F1": float(f1_score(labels, predicted, zero_division=0)),
        "TN": int(tn), "FP": int(fp), "FN": int(fn), "TP": int(tp),
    }


def conditions():
    yield "clean", 0, "original", None
    for kind, values in LEVELS.items():
        for level, value in enumerate(values, start=1):
            yield kind, level, value, value if kind == "noise" else None


def evaluate_models(
    data_dir: Path, checkpoints: dict[str, Path], thresholds: dict[str, float],
    output: Path, batch_size: int, device: torch.device, seed: int,
    noise_seeds: tuple[int, ...] | None = None,
) -> None:
    if batch_size < 1:
        raise ValueError("batch_size must be positive")
    if noise_seeds is None:
        noise_seeds = (seed, seed + 81, seed + 957)
    models_loaded = {name: load_model(path, device) for name, path in checkpoints.items()}
    records = []
    repeats = []
    for kind, level, value, noise_std in conditions():
        transform = raw_transform() if kind in ("clean", "noise") else degraded_transform(kind, level)
        dataset = image_dataset(data_dir / "test", transform)
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
        seeds = noise_seeds if kind == "noise" and level > 0 else (seed,)
        for run_seed in seeds:
            current = []
            for name, model in models_loaded.items():
                labels, probs = predict(model, loader, device, noise_std or 0.0, run_seed)
                row = {"Model": name, "Degradation": kind, "Level": level,
                       "Parameter": value, "Seed": run_seed,
                       "Threshold": thresholds[name], **metrics(labels, probs, thresholds[name])}
                current.append(row)
                if run_seed == seeds[0]:
                    records.append(row)
                if kind == "noise":
                    repeats.append(row)
            if run_seed == seeds[0]:
                print(f"{kind} L{level}: " + " | ".join(
                    f"{row['Model']} AUROC={row['AUROC']:.4f}" for row in current))
    output.mkdir(parents=True, exist_ok=True)
    save_csv(output / "metrics.csv", records)
    if repeats:
        save_csv(output / "noise_repeats.csv", repeats)
    if len(checkpoints) == 2:
        names = list(checkpoints)
        lookup = {(r["Model"], r["Degradation"], r["Level"]): r for r in records}
        summary = []
        for kind, level, value, _ in conditions():
            base, robust = (lookup[(name, kind, level)] for name in names)
            summary.append({"Degradation": kind, "Level": level, "Parameter": value,
                            f"{names[0]}_AUROC": base["AUROC"],
                            f"{names[1]}_AUROC": robust["AUROC"],
                            "Delta_AUROC": robust["AUROC"] - base["AUROC"]})
        save_csv(output / "comparison.csv", summary)
    print(f"Saved results to {output}")


def save_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
