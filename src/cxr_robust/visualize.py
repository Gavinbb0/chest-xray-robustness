"""Plots derived from measured CSV results and optional Grad-CAM comparison."""

from __future__ import annotations

import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image
from torchvision import transforms

from .data import MEAN, STD
from .experiment import load_model


def plot_results(metrics_csv: Path, output: Path) -> None:
    with metrics_csv.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError("No metrics rows found")
    names = list(dict.fromkeys(row["Model"] for row in rows))
    kinds = list(dict.fromkeys(row["Degradation"] for row in rows if row["Degradation"] != "clean"))
    fig, axes = plt.subplots(2, 3, figsize=(13, 8), sharex=True, sharey=True)
    for ax, kind in zip(axes.flat, kinds):
        for name in names:
            clean = next(row for row in rows if row["Model"] == name and row["Degradation"] == "clean")
            degraded = sorted((row for row in rows
                               if row["Model"] == name and row["Degradation"] == kind),
                              key=lambda row: int(row["Level"]))
            ax.plot(range(5), [float(clean["AUROC"])] + [float(row["AUROC"]) for row in degraded],
                    marker="o", label=name)
        ax.set_title(kind.capitalize())
        ax.set_xticks(range(5))
        ax.set_xlabel("Severity level")
        ax.grid(alpha=0.25)
    for ax in axes.flat[len(kinds):]:
        ax.set_visible(False)
    axes[0, 0].set_ylabel("AUROC")
    axes[1, 0].set_ylabel("AUROC")
    axes[0, 0].legend()
    fig.suptitle("Chest X-ray robustness on the test set")
    fig.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=220)
    plt.close(fig)
    print(f"Saved figure: {output}")


def _cam(model: torch.nn.Module, image: torch.Tensor) -> np.ndarray:
    captured = {}
    def capture(_module, _inputs, output):
        captured["activation"] = output
        output.register_hook(lambda grad: captured.update(gradient=grad))
    hook = model.layer4[-1].register_forward_hook(capture)
    try:
        model.zero_grad(set_to_none=True)
        output = model(image)
        output[0, 1].backward()
        activation = captured["activation"].detach()
        gradient = captured["gradient"].detach()
        weights = gradient.mean(dim=(2, 3), keepdim=True)
        cam = torch.relu((activation * weights).sum(dim=1, keepdim=True))
        cam = torch.nn.functional.interpolate(
            cam, size=image.shape[-2:], mode="bilinear", align_corners=False)
        cam = cam[0, 0]
        cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
        return cam.cpu().numpy()
    finally:
        hook.remove()


def compare_gradcam(image_path: Path, baseline: Path, robust: Path,
                    output: Path, device: torch.device) -> None:
    with Image.open(image_path) as original:
        image = original.convert("RGB").resize((224, 224))
    tensor = transforms.Compose([
        transforms.ToTensor(), transforms.Normalize(MEAN, STD),
    ])(image).unsqueeze(0).to(device)
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    axes[0].imshow(image)
    axes[0].set_title("X-ray")
    for ax, name, path in zip(axes[1:], ("Baseline", "Robust"), (baseline, robust)):
        model = load_model(path, device)
        ax.imshow(image)
        ax.imshow(_cam(model, tensor), cmap="jet", alpha=0.45, vmin=0, vmax=1)
        ax.set_title(f"{name} | pneumonia Grad-CAM")
    for ax in axes:
        ax.axis("off")
    fig.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=220)
    plt.close(fig)
    print(f"Saved figure: {output}")
