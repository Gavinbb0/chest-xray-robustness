"""Ten-minute LecTrace walkthrough of the chest X-ray robustness project.

This traces the evaluation *logic* using saved results. It does not rerun ResNet18.
Individual probabilities and thresholds come from final_project.pdf, Table 6 and
Section 5.1; the AUROC comparison is read from the saved final_summary.csv.
"""

from csv import DictReader
from pathlib import Path

from lectrace import image, note, text


ROOT = Path(__file__).resolve().parent
FIGURES = ROOT / "results" / "figures"
METRICS = ROOT / "results" / "metrics" / "final_summary.csv"


def show_figure(path: Path, description: str, width: int) -> None:
    """Use a local figure when available; keep the public script runnable without it."""
    if path.is_file():
        image(str(path), width=width)
    else:
        text(description)


def load_auroc(condition: str) -> tuple[float, float]:
    with METRICS.open(newline="", encoding="utf-8") as handle:
        for row in DictReader(handle):
            if row["Condition"] == condition:
                return float(row["Standard_AUROC"]), float(row["Robust_AUROC"])
    raise ValueError(f"Missing condition: {condition}")


def main() -> None:
    text("# When a chest X-ray gets noisy, does the model still work?")
    text(
        "**Healthcare problem (0:00–1:15).** A pneumonia classifier may see "
        "blurred, noisy, dark, low-contrast, or compressed images. A wrong alarm "
        "can send a normal case for unnecessary follow-up."
    )
    text(
        "**Motivation and approach (1:15–2:30).** Compare a standard ResNet18 "
        "with the same model trained on clean and synthetically degraded images. "
        "Keep the official test set untouched and compare both models on the "
        "same degraded test images."
    )
    note("Say that this is a research robustness study, not a clinical device.")

    text("## State 1 — The input and its label (2:30–3:15)")
    case_id = "IM-0001-0001"  # @inspect case_id
    true_label = "Normal"  # @inspect true_label
    text(f"Selected test case: `{case_id}`. Ground truth: **{true_label}**.")
    show_figure(
        FIGURES / "error_cases.png",
        "Optional case figure: a normal chest X-ray under Level 4 noise; "
        "standard model 0.998 versus robust model 0.194 pneumonia probability.",
        width=600,
    )  # @stepover
    note(
        "If the optional case figure is present, use only its top row while "
        "speaking. The label remains Normal in every later state."
    )

    text("## State 2 — Apply the test degradation (3:15–4:15)")
    severity = 0  # @inspect severity
    severity = 4  # @inspect severity
    noise_std = 0.15  # @inspect noise_std
    text(
        "The evaluation changes the image to **Gaussian-noise Level 4** "
        f"(standard deviation {noise_std}). The diagnosis does not change."
    )
    show_figure(
        FIGURES / "noise_preview_v2.png",
        "Optional noise preview: Level 4 adds Gaussian noise with standard "
        "deviation 0.15 on the [0, 1] image scale.",
        width=700,
    )  # @stepover
    note(
        "Explain that noise is added on the [0, 1] image scale before "
        "normalization. The optional preview shows severity, not a new disease label."
    )

    text("## State 3 — Standard model decision (4:15–5:30)")
    text(
        "In the real evaluation code: resize → tensor → add noise → "
        "ImageNet normalization → ResNet18 logits → softmax pneumonia "
        "probability → threshold decision."
    )
    standard_probability = 0.998  # @inspect standard_probability
    standard_threshold = 0.8009  # @inspect standard_threshold
    standard_prediction = (
        "Pneumonia" if standard_probability >= standard_threshold else "Normal"
    )  # @inspect standard_prediction
    text(
        f"Reported pneumonia probability: **{standard_probability:.3f}**; "
        f"validation threshold: **{standard_threshold:.4f}**. "
        f"Decision: **{standard_prediction}** — a false positive."
    )
    note(
        "The probability is a saved output from the report, not a live neural "
        "forward pass in this LecTrace script. The comparison with the threshold "
        "is executed here."
    )

    text("## State 4 — Degradation-aware decision (5:30–6:45)")
    robust_probability = 0.194  # @inspect robust_probability
    robust_threshold = 0.5728  # @inspect robust_threshold
    robust_prediction = (
        "Pneumonia" if robust_probability >= robust_threshold else "Normal"
    )  # @inspect robust_prediction
    text(
        f"Reported pneumonia probability: **{robust_probability:.3f}**; "
        f"validation threshold: **{robust_threshold:.4f}**. "
        f"Decision: **{robust_prediction}** — correct for this case."
    )
    note(
        "Only the training strategy changed. Both models saw the same test "
        "image; each uses its own validation-derived threshold."
    )

    text("## State 5 — One case versus the whole test set (6:45–8:15)")
    standard_clean, robust_clean = load_auroc("Clean")  # @stepover
    standard_noisy, robust_noisy = load_auroc("Noise L4")  # @stepover
    standard_drop = standard_clean - standard_noisy  # @inspect standard_drop
    robust_drop = robust_clean - robust_noisy  # @inspect robust_drop
    text(
        "From the saved single-realization summary: "
        f"standard AUROC **{standard_clean:.4f} → {standard_noisy:.4f}**, "
        f"robust AUROC **{robust_clean:.4f} → {robust_noisy:.4f}**. "
        "AUROC measures ranking over the full test set, not whether one case "
        "passes a threshold."
    )
    show_figure(
        FIGURES / "severe_standard_vs_robust.png",
        "Optional comparison figure: under Level 4 noise, standard AUROC is "
        "0.6773 and robust AUROC is 0.9559 (three-seed mean).",
        width=700,
    )  # @stepover
    note(
        "The report's three-noise-seed mean is 0.6773 ± 0.0133 versus "
        "0.9559 ± 0.0023. The saved summary CSV above is one realization, "
        "so its robust value differs slightly."
    )

    text("## Healthcare interpretation (8:15–10:00)")
    text(
        "Degradation-aware training substantially reduced the loss of "
        "discrimination under the tested synthetic noise. In the selected "
        "normal case, it also avoided a false pneumonia alarm. However, fixed-"
        "threshold specificity remained limited, patient-level separation was "
        "not explicitly checked, and external clinical data were not tested."
    )
    note(
        "Close with one sentence: clean-image accuracy alone does not show "
        "whether a model remains reliable as image quality worsens."
    )


if __name__ == "__main__":
    main()
