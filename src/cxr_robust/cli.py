"""Command line interface for the two-stage experiment."""

from __future__ import annotations

import argparse
from pathlib import Path

from .data import prepare_split
from .experiment import (
    device_for, evaluate_models, get_threshold, train_model, validation_threshold,
)


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Chest X-ray ResNet18 robustness project")
    commands = p.add_subparsers(dest="command", required=True)
    prepare = commands.add_parser("prepare", help="Create train/val/test folders by copying images")
    prepare.add_argument("--source", type=Path, required=True,
                         help="Original root containing train/ and test/ class folders")
    prepare.add_argument("--data-dir", type=Path, required=True)
    prepare.add_argument("--val-fraction", type=float, default=0.2)
    prepare.add_argument("--seed", type=int, default=42)

    train = commands.add_parser("train", help="Train baseline or robust ResNet18")
    train.add_argument("--data-dir", type=Path, required=True)
    train.add_argument("--variant", choices=("baseline", "robust"), required=True)
    train.add_argument("--checkpoint", type=Path, required=True)
    train.add_argument("--epochs", type=int, default=10)
    train.add_argument("--batch-size", type=int, default=32)
    train.add_argument("--lr", type=float, default=1e-4)
    train.add_argument("--seed", type=int, default=42)
    train.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    train.add_argument("--no-pretrained", action="store_true",
                       help="Use random initial weights; default downloads ImageNet ResNet18")

    calibrate = commands.add_parser("calibrate", help="Select threshold on validation data only")
    calibrate.add_argument("--data-dir", type=Path, required=True)
    calibrate.add_argument("--checkpoint", type=Path, required=True)
    calibrate.add_argument("--batch-size", type=int, default=32)
    calibrate.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")

    evaluate = commands.add_parser("evaluate", help="Test one model on clean and degraded images")
    evaluate.add_argument("--data-dir", type=Path, required=True)
    evaluate.add_argument("--checkpoint", type=Path, required=True)
    evaluate.add_argument("--name", default="model")
    evaluate.add_argument("--threshold", type=float, default=None)
    evaluate.add_argument("--output", type=Path, required=True)
    evaluate.add_argument("--batch-size", type=int, default=32)
    evaluate.add_argument("--seed", type=int, default=42)
    evaluate.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")

    compare = commands.add_parser("compare", help="Evaluate both models on matching test images")
    compare.add_argument("--data-dir", type=Path, required=True)
    compare.add_argument("--baseline", type=Path, required=True)
    compare.add_argument("--robust", type=Path, required=True)
    compare.add_argument("--baseline-threshold", type=float, default=None)
    compare.add_argument("--robust-threshold", type=float, default=None)
    compare.add_argument("--output", type=Path, required=True)
    compare.add_argument("--batch-size", type=int, default=32)
    compare.add_argument("--seed", type=int, default=42)
    compare.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")

    plot = commands.add_parser("plot", help="Plot results from metrics.csv")
    plot.add_argument("--metrics", type=Path, required=True)
    plot.add_argument("--output", type=Path, required=True)

    gradcam = commands.add_parser("gradcam", help="Compare pneumonia Grad-CAM maps")
    gradcam.add_argument("--image", type=Path, required=True)
    gradcam.add_argument("--baseline", type=Path, required=True)
    gradcam.add_argument("--robust", type=Path, required=True)
    gradcam.add_argument("--output", type=Path, required=True)
    gradcam.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    return p


def main() -> None:
    args = parser().parse_args()
    if args.command == "prepare":
        print(prepare_split(args.source, args.data_dir, args.val_fraction, args.seed))
    elif args.command == "train":
        train_model(args.data_dir, args.checkpoint, args.variant == "robust",
                    args.epochs, args.batch_size, args.lr, args.seed,
                    device_for(args.device), not args.no_pretrained)
    elif args.command == "calibrate":
        validation_threshold(args.data_dir, args.checkpoint, args.batch_size,
                             device_for(args.device))
    elif args.command == "evaluate":
        evaluate_models(args.data_dir, {args.name: args.checkpoint},
                        {args.name: get_threshold(args.checkpoint, args.threshold)},
                        args.output, args.batch_size, device_for(args.device), args.seed)
    elif args.command == "compare":
        evaluate_models(args.data_dir, {"baseline": args.baseline, "robust": args.robust},
                        {"baseline": get_threshold(args.baseline, args.baseline_threshold),
                         "robust": get_threshold(args.robust, args.robust_threshold)},
                        args.output, args.batch_size, device_for(args.device), args.seed)
    elif args.command == "plot":
        from .visualize import plot_results
        plot_results(args.metrics, args.output)
    elif args.command == "gradcam":
        from .visualize import compare_gradcam
        compare_gradcam(args.image, args.baseline, args.robust,
                        args.output, device_for(args.device))


if __name__ == "__main__":
    main()
