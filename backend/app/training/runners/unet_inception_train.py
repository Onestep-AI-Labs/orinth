import argparse
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--dataset-root", required=True)
    parser.add_argument("--epochs", type=int, required=True)
    args = parser.parse_args()

    run_dir = Path(args.run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    notes = run_dir / "README.txt"
    notes.write_text(
        "\n".join(
            [
                "U-Net + Inception training runner placeholder.",
                "",
                "Inference and testing are implemented for this model family.",
                "The notebook reference contains the full two-stage training flow.",
                "Promote this runner after validating the long TensorFlow training path",
                "with local COCO data and hardware-specific batch sizes.",
                f"Dataset root: {args.dataset_root}",
                f"Requested epochs: {args.epochs}",
            ]
        ),
        encoding="utf-8",
    )
    raise SystemExit(
        "U-Net + Inception training runner is intentionally gated until YOLO training is stable."
    )


if __name__ == "__main__":
    main()
