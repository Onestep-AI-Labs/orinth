import argparse
from pathlib import Path
from typing import Any

from app.ml.vision.yolo.catalog import YOLO_ADVANCED_PARAMETERS
from app.training.runners.advanced import allowed_keys, log_ignored, parse_advanced, partition

# Every advanced key for YOLO is a direct `model.train(**kwargs)` argument, so
# the accepted subset merges straight onto the base train args.
YOLO_ADVANCED_KEYS = allowed_keys(YOLO_ADVANCED_PARAMETERS)


def build_advanced_train_args(raw: str | None) -> tuple[dict[str, Any], list[str]]:
    """Return ``(train-arg overrides, ignored keys)`` from the advanced blob."""

    return partition(parse_advanced(raw), YOLO_ADVANCED_KEYS)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--dataset-root", required=True)
    parser.add_argument("--initial-weights", required=True)
    parser.add_argument("--class-names", default="object")
    parser.add_argument("--epochs", type=int, required=True)
    parser.add_argument("--image-size", type=int, required=True)
    parser.add_argument("--batch-size", type=int, required=True)
    parser.add_argument("--cache", choices=["disk", "ram", "none"], default="disk")
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--patience", type=int, default=50)
    parser.add_argument("--optimizer", default="AdamW")
    parser.add_argument("--learning-rate", type=float, default=0.002)
    parser.add_argument("--device", default="")
    parser.add_argument("--advanced", default="{}")
    args = parser.parse_args()

    advanced, ignored = build_advanced_train_args(args.advanced)
    log_ignored(ignored)

    from ultralytics import YOLO

    run_dir = Path(args.run_dir)
    dataset_root = Path(args.dataset_root)
    class_names = [name.strip() for name in args.class_names.split(",") if name.strip()]
    if not class_names:
        class_names = ["object"]
    names = "[" + ", ".join(repr(name) for name in class_names) + "]"
    local_yaml = run_dir / "data.local.yaml"
    local_yaml.write_text(
        "\n".join(
            [
                f"path: {dataset_root}",
                "train: train/images",
                "val: valid/images",
                "test: test/images",
                f"nc: {len(class_names)}",
                f"names: {names}",
                "",
            ]
        ),
        encoding="utf-8",
    )

    model = YOLO(args.initial_weights)
    cache_value: bool | str = False if args.cache == "none" else args.cache
    train_args = {
        "data": str(local_yaml),
        "epochs": args.epochs,
        "imgsz": args.image_size,
        "batch": args.batch_size,
        "optimizer": args.optimizer,
        "lr0": args.learning_rate,
        "lrf": 0.15,
        "momentum": 0.9,
        "weight_decay": 0.0005,
        "warmup_epochs": 5,
        "patience": args.patience,
        "close_mosaic": 25,
        "hsv_h": 0.005,
        "hsv_s": 0.5,
        "hsv_v": 0.3,
        "degrees": 5.0,
        "translate": 0.05,
        "scale": 0.2,
        "shear": 0.0,
        "perspective": 0.0,
        "flipud": 0.0,
        "fliplr": 0.5,
        "mosaic": 0.5,
        "mixup": 0.1,
        "cutmix": 0.0,
        "cache": cache_value,
        "workers": args.workers,
        "amp": True,
        "project": str(run_dir.parent),
        "name": run_dir.name,
        "exist_ok": True,
    }
    # Advanced overrides win over the base recipe above; every accepted key is a
    # valid Ultralytics train argument.
    train_args.update(advanced)
    if args.device:
        train_args["device"] = args.device

    results = model.train(**train_args)
    print(f"Training finished. Results saved to: {results.save_dir}")


if __name__ == "__main__":
    main()
