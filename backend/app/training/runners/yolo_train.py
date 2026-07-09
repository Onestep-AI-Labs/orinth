import argparse
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--dataset-root", required=True)
    parser.add_argument("--initial-weights", required=True)
    parser.add_argument("--epochs", type=int, required=True)
    parser.add_argument("--image-size", type=int, required=True)
    parser.add_argument("--batch-size", type=int, required=True)
    args = parser.parse_args()

    from ultralytics import YOLO

    run_dir = Path(args.run_dir)
    dataset_root = Path(args.dataset_root)
    local_yaml = run_dir / "data.local.yaml"
    local_yaml.write_text(
        "\n".join(
            [
                f"path: {dataset_root}",
                "train: train/images",
                "val: valid/images",
                "test: test/images",
                "nc: 2",
                "names: ['granuloma', 'kista']",
                "",
            ]
        ),
        encoding="utf-8",
    )

    model = YOLO(args.initial_weights)
    results = model.train(
        data=str(local_yaml),
        epochs=args.epochs,
        imgsz=args.image_size,
        batch=args.batch_size,
        optimizer="AdamW",
        lr0=0.002,
        lrf=0.15,
        momentum=0.9,
        weight_decay=0.0005,
        warmup_epochs=5,
        patience=50,
        close_mosaic=25,
        hsv_h=0.005,
        hsv_s=0.5,
        hsv_v=0.3,
        degrees=5.0,
        translate=0.05,
        scale=0.2,
        shear=0.0,
        perspective=0.0,
        flipud=0.0,
        fliplr=0.5,
        mosaic=0.5,
        mixup=0.1,
        cutmix=0.0,
        cache=True,
        amp=True,
        project=str(run_dir.parent),
        name=run_dir.name,
        exist_ok=True,
    )
    print(f"Training finished. Results saved to: {results.save_dir}")


if __name__ == "__main__":
    main()
