#!/usr/bin/env python
"""Train a YOLO model using the ultralytics package.

Defaults to `yolov10n.pt` and uses GPU if available. Adjust command-line
options for `--data`, `--epochs`, `--batch`, `--imgsz`, and `--model`.
"""
import argparse
import os
import math
import sys

def parse_args():
    p = argparse.ArgumentParser(description="Train YOLO with ultralytics")
    p.add_argument("--data", default="data.yaml", help="Path to data.yaml")
    p.add_argument("--model", default="yolov10n.pt", help="Pretrained weights / model name")
    p.add_argument("--epochs", type=int, default=50, help="Number of epochs")
    p.add_argument("--imgsz", type=int, default=640, help="Image size (square)")
    p.add_argument("--batch", type=int, default=None, help="Batch size (auto if not set)")
    p.add_argument("--device", default=None, help="Device to use, e.g. 0 or 'cpu' (auto if not set)")
    p.add_argument("--name", default="yolov10_train", help="Run name (output folder)")
    return p.parse_args()

def recommend_batch_from_vram(vram_gb: float) -> int:
    # Very simple heuristic for batch size recommendations based on VRAM
    if vram_gb < 4:
        return 2
    if vram_gb < 6:
        return 4
    if vram_gb < 8:
        return 8
    if vram_gb < 12:
        return 16
    return 32

def main():
    args = parse_args()

    # Import here so users without packages get friendly message
    try:
        import torch
    except Exception as e:
        print("ERROR: torch not found or failed to import. Activate venv and install requirements.")
        raise

    try:
        from ultralytics import YOLO
    except Exception as e:
        print("ERROR: ultralytics not found or failed to import. Activate venv and install requirements.")
        raise

    # Determine device
    device = args.device
    if device is None:
        device = 0 if torch.cuda.is_available() else 'cpu'

    print(f"Using device: {device}")

    # VRAM info (if GPU)
    vram_gb = None
    if isinstance(device, int) or (isinstance(device, str) and device.isdigit()):
        if torch.cuda.is_available():
            try:
                dev_idx = int(device)
                prop = torch.cuda.get_device_properties(dev_idx)
                vram_gb = round(prop.total_memory / 1024**3, 2)
                print(f"Detected GPU: {prop.name} with {vram_gb} GB VRAM")
            except Exception:
                # fallback if any failure
                vram_gb = None

    # Auto batch sizing
    if args.batch is None:
        if vram_gb is not None:
            batch = recommend_batch_from_vram(vram_gb)
            print(f"Auto-selected batch size {batch} based on {vram_gb} GB VRAM")
        else:
            batch = 4
            print(f"No GPU VRAM detected; defaulting batch size to {batch}")
    else:
        batch = args.batch

    print(f"Training config: model={args.model}, data={args.data}, epochs={args.epochs}, imgsz={args.imgsz}, batch={batch}")

    # Create output directory
    project = os.path.join("runs", "train")
    os.makedirs(project, exist_ok=True)

    # Start training
    try:
        y = YOLO(args.model)
        y.train(
            data=args.data,
            epochs=args.epochs,
            imgsz=args.imgsz,
            batch=batch,
            device=device,
            project=project,
            name=args.name,
        )
    except Exception as e:
        print("Training failed:", e)
        raise

if __name__ == '__main__':
    main()
