#!/usr/bin/env python3
"""
YOLOv8 Training Script with GPU Verification and Monitoring

This script trains a YOLOv8 model with:
- Automatic GPU detection and usage
- Training progress monitoring
- Validation during training
- Best model (best.pt) and last checkpoint (last.pt) generation
- Memory optimization for 6GB VRAM
- Comprehensive logging

Hardware Requirements:
    - GPU: NVIDIA RTX A3000 (6GB VRAM)
    - CUDA: 11.7+
    - PyTorch: 2.0.1+

Usage Examples:
    # Basic training
    python train_yolo.py

    # Custom configuration
    python train_yolo.py --data "path/to/data.yaml" --epochs 100 --batch 8

    # Resume training
    python train_yolo.py --resume --weights "runs/detect/train/weights/last.pt"

    # Fine-tune pretrained model
    python train_yolo.py --weights yolov8n.pt --epochs 50

Output Structure:
    runs/detect/train/
    ├── weights/
    │   ├── best.pt      (Best model based on validation mAP)
    │   └── last.pt      (Last checkpoint)
    ├── results.csv      (Training metrics)
    ├── results.png      (Training curves)
    ├── confusion_matrix.png
    ├── F1_curve.png
    ├── PR_curve.png
    └── args.yaml        (Training arguments)
"""

import argparse
import os
import sys
import torch
from pathlib import Path
from datetime import datetime
import warnings

warnings.filterwarnings('ignore')

# Import after checking environment
try:
    from ultralytics import YOLO
    from ultralytics.utils import LOGGER
    from ultralytics.utils.checks import check_requirements
except ImportError as e:
    print(f"❌ Error: Ultralytics not installed properly")
    print(f"   Run: pip install ultralytics")
    sys.exit(1)

# Workaround for PyTorch 2.6+ safe globals when loading pickled ultralytics checkpoints.
# This allowlists `ultralytics.nn.tasks.DetectionModel` so torch.load can unpickle model files
# that reference this class. It's safe to do this if you trust the checkpoint source (official weights).
try:
    import ultralytics.nn.tasks as _ul_tasks
    if hasattr(torch.serialization, 'add_safe_globals'):
        try:
            torch.serialization.add_safe_globals([_ul_tasks.DetectionModel])
        except Exception:
            # If the process cannot register (older/newer torch), ignore and continue
            pass
    # Some torch versions provide a context manager 'safe_globals' instead
    elif hasattr(torch.serialization, 'safe_globals'):
        try:
            # Ensure DetectionModel is present in safe globals by using the context manager once (no-op here)
            with torch.serialization.safe_globals([_ul_tasks.DetectionModel]):
                pass
        except Exception:
            pass
except Exception:
    # If anything fails here, continue — the downstream load will raise the original informative error.
    pass


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Train YOLOv8 model with GPU support",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    
    # Model and data
    parser.add_argument(
        "--weights",
        type=str,
        default="yolov8n.pt",
        help="Path to initial weights (yolov8n.pt, yolov8s.pt, yolov8m.pt, etc.)"
    )
    
    parser.add_argument(
        "--data",
        type=str,
        default=r"D:\New folder\yolo\data.yaml",
        help="Path to data.yaml configuration file"
    )
    
    # Training parameters
    parser.add_argument(
        "--epochs",
        type=int,
        default=100,
        help="Number of training epochs (default: 100)"
    )
    
    parser.add_argument(
        "--batch",
        type=int,
        default=16,
        help="Batch size (default: 16, reduce if OOM error)"
    )
    
    parser.add_argument(
        "--imgsz",
        type=int,
        default=640,
        help="Input image size (default: 640)"
    )
    
    parser.add_argument(
        "--device",
        type=str,
        default="0",
        help="Device to use: '0' for GPU, 'cpu' for CPU (default: 0)"
    )
    
    # Optimization
    parser.add_argument(
        "--optimizer",
        type=str,
        default="auto",
        choices=["SGD", "Adam", "AdamW", "auto"],
        help="Optimizer type (default: auto)"
    )
    
    parser.add_argument(
        "--lr0",
        type=float,
        default=0.01,
        help="Initial learning rate (default: 0.01)"
    )
    
    parser.add_argument(
        "--lrf",
        type=float,
        default=0.01,
        help="Final learning rate factor (default: 0.01)"
    )
    
    parser.add_argument(
        "--momentum",
        type=float,
        default=0.937,
        help="SGD momentum/Adam beta1 (default: 0.937)"
    )
    
    parser.add_argument(
        "--weight-decay",
        type=float,
        default=0.0005,
        help="Weight decay (default: 0.0005)"
    )
    
    # Data augmentation
    parser.add_argument(
        "--hsv-h",
        type=float,
        default=0.015,
        help="HSV-Hue augmentation (default: 0.015)"
    )
    
    parser.add_argument(
        "--hsv-s",
        type=float,
        default=0.7,
        help="HSV-Saturation augmentation (default: 0.7)"
    )
    
    parser.add_argument(
        "--hsv-v",
        type=float,
        default=0.4,
        help="HSV-Value augmentation (default: 0.4)"
    )
    
    parser.add_argument(
        "--degrees",
        type=float,
        default=0.0,
        help="Rotation augmentation degrees (default: 0.0)"
    )
    
    parser.add_argument(
        "--translate",
        type=float,
        default=0.1,
        help="Translation augmentation (default: 0.1)"
    )
    
    parser.add_argument(
        "--scale",
        type=float,
        default=0.5,
        help="Scaling augmentation (default: 0.5)"
    )
    
    parser.add_argument(
        "--mosaic",
        type=float,
        default=1.0,
        help="Mosaic augmentation probability (default: 1.0)"
    )
    
    # Validation
    parser.add_argument(
        "--val",
        action="store_true",
        default=True,
        help="Validate during training (default: True)"
    )
    
    parser.add_argument(
        "--save-period",
        type=int,
        default=-1,
        help="Save checkpoint every x epochs (-1 = disabled)"
    )
    
    # Resume training
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume training from last checkpoint"
    )
    
    parser.add_argument(
        "--project",
        type=str,
        default="runs/detect",
        help="Project directory (default: runs/detect)"
    )
    
    parser.add_argument(
        "--name",
        type=str,
        default="train",
        help="Experiment name (default: train)"
    )
    
    parser.add_argument(
        "--exist-ok",
        action="store_true",
        help="Allow overwriting existing project/name"
    )
    
    # Advanced
    parser.add_argument(
        "--workers",
        type=int,
        default=8,
        help="Number of dataloader workers (default: 8)"
    )
    
    parser.add_argument(
        "--cache",
        action="store_true",
        help="Cache images for faster training"
    )
    
    parser.add_argument(
        "--amp",
        action="store_true",
        default=True,
        help="Use Automatic Mixed Precision (default: True)"
    )
    
    parser.add_argument(
        "--patience",
        type=int,
        default=50,
        help="Epochs to wait for no improvement for early stopping (default: 50)"
    )
    
    parser.add_argument(
        "--close-mosaic",
        type=int,
        default=10,
        help="Disable mosaic augmentation in final N epochs (default: 10)"
    )
    
    return parser.parse_args()


def verify_gpu():
    """Verify GPU availability and print information."""
    print("\n" + "=" * 70)
    print("🔍 GPU VERIFICATION")
    print("=" * 70)
    
    # Check CUDA availability
    cuda_available = torch.cuda.is_available()
    print(f"CUDA Available: {cuda_available}")
    
    if cuda_available:
        print(f"CUDA Version: {torch.version.cuda}")
        print(f"PyTorch Version: {torch.__version__}")
        print(f"Number of GPUs: {torch.cuda.device_count()}")
        
        for i in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(i)
            print(f"\nGPU {i}: {props.name}")
            print(f"  Total Memory: {props.total_memory / 1024**3:.2f} GB")
            print(f"  Compute Capability: {props.major}.{props.minor}")
            
            # Check current memory
            allocated = torch.cuda.memory_allocated(i) / 1024**3
            reserved = torch.cuda.memory_reserved(i) / 1024**3
            print(f"  Memory Allocated: {allocated:.2f} GB")
            print(f"  Memory Reserved: {reserved:.2f} GB")
        
        return True
    else:
        print("⚠️  WARNING: CUDA not available!")
        print("   Training will use CPU (much slower)")
        print("\nPossible issues:")
        print("  1. NVIDIA drivers not installed")
        print("  2. PyTorch installed without CUDA support")
        print("  3. GPU not detected by system")
        return False


def verify_data_yaml(data_path):
    """Verify data.yaml exists and is properly formatted."""
    print("\n" + "=" * 70)
    print("📄 DATA CONFIGURATION VERIFICATION")
    print("=" * 70)
    
    data_path = Path(data_path)
    
    if not data_path.exists():
        print(f"❌ Error: data.yaml not found at {data_path}")
        return False
    
    print(f"✅ Found data.yaml at: {data_path}")
    
    # Try to read and validate
    import yaml
    try:
        with open(data_path, 'r') as f:
            data = yaml.safe_load(f)
        
        print("\nConfiguration:")
        print(f"  Number of classes: {data.get('nc', 'NOT FOUND')}")
        print(f"  Class names: {data.get('names', 'NOT FOUND')}")
        print(f"  Train path: {data.get('train', 'NOT FOUND')}")
        print(f"  Val path: {data.get('val', 'NOT FOUND')}")
        
        # Verify paths exist
        base_path = data_path.parent if 'path' not in data else Path(data['path'])
        
        if 'train' in data:
            train_path = base_path / data['train']
            if train_path.exists():
                print(f"  ✅ Train images found")
            else:
                print(f"  ⚠️  Train path not found: {train_path}")
        
        if 'val' in data:
            val_path = base_path / data['val']
            if val_path.exists():
                print(f"  ✅ Validation images found")
            else:
                print(f"  ⚠️  Validation path not found: {val_path}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error reading data.yaml: {e}")
        return False


def optimize_batch_size_for_vram(initial_batch, vram_gb=6):
    """Suggest optimal batch size for available VRAM."""
    print("\n" + "=" * 70)
    print("⚙️  BATCH SIZE OPTIMIZATION")
    print("=" * 70)
    
    print(f"GPU VRAM: {vram_gb} GB")
    print(f"Requested batch size: {initial_batch}")
    
    # Rough estimation for YOLOv8n with 640x640 images
    # ~0.5GB per batch of 8 images
    safe_batch_sizes = {
        6: 16,   # RTX A3000
        8: 24,
        12: 32,
        16: 48,
        24: 64
    }
    
    recommended = safe_batch_sizes.get(vram_gb, 16)
    
    if initial_batch > recommended:
        print(f"⚠️  Batch size {initial_batch} may cause OOM errors")
        print(f"💡 Recommended batch size: {recommended}")
        print(f"   Use --batch {recommended} to avoid memory issues")
        return recommended
    else:
        print(f"✅ Batch size {initial_batch} should work fine")
        return initial_batch


def train_model(args):
    """Train YOLOv8 model with specified arguments."""
    
    # Verify GPU
    gpu_available = verify_gpu()
    
    # Verify data configuration
    if not verify_data_yaml(args.data):
        print("\n❌ Please fix data.yaml configuration before training")
        return
    
    # Optimize batch size
    if gpu_available:
        vram = torch.cuda.get_device_properties(0).total_memory / 1024**3
        suggested_batch = optimize_batch_size_for_vram(args.batch, vram)
        if suggested_batch != args.batch:
            response = input(f"\nUse recommended batch size {suggested_batch}? (y/n): ")
            if response.lower() == 'y':
                args.batch = suggested_batch
    
    print("\n" + "=" * 70)
    print("🚀 STARTING TRAINING")
    print("=" * 70)
    print(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    try:
        # Load model
        print(f"\n📦 Loading model: {args.weights}")
        # Some PyTorch versions restrict globals during torch.load (weights_only behavior).
        # To load trusted official weights (ultralytics), temporarily call torch.load with
        # weights_only=False to allow full unpickling. We restore torch.load afterwards.
        _orig_torch_load = getattr(torch, 'load', None)
        try:
            def _torch_load_force_weights_false(f, *a, **kw):
                # ensure weights_only=False when present in this torch version
                if 'weights_only' not in kw:
                    kw['weights_only'] = False
                return _orig_torch_load(f, *a, **kw)

            if _orig_torch_load is not None:
                torch.load = _torch_load_force_weights_false
            model = YOLO(args.weights)
        finally:
            if _orig_torch_load is not None:
                torch.load = _orig_torch_load
        
        # Print model info
        print(f"✅ Model loaded successfully")
        print(f"   Parameters: {sum(p.numel() for p in model.model.parameters()):,}")
        
        # Training arguments
        train_args = {
            'data': args.data,
            'epochs': args.epochs,
            'batch': args.batch,
            'imgsz': args.imgsz,
            'device': args.device,
            'optimizer': args.optimizer,
            'lr0': args.lr0,
            'lrf': args.lrf,
            'momentum': args.momentum,
            'weight_decay': args.weight_decay,
            'hsv_h': args.hsv_h,
            'hsv_s': args.hsv_s,
            'hsv_v': args.hsv_v,
            'degrees': args.degrees,
            'translate': args.translate,
            'scale': args.scale,
            'mosaic': args.mosaic,
            'val': args.val,
            'save_period': args.save_period,
            'project': args.project,
            'name': args.name,
            'exist_ok': args.exist_ok,
            'workers': args.workers,
            'cache': args.cache,
            'amp': args.amp,
            'patience': args.patience,
            'close_mosaic': args.close_mosaic,
            'verbose': True,
            'plots': True,
        }
        
        # Add resume if specified
        if args.resume:
            train_args['resume'] = True
        
        print("\n📋 Training Configuration:")
        print("-" * 70)
        for key, value in train_args.items():
            if key not in ['verbose', 'plots']:
                print(f"  {key}: {value}")
        
        print("\n🏋️  Training started...")
        print("=" * 70)
        
        # Train the model
        results = model.train(**train_args)
        
        print("\n" + "=" * 70)
        print("✅ TRAINING COMPLETED")
        print("=" * 70)
        print(f"End time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
        # Find output directory
        save_dir = Path(args.project) / args.name
        
        print(f"\n📁 Output Directory: {save_dir}")
        print("\n📊 Generated Files:")
        
        # Check for generated files
        weights_dir = save_dir / "weights"
        if weights_dir.exists():
            best_pt = weights_dir / "best.pt"
            last_pt = weights_dir / "last.pt"
            
            if best_pt.exists():
                size_mb = best_pt.stat().st_size / 1024**2
                print(f"  ✅ best.pt ({size_mb:.2f} MB)")
            else:
                print(f"  ⚠️  best.pt not found")
            
            if last_pt.exists():
                size_mb = last_pt.stat().st_size / 1024**2
                print(f"  ✅ last.pt ({size_mb:.2f} MB)")
            else:
                print(f"  ⚠️  last.pt not found")
        
        # Check for other files
        files_to_check = [
            "results.csv",
            "results.png",
            "confusion_matrix.png",
            "F1_curve.png",
            "PR_curve.png",
            "args.yaml"
        ]
        
        for file in files_to_check:
            file_path = save_dir / file
            if file_path.exists():
                print(f"  ✅ {file}")
        
        # Print final metrics
        if results:
            print("\n📈 Final Metrics:")
            print("-" * 70)
            # Results object contains metrics
            # Print what's available
            
        print("\n💡 Next Steps:")
        print("  1. Check training curves: results.png")
        print("  2. Use best.pt for inference:")
        print(f"     yolo predict model={weights_dir / 'best.pt'} source=<image_path>")
        print("  3. Validate model performance:")
        print(f"     yolo val model={weights_dir / 'best.pt'} data={args.data}")
        
        return True
        
    except KeyboardInterrupt:
        print("\n\n⚠️  Training interrupted by user")
        return False
        
    except torch.cuda.OutOfMemoryError:
        print("\n\n❌ CUDA Out of Memory Error!")
        print("💡 Solutions:")
        print(f"  1. Reduce batch size: --batch {args.batch // 2}")
        print(f"  2. Reduce image size: --imgsz 480")
        print(f"  3. Use smaller model: --weights yolov8n.pt")
        print(f"  4. Disable AMP: remove --amp flag")
        return False
        
    except Exception as e:
        print(f"\n\n❌ Training failed with error:")
        print(f"   {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Main execution function."""
    args = parse_args()
    
    print("\n" + "=" * 70)
    print("YOLOv8 TRAINING SCRIPT")
    print("=" * 70)
    print("Hardware: NVIDIA RTX A3000 (6GB VRAM)")
    print("CUDA: 11.7")
    print("=" * 70)
    
    success = train_model(args)
    
    if success:
        print("\n🎉 Training completed successfully!")
        sys.exit(0)
    else:
        print("\n❌ Training failed or was interrupted")
        sys.exit(1)


if __name__ == "__main__":
    main()