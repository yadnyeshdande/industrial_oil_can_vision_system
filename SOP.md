# YOLO Training SOP (Windows)

This SOP documents how to prepare a Windows machine and create a reproducible Python environment for training Ultralytics YOLO (yolov10/yolov8 family) with GPU support, and how to avoid common pitfalls we encountered (NumPy 2.x ABI, PyTorch wheel compatibility, PyTorch unpickling changes).

Follow the checklist and copy-paste commands for PowerShell (PS) or CMD. Replace paths with your project path.

---

## Prerequisites
- Install Microsoft Visual C++ Build Tools (for some wheels). (You said you already know how to.)
- Install latest NVIDIA driver for your GPU. Verify with `nvidia-smi`.
- Install Python 3.10.10 (recommended for PyTorch compatibility). Verify with `py -3.10 --version`.

---

## Quick verification (before starting)
- GPU and driver: `nvidia-smi`
- Python 3.10 available: `py -3.10 --version`

---

## 1 Create and activate a Python 3.10 venv
From your project directory (example uses `D:\New folder\yolo`):

- Create venv:
  - PowerShell or CMD:

    ```powershell
    py -3.10 -m venv "D:\New folder\yolo\yolo_env_py310"
    ```

- Activate venv:
  - PowerShell:

    ```powershell
    & "D:\New folder\yolo\yolo_env_py310\Scripts\Activate.ps1"
    ```

  - CMD:

    ```cmd
    D:\New folder\yolo\yolo_env_py310\Scripts\activate.bat
    ```

Verify `python -V` returns `3.10.10` (or similar 3.10.x).

---

## 2 Upgrade pip and build tools

```powershell
python -m pip install --upgrade pip setuptools wheel
```

---

## 3 Choose a compatible PyTorch wheel (CUDA)
- Check `nvidia-smi` for driver and reported CUDA version. We used `cu118` (CUDA 11.8) which works with many drivers including 11.7+.

- Install CUDA-enabled PyTorch (example that worked in this SOP):

```powershell
python -m pip install --index-url https://download.pytorch.org/whl/cu118 torch==2.0.1+cu118 torchvision==0.15.2+cu118
```

Notes:
- If dependency conflicts arise, install `torch` and `torchvision` first, then the rest of the packages.
- If you prefer Conda, see the Conda alternative below (recommended for complex binary dependencies).

---

## 4 Install Ultralytics YOLO and common dependencies

Install after PyTorch is confirmed installed (order matters to avoid pip choosing incompatible binaries):

```powershell
python -m pip install ultralytics opencv-python numpy pyyaml tqdm pillow matplotlib scipy
```

If you plan to use `torchaudio`, check compatible torchaudio version for your `torch` wheel and install afterwards.

---

## 5 Common binary compatibility issues & fixes

- NumPy ABI errors (e.g., "A module compiled using NumPy 1.x cannot be run in NumPy 2.x"):
  - Fix: pin to `numpy<2`:

    ```powershell
    python -m pip install "numpy<2"
    ```

  - After downgrading NumPy, re-check packages such as `opencv-python` if they complain; reinstall if required.

- PyTorch UnpicklingError when loading full ultralytics checkpoints (due to `weights_only` default change in PyTorch):
  - Safe workaround (in your training script) to register safe globals or temporarily call `torch.load(..., weights_only=False)` while loading trusted official checkpoints. See the example code snippet below.

---

## 6 Quick verification snippets

- Check torch + CUDA availability:

```powershell
python -c "import torch, ultralytics; print(torch.__version__, torch.cuda.is_available(), getattr(torch.version,'cuda',None), ultralytics.__version__)"
```

- Test loading a model (this may download weights):

```powershell
python - <<'PY'
from ultralytics import YOLO
model = YOLO('yolov10n.pt')
print('Loaded model type:', type(model.model))
PY
```

If loading fails with `UnpicklingError`, apply the safe-globals / temporary `torch.load` workaround (next section).

---

## 7 Recommended safe code snippets (put in your training script before loading weights)

1) Register DetectionModel as safe global (preferred):

```python
import torch
try:
    import ultralytics.nn.tasks as _ut
    if hasattr(torch.serialization, 'add_safe_globals'):
        torch.serialization.add_safe_globals([_ut.DetectionModel])
except Exception:
    pass
```

2) If UnpicklingError persists and you trust the weight file (official releases), temporarily force `weights_only=False` while loading:

```python
# Temporary monkeypatch to call torch.load(..., weights_only=False)
_orig = getattr(torch, 'load', None)
try:
    def _wrap(f, *a, **kw):
        if 'weights_only' not in kw:
            kw['weights_only'] = False
        return _orig(f, *a, **kw)
    torch.load = _wrap
    model = YOLO('yolov10n.pt')
finally:
    if _orig is not None:
        torch.load = _orig
```

Use this only for files you trust (official ultralytics weights). Loading arbitrary pickled checkpoints with `weights_only=False` may execute code.

---

## 8 Example training command (test run)

Use a small test run first (1–2 epochs):

```powershell
python train_updated.py --data data.yaml --weights yolov10n.pt --epochs 2 --batch 4
```

Adjust `--batch` and `--imgsz` for your GPU VRAM. For 6GB VRAM start with `--batch 4` or `--batch 8` depending on `imgsz`.

---

## 9 Output location and resume
- Outputs: `runs/<project>/<name>/weights/best.pt` and `last.pt` plus `results.csv`, `results.png`, `args.yaml`.
- Resume training: pass `--resume` or point `--weights` to `last.pt`.

---

## 10 Conda alternative (recommended for robust binary dependency management)

```powershell
conda create -n yolo310 python=3.10 -y
conda activate yolo310
conda install -c pytorch -c nvidia pytorch torchvision pytorch-cuda=11.8 -y
python -m pip install ultralytics opencv-python pyyaml tqdm pillow
```

Conda often avoids noisy pip ABI conflicts for compiled packages.

---

## 11 Troubleshooting checklist
- `torch.cuda.is_available()` is False: reinstall CUDA-enabled `torch`, confirm drivers and `nvidia-smi`.
- `UnpicklingError`: add safe-globals or use temporary `torch.load` workaround.
- NumPy ABI errors: `pip install "numpy<2"` (or use conda).
- Pip dependency conflict: install `torch` first, then `ultralytics` and others; or use conda.

---

## 12 Final recommendations
- Use Python 3.10 for best wheel support.
- Always run a short smoke test (1–2 epochs) before committing to a long run.
- Keep a small wrapper script (or `train_updated.py`) that sets safe-globals/monkeypatch (see above) so you won't hit the unpickling error.

---

If you want, run the included `setup_yolo_env.ps1` which automates the venv creation and package installs for the common path layout.
