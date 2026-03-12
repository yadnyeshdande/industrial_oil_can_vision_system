# Windows Production Setup

## Critical: Increase Paging File

PyTorch + CUDA loads cuBLAS, cuFFT, and other DLLs that each reserve
large virtual address space windows (~500 MB each). With the default
Windows automatic paging file this can cause:

    WinError 1455: The paging file is too small

**Steps:**

1. Open **System Properties** → Advanced → Performance → Settings
2. Go to **Advanced** tab → Virtual Memory → Change
3. Uncheck **Automatically manage paging file size**
4. Select your OS drive (usually C:)
5. Choose **Custom size**:
   - Initial size:  **16384 MB**
   - Maximum size:  **32768 MB**
6. Click Set → OK → **Reboot**

This is required for any PyTorch/CUDA application on Windows.

---

## Expected VRAM Usage (v3.4 shared-model pool)

| Component              | VRAM      |
|------------------------|-----------|
| YOLO model (shared)    | ~1.5 GB   |
| CUDA runtime           | ~400 MB   |
| Tensor/inference bufs  | ~500 MB   |
| Pool overhead          | ~100 MB   |
| **Total**              | **~2.5 GB** |

With `pool_size: 2` and the old per-process architecture this was ~4.6 GB.
The shared-model pool saves approximately 1.5–2 GB of VRAM.

---

## Config Recommendations for RTX 3050 6GB

```yaml
gpu_pool:
  pool_size: 2          # 2 inference threads, 1 shared model
  vram_limit_mb: 6800   # 95% of 6144 MB physical + driver headroom

gpu_monitor:
  vram_threshold_mb: 6800
  vram_sustained_seconds: 45      # ignore spikes < 45s (model warmup)
  vram_restart_cooldown_seconds: 120

supervisor:
  storm_max_restarts: 5
  storm_window_seconds: 120
```

If you have 8 GB VRAM, raise both `vram_limit_mb` and `vram_threshold_mb` to **7500**.
