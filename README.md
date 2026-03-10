# Industrial Multi-Camera Vision Detection System v2.0

## Overview

Production-grade, self-healing, GPU-accelerated, multi-camera machine vision system for 24/7 industrial operation. Replaces monolithic QThread architecture with fully process-isolated, supervisor-controlled framework.

---

## Architecture

```
Supervisor
├── Camera Process × 3      (RTSP capture → Shared Memory)
├── Detection Process × 3   (YOLO GPU → Boundary Pairing → Results)
├── Relay Process            (USB Relay control)
├── GUI Process              (OpenCV visualization)
└── GPU Monitor Process      (VRAM + Temperature watchdog)
```

**IPC Model:**
- Frames: `multiprocessing.shared_memory` (overwrite model, no backlog)
- Control: `multiprocessing.Queue` with typed schema
- All messages include: `type`, `source`, `camera_id`, `timestamp`, `payload`

---

## Hardware Requirements

| Component | Minimum | Recommended |
|-----------|---------|-------------|
| GPU | RTX 3050 6GB | RTX 3060 8GB+ |
| RAM | 16 GB | 32 GB |
| OS | Windows 11 | Windows 11 |
| Cameras | 3× RTSP/PoE | 3× RTSP/PoE |
| USB | USB-A for relay | USB-A for relay |

---

## Folder Structure

```
vision_system/
├── main.py                      # Entry point
├── config/
│   └── config.yaml              # All system parameters
├── boundaries/
│   ├── camera_0_boundaries.json
│   ├── camera_1_boundaries.json
│   └── camera_2_boundaries.json
├── models/
│   └── best.pt                  # YOLO model (place here)
├── core/
│   ├── ipc_schema.py            # Message schema
│   ├── config_loader.py         # Config access
│   ├── logging_setup.py         # Logging
│   ├── resource_monitor.py      # Memory/GPU monitoring
│   ├── shared_frame.py          # Shared memory transport
│   └── boundary_engine.py       # Boundary + pair logic
├── camera/
│   └── camera_process.py        # RTSP capture process
├── detection/
│   └── detection_process.py     # YOLO GPU detection process
├── relay/
│   └── relay_process.py         # USB relay control process
├── gui/
│   └── gui_process.py           # Visualization process
├── supervisor/
│   ├── supervisor.py            # Master supervisor
│   └── gpu_monitor.py           # GPU health monitor
├── logs/                        # Auto-created, rotating logs
├── scripts/
│   ├── setup.bat                # Environment setup
│   ├── install_service.bat      # NSSM service install
│   └── health_dashboard.py      # CLI health monitor
└── tests/
    ├── test_system.py           # Unit/integration tests
    └── simulate_failures.py     # Failure simulation
```

---

## Installation

### Step 1: Python Environment

```bat
scripts\setup.bat
```

Or manually:
```bat
python -m venv venv
venv\Scripts\activate

# CUDA PyTorch (CUDA 11.8)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118

# Other dependencies
pip install numpy opencv-python PyYAML psutil ultralytics pynvml

# USB relay (optional)
pip install pyhid-usb-relay hid
```

### Step 2: Configure

Edit `config/config.yaml`:
```yaml
cameras:
  - id: 0
    rtsp_url: "rtsp://admin:password@192.168.1.101:554/stream1"
```

### Step 3: Boundaries

Edit `boundaries/camera_X_boundaries.json` for each camera.

Boundary types supported:
- `polygon`: `"points": [[x1,y1],[x2,y2],...]`
- `rect`: `"points": [[x1,y1],[x2,y2]]`

### Step 4: YOLO Model

Place your trained model at: `models/best.pt`

### Step 5: Test

```bat
python tests\test_system.py
```

### Step 6: Run

```bat
python main.py
```

---

## Windows Service Deployment

```bat
# Run as Administrator
scripts\install_service.bat
```

The system will run as `IndustrialVisionSystem` Windows service, auto-starting on boot.

---

## Detection Logic

### Object Classes
- `oil_can` (class 0)
- `bunk_hole` (class 1)

### Boundary Pairs
| Pair | Oil Can Zone | Bunk Hole Zone | Relay |
|------|-------------|----------------|-------|
| 1 | OC1 | BH1 | Relay 1 |
| 2 | OC2 | BH2 | Relay 2 |
| 3 | OC3 | BH3 | Relay 3 |

### Pair Status Logic
| Oil Can | Bunk Hole | Status | Relay |
|---------|-----------|--------|-------|
| ✓ | ✓ | `OK` | OFF |
| ✗ | ✗ | `BOTH_ABSENT` | **ON** |
| ✓ | ✗ | `OC_ONLY` | **ON** |
| ✗ | ✓ | `BH_ONLY` | **ON** |

---

## Supervisor Recovery Policy

| Condition | Action |
|-----------|--------|
| No heartbeat > 10s | Restart process |
| Memory > limit | Restart process |
| FPS = 0 for 30s | Restart detection |
| Process dies | Immediate restart |
| VRAM > 5200MB | Sequential detection restart |
| GPU temp > 85°C | Throttle FPS |
| GPU temp > 85°C for 30s | Restart detection |
| Every 24 hours | Scheduled restart (all processes) |

---

## Resource Limits

| Process | RAM Limit | Notes |
|---------|-----------|-------|
| Camera | 400 MB | Per camera |
| Detection | 1500 MB | Per camera |
| Relay | 300 MB | Shared |
| GUI | 700 MB | Optional |
| Total GPU VRAM | 5200 MB | Guard |

---

## Logs

Each process writes to its own rotating log in `logs/`:
- `logs/supervisor.log`
- `logs/camera_0.log`, `camera_1.log`, `camera_2.log`
- `logs/detection_0.log`, `detection_1.log`, `detection_2.log`
- `logs/relay.log`
- `logs/gpu_monitor.log`
- `logs/gui.log`
- `logs/*_crash.log` — crash dumps

Max 10MB per file, 5 backups.

---

## Scaling

To add a 4th camera:
1. Add entry to `cameras:` in `config.yaml`
2. Create `boundaries/camera_3_boundaries.json`
3. Restart system — supervisor auto-spawns new processes

No code changes required.

---

## Pre-Production Checklist

- [ ] Static IPs assigned to all cameras
- [ ] Windows Power Plan: High Performance
- [ ] Sleep/hibernate disabled
- [ ] Windows Update auto-restart disabled
- [ ] CUDA PyTorch installed and verified
- [ ] YOLO model placed in `models/best.pt`
- [ ] All 3 camera RTSP URLs validated
- [ ] Boundary JSON files validated per camera
- [ ] USB relay connected and `pyhid_usb_relay` working
- [ ] `python tests/test_system.py` — all pass
- [ ] 72-hour stress test completed
- [ ] Failure simulation tests completed
- [ ] Installed as Windows service via NSSM

---

## Failure Simulation

After deployment, validate recovery:
```bat
python tests\simulate_failures.py --test all
```

Individual tests:
```bat
python tests\simulate_failures.py --test camera --camera-id 0
python tests\simulate_failures.py --test detection
python tests\simulate_failures.py --test relay
python tests\simulate_failures.py --test gui
```

---

## Health Dashboard (CLI)

```bat
python scripts\health_dashboard.py
```

---

## Configuration Reference

All configurable parameters in `config/config.yaml`. Key sections:

| Section | Key Parameters |
|---------|---------------|
| `system` | `daily_restart_interval_hours`, `heartbeat_timeout_seconds` |
| `cameras` | `rtsp_url`, `fps_limit`, `reconnect_max_delay` |
| `detection` | `model_path`, `confidence_threshold`, `use_fp16`, `vram_limit_mb` |
| `relay` | `library`, `retry_attempts`, `reinit_after_failures` |
| `gpu_monitor` | `temperature_threshold_celsius`, `vram_threshold_mb` |
| `supervisor` | `max_restart_attempts`, `sequential_detection_restart_delay` |
