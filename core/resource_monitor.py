"""
core/resource_monitor.py  v2.2
================================
Process + GPU resource monitoring.
v2.2: nvidia-ml-py preferred (pynvml is the legacy name for same package).
      Both names work identically — just try both.
"""
from __future__ import annotations
import logging, os, time
from typing import Optional, Dict

logger = logging.getLogger(__name__)

try:
    import psutil
    _PSUTIL = True
except ImportError:
    _PSUTIL = False
    logger.warning("psutil not installed — memory monitoring limited")

# ── NVML init (works with both 'pynvml' and 'nvidia-ml-py' packages) ──────────
_nvml_initialized = False
_pynvml = None

def _try_import_nvml():
    global _pynvml
    if _pynvml is not None:
        return _pynvml
    # nvidia-ml-py installs as 'pynvml' module in newer versions
    try:
        import pynvml as _m
        _pynvml = _m
        return _pynvml
    except ImportError:
        pass
    # Try the explicit nvidia-ml-py package name
    try:
        import nvidia_ml_py as _m
        _pynvml = _m
        return _pynvml
    except ImportError:
        pass
    return None


def _ensure_nvml():
    global _nvml_initialized
    if _nvml_initialized:
        return True
    m = _try_import_nvml()
    if m is None:
        return False
    try:
        m.nvmlInit()
        _nvml_initialized = True
        logger.info("NVML initialized (%s)", m.__name__ if hasattr(m, '__name__') else 'nvml')
        return True
    except Exception as e:
        logger.debug("NVML init failed: %s", e)
        return False


def get_gpu_stats(device_index: int = 0) -> dict:
    """Return GPU metrics dict. Returns {} on failure."""
    if not _ensure_nvml():
        return {}
    m = _try_import_nvml()
    if m is None:
        return {}
    try:
        handle = m.nvmlDeviceGetHandleByIndex(device_index)
        mem    = m.nvmlDeviceGetMemoryInfo(handle)
        temp   = m.nvmlDeviceGetTemperature(handle, m.NVML_TEMPERATURE_GPU)
        util   = m.nvmlDeviceGetUtilizationRates(handle)
        try:
            power = m.nvmlDeviceGetPowerUsage(handle) / 1000.0
        except Exception:
            power = 0.0
        name = m.nvmlDeviceGetName(handle)
        if isinstance(name, bytes):
            name = name.decode("utf-8", errors="replace")
        return {
            "vram_used_mb":    mem.used    / (1024 * 1024),
            "vram_total_mb":   mem.total   / (1024 * 1024),
            "vram_free_mb":    mem.free    / (1024 * 1024),
            "temperature_c":   float(temp),
            "utilization_pct": float(util.gpu),
            "memory_util_pct": float(util.memory),
            "power_w":         power,
            "name":            name,
        }
    except Exception as e:
        logger.debug("get_gpu_stats error: %s", e)
        return {}


def is_vram_over_limit(limit_mb: float, device_index: int = 0) -> bool:
    stats = get_gpu_stats(device_index)
    used = stats.get("vram_used_mb", 0.0)
    if used > limit_mb:
        logger.warning("VRAM %.1f MB > limit %.1f MB", used, limit_mb)
        return True
    return False


def get_process_memory_mb(pid: Optional[int] = None) -> float:
    """Return RSS memory in MB for this (or given) process."""
    if not _PSUTIL:
        try:
            import resource
            return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0
        except Exception:
            return 0.0
    try:
        proc = psutil.Process(pid or os.getpid())
        return proc.memory_info().rss / (1024 * 1024)
    except Exception:
        return 0.0


def is_memory_over_limit(limit_mb: float) -> bool:
    mem = get_process_memory_mb()
    if mem > limit_mb:
        logger.warning("Memory %.1f MB > limit %.1f MB", mem, limit_mb)
        return True
    return False


def get_cpu_percent() -> float:
    if not _PSUTIL:
        return 0.0
    try:
        return psutil.cpu_percent(interval=None)
    except Exception:
        return 0.0


def get_ram_used_mb() -> float:
    if not _PSUTIL:
        return 0.0
    try:
        return psutil.virtual_memory().used / (1024 * 1024)
    except Exception:
        return 0.0


def shutdown_nvml():
    global _nvml_initialized
    m = _try_import_nvml()
    if m and _nvml_initialized:
        try:
            m.nvmlShutdown()
            _nvml_initialized = False
        except Exception:
            pass
