"""
core/config_loader.py  v2.1
============================
Loads, validates, and provides typed access to config.yaml.
v2.1: Added ModelConfig, GPUPoolConfig, extended RelayConfig (9 relays),
      extended GUIConfig, extended SupervisorConfig.
"""
from __future__ import annotations
import os, json, logging
from pathlib import Path
from typing import Any, Dict, List, Optional
import yaml

logger = logging.getLogger(__name__)
_CONFIG_PATH = Path(__file__).parent.parent / "config" / "config.yaml"
_BOUNDARIES_DIR = Path(__file__).parent.parent / "boundaries"


class CameraConfig:
    def __init__(self, d):
        self.id = d["id"]; self.name = d["name"]; self.rtsp_url = d["rtsp_url"]
        self.enabled = d.get("enabled", True); self.fps_limit = d.get("fps_limit", 12)
        self.reconnect_base_delay = d.get("reconnect_base_delay", 2.0)
        self.reconnect_max_delay = d.get("reconnect_max_delay", 60.0)
        self.reconnect_max_attempts = d.get("reconnect_max_attempts", 0)
        self.buffer_size = d.get("buffer_size", 1)
        self.shared_memory_name = d.get("shared_memory_name", f"cam{self.id}_frame")
        self.frame_width = d.get("frame_width", 1280); self.frame_height = d.get("frame_height", 720)
        self.memory_limit_mb = d.get("memory_limit_mb", 400)


class ModelConfig:
    """v2.1 — top-level model config block."""
    def __init__(self, d):
        self.path = d.get("path", "models/best.pt")
        self.confidence = d.get("confidence", 0.5)
        self.iou = d.get("iou", 0.45)
        self.class_names = d.get("class_names", ["oil_can", "bunk_hole"])
        self.oil_can_class_id = d.get("oil_can_class_id", 0)
        self.bunk_hole_class_id = d.get("bunk_hole_class_id", 1)
        self.strict_boundary_mode = d.get("strict_boundary_mode", True)


class GPUPoolConfig:
    """v2.1 — GPU Worker Pool configuration."""
    def __init__(self, d):
        self.enabled = d.get("enabled", True)
        self.pool_size = d.get("pool_size", 2)
        self.device = d.get("device", "cuda")
        self.use_fp16 = d.get("use_fp16", True)
        self.fps_limit = d.get("fps_limit", 12)
        self.memory_limit_mb = d.get("memory_limit_mb", 2500)
        self.vram_limit_mb = d.get("vram_limit_mb", 5200)
        self.inference_timeout_seconds = d.get("inference_timeout_seconds", 5)
        self.heartbeat_interval_seconds = d.get("heartbeat_interval_seconds", 2)
        self.boundary_poll_interval_seconds = d.get("boundary_poll_interval_seconds", 5)


class DetectionConfig:
    def __init__(self, d):
        self.model_path = d.get("model_path", "models/best.pt")
        self.device = d.get("device", "cuda"); self.use_fp16 = d.get("use_fp16", True)
        self.confidence_threshold = d.get("confidence_threshold", 0.5)
        self.iou_threshold = d.get("iou_threshold", 0.45)
        self.class_names = d.get("class_names", ["oil_can", "bunk_hole"])
        self.oil_can_class_id = d.get("oil_can_class_id", 0)
        self.bunk_hole_class_id = d.get("bunk_hole_class_id", 1)
        self.strict_boundary_mode = d.get("strict_boundary_mode", True)
        self.fps_limit = d.get("fps_limit", 12)
        self.memory_limit_mb = d.get("memory_limit_mb", 1500)
        self.vram_limit_mb = d.get("vram_limit_mb", 5200)
        self.inference_timeout_seconds = d.get("inference_timeout_seconds", 5)
        self.heartbeat_interval_seconds = d.get("heartbeat_interval_seconds", 2)


class RelayConfig:
    """v2.1 — extended to 9 relays with per-camera mapping."""
    def __init__(self, d):
        self.enabled = d.get("enabled", True)
        self.library = d.get("library", "simulated")
        self.retry_attempts = d.get("retry_attempts", 3)
        self.retry_delay_seconds = d.get("retry_delay_seconds", 0.5)
        self.reinit_after_failures = d.get("reinit_after_failures", 5)
        self.memory_limit_mb = d.get("memory_limit_mb", 300)
        self.heartbeat_interval_seconds = d.get("heartbeat_interval_seconds", 2)
        self.relay_count = d.get("relay_count", 9)
        # relay_map: relay_index(0-based) → pair_index
        self.relay_map: Dict[int,int] = {int(k):int(v) for k,v in d.get("relay_map",{}).items()}
        # per-camera relay mapping: "camera_0" -> [0,1,2]
        raw_mapping = d.get("relay_mapping", {
            "camera_0":[0,1,2],"camera_1":[3,4,5],"camera_2":[6,7,8]
        })
        self.relay_mapping: Dict[int, List[int]] = {}
        for key, indices in raw_mapping.items():
            cam_id = int(key.replace("camera_",""))
            self.relay_mapping[cam_id] = [int(i) for i in indices]

    def get_relay_indices(self, camera_id: int) -> List[int]:
        """Return [relay0, relay1, relay2] for a camera (0-based)."""
        return self.relay_mapping.get(camera_id, [camera_id*3, camera_id*3+1, camera_id*3+2])


class GUIConfig:
    def __init__(self, d):
        self.enabled = d.get("enabled", True)
        self.window_title = d.get("window_title", "Industrial Vision System")
        self.window_width = d.get("window_width", 1600)
        self.window_height = d.get("window_height", 900)
        self.update_interval_ms = d.get("update_interval_ms", 50)
        self.memory_limit_mb = d.get("memory_limit_mb", 700)
        self.font_scale = d.get("font_scale", 0.55)
        self.overlay_alpha = d.get("overlay_alpha", 0.25)
        self.show_fps = d.get("show_fps", True)
        self.show_uptime = d.get("show_uptime", True)
        self.show_success_rate = d.get("show_success_rate", True)
        self.show_health_metrics = d.get("show_health_metrics", True)
        self.show_relay_state = d.get("show_relay_state", True)
        self.bounding_box_thickness = d.get("bounding_box_thickness", 2)
        self.boundary_thickness = d.get("boundary_thickness", 2)
        self.enable_boundary_editor = d.get("enable_boundary_editor", True)
        self.start_in_preview_mode = d.get("start_in_preview_mode", False)


class GPUMonitorConfig:
    def __init__(self, d):
        self.enabled                    = d.get("enabled",                      True)
        self.poll_interval_seconds      = d.get("poll_interval_seconds",        5)
        self.vram_threshold_mb          = d.get("vram_threshold_mb",            5800)
        self.temperature_threshold_celsius  = d.get("temperature_threshold_celsius",  85)
        self.critical_temperature_celsius   = d.get("critical_temperature_celsius",   90)
        self.fps_throttle_on_overheat   = d.get("fps_throttle_on_overheat",     8)
        self.restart_on_persistent_overheat = d.get("restart_on_persistent_overheat", True)
        self.overheat_duration_seconds  = d.get("overheat_duration_seconds",    30)
        # Storm-guard fields (new in v3.3 — safe defaults for older configs)
        self.vram_sustained_seconds         = d.get("vram_sustained_seconds",         45)
        self.vram_restart_cooldown_seconds  = d.get("vram_restart_cooldown_seconds",  120)
        self.vram_max_restarts              = d.get("vram_max_restarts",              5)


class SupervisorConfig:
    def __init__(self, d):
        self.memory_poll_interval_seconds       = d.get("memory_poll_interval_seconds", 10)
        self.health_log_interval_seconds        = d.get("health_log_interval_seconds", 60)
        self.health_broadcast_interval_seconds  = d.get("health_broadcast_interval_seconds", 3)
        self.max_restart_attempts               = d.get("max_restart_attempts", 10)
        self.restart_backoff_seconds            = d.get("restart_backoff_seconds", 5)
        self.sequential_detection_restart_delay = d.get("sequential_detection_restart_delay", 10)
        self.process_start_timeout_seconds      = d.get("process_start_timeout_seconds", 30)
        self.validate_model_on_startup          = d.get("validate_model_on_startup", True)
        # Storm guard: if a process restarts > storm_max_restarts times within
        # storm_window_seconds, it is suspended until a supervisor restart.
        # Prevents Windows paging file exhaustion from repeated CUDA DLL loads.
        self.storm_max_restarts                 = d.get("storm_max_restarts", 5)
        self.storm_window_seconds               = d.get("storm_window_seconds", 120)


class LoggingConfig:
    def __init__(self, d):
        self.log_dir = d.get("log_dir", "logs")
        self.max_bytes = d.get("max_bytes", 10485760)
        self.backup_count = d.get("backup_count", 5)
        self.format = d.get("format", "%(asctime)s [%(levelname)s] [%(name)s] %(message)s")
        self.date_format = d.get("date_format", "%Y-%m-%d %H:%M:%S")


class SystemConfig:
    def __init__(self, d):
        self.name = d.get("name", "IndustrialVisionSystem")
        self.version = d.get("version", "2.1")
        self.log_level = d.get("log_level", "INFO")
        self.daily_restart_interval_hours = d.get("daily_restart_interval_hours", 24)
        self.heartbeat_timeout_seconds = d.get("heartbeat_timeout_seconds", 10)
        self.fps_zero_timeout_seconds = d.get("fps_zero_timeout_seconds", 30)
        self.startup_grace_period_seconds = d.get("startup_grace_period_seconds", 15)


class VisionSystemConfig:
    def __init__(self, config_path=None):
        self._path = Path(config_path) if config_path else _CONFIG_PATH
        self._raw: Dict = {}
        self.load()

    def load(self):
        if not self._path.exists():
            raise FileNotFoundError(f"Config not found: {self._path}")
        with open(self._path, "r", encoding="utf-8") as f:
            self._raw = yaml.safe_load(f)
        self.system = SystemConfig(self._raw.get("system", {}))
        self.cameras: List[CameraConfig] = [
            CameraConfig(c) for c in self._raw.get("cameras", []) if c.get("enabled", True)
        ]
        self.model = ModelConfig(self._raw.get("model", {}))
        self.gpu_pool = GPUPoolConfig(self._raw.get("gpu_pool", {}))
        self.detection = DetectionConfig(self._raw.get("detection", {}))
        self.relay = RelayConfig(self._raw.get("relay", {}))
        self.gui = GUIConfig(self._raw.get("gui", {}))
        self.gpu_monitor = GPUMonitorConfig(self._raw.get("gpu_monitor", {}))
        self.supervisor = SupervisorConfig(self._raw.get("supervisor", {}))
        self.logging = LoggingConfig(self._raw.get("logging", {}))
        bd = self._raw.get("boundaries", {})
        self.boundaries_dir = Path(bd.get("boundary_dir", "boundaries"))
        if not self.boundaries_dir.is_absolute():
            self.boundaries_dir = self._path.parent.parent / self.boundaries_dir
        logger.info("Config loaded v%s: %d cameras", self.system.version, len(self.cameras))

    def reload(self):
        self.load()
        logger.info("Config reloaded")

    def get_camera(self, camera_id: int) -> Optional[CameraConfig]:
        return next((c for c in self.cameras if c.id == camera_id), None)

    def load_boundaries(self, camera_id: int) -> Optional[Dict]:
        path = self.boundaries_dir / f"camera_{camera_id}_boundaries.json"
        if not path.exists():
            logger.warning("Boundary file not found: %s", path)
            return None
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)

    def get_boundary_path(self, camera_id: int) -> Path:
        return self.boundaries_dir / f"camera_{camera_id}_boundaries.json"

    @property
    def camera_ids(self) -> List[int]:
        return [c.id for c in self.cameras]

    def validate_model(self) -> bool:
        """Check model file exists. Returns True if OK."""
        model_path = Path(self.model.path)
        if not model_path.is_absolute():
            model_path = self._path.parent.parent / model_path
        if model_path.exists():
            return True
        # Also check detection.model_path for backward compat
        alt = Path(self.detection.model_path)
        if not alt.is_absolute():
            alt = self._path.parent.parent / alt
        return alt.exists()


_instance: Optional[VisionSystemConfig] = None

def get_config(config_path=None) -> VisionSystemConfig:
    global _instance
    if _instance is None:
        _instance = VisionSystemConfig(config_path)
    return _instance

def reload_config():
    global _instance
    if _instance:
        _instance.reload()
