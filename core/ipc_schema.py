"""
core/ipc_schema.py  (v2.1 — extended, fully backward-compatible)
"""
from __future__ import annotations
import time, uuid
from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Optional, Tuple
from enum import Enum

class MessageType(str, Enum):
    FRAME_READY       = "frame_ready"
    DETECTION_RESULT  = "detection_result"
    VISUAL_UPDATE     = "visual_update"
    HEARTBEAT         = "heartbeat"
    RESTART           = "restart"
    ERROR             = "error"
    HEALTH_REPORT     = "health_report"
    RELAY_COMMAND     = "relay_command"
    RELAY_STATE       = "relay_state"
    SHUTDOWN          = "shutdown"
    GPU_STATS         = "gpu_stats"
    PROCESS_STATUS    = "process_status"
    INFERENCE_REQUEST = "inference_request"
    HEALTH_SNAPSHOT   = "health_snapshot"
    BOUNDARY_RELOAD   = "boundary_reload"
    GUI_COMMAND       = "gui_command"

class ProcessSource(str, Enum):
    SUPERVISOR  = "supervisor"
    CAMERA      = "camera"
    DETECTION   = "detection"
    RELAY       = "relay"
    GUI         = "gui"
    GPU_MONITOR = "gpu_monitor"
    HEALTH      = "health_monitor"
    GPU_POOL    = "gpu_pool"

class PairStatus(str, Enum):
    OK          = "OK"
    PROBLEM     = "PROBLEM"
    BOTH_ABSENT = "BOTH_ABSENT"
    OC_ONLY     = "OC_ONLY"
    BH_ONLY     = "BH_ONLY"
    UNKNOWN     = "UNKNOWN"

@dataclass
class DetectionObject:
    class_id: int
    class_name: str
    confidence: float
    bbox: Tuple[float,float,float,float]
    boundary_id: Optional[str] = None
    def to_dict(self): return asdict(self)
    @classmethod
    def from_dict(cls,d): return cls(**d)

@dataclass
class PairResult:
    pair_id: int
    pair_name: str
    oil_can_boundary: str
    bunk_hole_boundary: str
    oil_can_present: bool
    bunk_hole_present: bool
    status: PairStatus
    relay_index: int
    relay_active: bool
    def to_dict(self):
        d = asdict(self); d["status"] = self.status.value; return d
    @classmethod
    def from_dict(cls,d):
        d["status"] = PairStatus(d["status"]); return cls(**d)

@dataclass
class BaseMessage:
    type: MessageType
    source: ProcessSource
    camera_id: Optional[int]
    timestamp: float = field(default_factory=time.time)
    message_id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    def to_dict(self):
        return {"type":self.type.value,"source":self.source.value,
                "camera_id":self.camera_id,"timestamp":self.timestamp,
                "message_id":self.message_id}

@dataclass
class FrameReadyMessage(BaseMessage):
    type: MessageType = field(default=MessageType.FRAME_READY,init=False)
    shm_name: str = ""; frame_shape: Tuple = (720,1280,3); frame_index: int = 0
    def to_dict(self):
        d=super().to_dict(); d.update({"shm_name":self.shm_name,"frame_shape":self.frame_shape,"frame_index":self.frame_index}); return d

@dataclass
class DetectionResultMessage(BaseMessage):
    type: MessageType = field(default=MessageType.DETECTION_RESULT,init=False)
    detections: List[Dict] = field(default_factory=list)
    pair_results: List[Dict] = field(default_factory=list)
    inference_time_ms: float = 0.0; fps: float = 0.0
    total_detections: int = 0; problem_count: int = 0; success_rate: float = 100.0
    frame_shape: Tuple = (720,1280,3); frame_data: Optional[bytes] = None
    def to_dict(self):
        d=super().to_dict(); d.update({"detections":self.detections,"pair_results":self.pair_results,
            "inference_time_ms":self.inference_time_ms,"fps":self.fps,
            "total_detections":self.total_detections,"problem_count":self.problem_count,
            "success_rate":self.success_rate,"frame_shape":self.frame_shape,"frame_data":self.frame_data}); return d

@dataclass
class HeartbeatMessage(BaseMessage):
    type: MessageType = field(default=MessageType.HEARTBEAT,init=False)
    process_name: str=""; pid: int=0; memory_mb: float=0.0; fps: float=0.0; status: str="running"; extra: Dict=field(default_factory=dict)
    def to_dict(self):
        d=super().to_dict(); d.update({"process_name":self.process_name,"pid":self.pid,
            "memory_mb":self.memory_mb,"fps":self.fps,"status":self.status,"extra":self.extra}); return d

@dataclass
class RelayCommandMessage(BaseMessage):
    type: MessageType = field(default=MessageType.RELAY_COMMAND,init=False)
    relay_states: List[bool] = field(default_factory=lambda:[False]*9)
    def to_dict(self):
        d=super().to_dict(); d["relay_states"]=self.relay_states; return d

@dataclass
class RelayStateMessage(BaseMessage):
    type: MessageType = field(default=MessageType.RELAY_STATE,init=False)
    relay_states: List[bool] = field(default_factory=lambda:[False]*9)
    last_update: float = field(default_factory=time.time)
    def to_dict(self):
        d=super().to_dict(); d["relay_states"]=self.relay_states; d["last_update"]=self.last_update; return d

@dataclass
class ErrorMessage(BaseMessage):
    type: MessageType = field(default=MessageType.ERROR,init=False)
    error_type: str=""; error_msg: str=""; traceback: str=""; severity: str="error"
    def to_dict(self):
        d=super().to_dict(); d.update({"error_type":self.error_type,"error_msg":self.error_msg,"traceback":self.traceback,"severity":self.severity}); return d

@dataclass
class GPUStatsMessage(BaseMessage):
    type: MessageType = field(default=MessageType.GPU_STATS,init=False)
    source: ProcessSource = field(default=ProcessSource.GPU_MONITOR,init=False)
    camera_id: Optional[int]=None; vram_used_mb: float=0.0; vram_total_mb: float=6144.0
    temperature_c: float=0.0; utilization_pct: float=0.0; power_w: float=0.0; throttle_fps: Optional[float]=None
    def to_dict(self):
        d=super().to_dict(); d.update({"vram_used_mb":self.vram_used_mb,"vram_total_mb":self.vram_total_mb,
            "temperature_c":self.temperature_c,"utilization_pct":self.utilization_pct,
            "power_w":self.power_w,"throttle_fps":self.throttle_fps}); return d

@dataclass
class ShutdownMessage(BaseMessage):
    type: MessageType = field(default=MessageType.SHUTDOWN,init=False)
    reason: str="normal"
    def to_dict(self):
        d=super().to_dict(); d["reason"]=self.reason; return d

@dataclass
class InferenceRequestMessage(BaseMessage):
    type: MessageType = field(default=MessageType.INFERENCE_REQUEST,init=False)
    shm_name: str=""; frame_shape: Tuple=(720,1280,3); frame_index: int=0
    def to_dict(self):
        d=super().to_dict(); d.update({"shm_name":self.shm_name,"frame_shape":self.frame_shape,"frame_index":self.frame_index}); return d

@dataclass
class HealthSnapshotMessage(BaseMessage):
    type: MessageType = field(default=MessageType.HEALTH_SNAPSHOT,init=False)
    source: ProcessSource = field(default=ProcessSource.SUPERVISOR,init=False)
    camera_id: Optional[int]=None
    processes: List[Dict]=field(default_factory=list)
    gpu_stats: Dict=field(default_factory=dict)
    cpu_pct: float=0.0; ram_used_mb: float=0.0; system_uptime_s: float=0.0; total_restarts: int=0
    def to_dict(self):
        d=super().to_dict(); d.update({"processes":self.processes,"gpu_stats":self.gpu_stats,
            "cpu_pct":self.cpu_pct,"ram_used_mb":self.ram_used_mb,
            "system_uptime_s":self.system_uptime_s,"total_restarts":self.total_restarts}); return d

@dataclass
class BoundaryReloadMessage(BaseMessage):
    type: MessageType = field(default=MessageType.BOUNDARY_RELOAD,init=False)
    source: ProcessSource = field(default=ProcessSource.GUI,init=False)
    boundary_file: str=""
    def to_dict(self):
        d=super().to_dict(); d["boundary_file"]=self.boundary_file; return d

@dataclass
class GUICommandMessage(BaseMessage):
    type: MessageType = field(default=MessageType.GUI_COMMAND,init=False)
    source: ProcessSource = field(default=ProcessSource.GUI,init=False)
    command: str=""; params: Dict=field(default_factory=dict)
    def to_dict(self):
        d=super().to_dict(); d["command"]=self.command; d["params"]=self.params; return d

def make_heartbeat(source,camera_id,process_name,pid,memory_mb,fps,status="running",extra=None):
    return HeartbeatMessage(source=source,camera_id=camera_id,process_name=process_name,
        pid=pid,memory_mb=memory_mb,fps=fps,status=status,extra=extra or {})

def make_error(source,camera_id,error_type,error_msg,traceback="",severity="error"):
    return ErrorMessage(source=source,camera_id=camera_id,error_type=error_type,
        error_msg=error_msg,traceback=traceback,severity=severity)

def make_inference_request(camera_id,shm_name,frame_shape,frame_index):
    return InferenceRequestMessage(source=ProcessSource.CAMERA,camera_id=camera_id,
        shm_name=shm_name,frame_shape=frame_shape,frame_index=frame_index)
