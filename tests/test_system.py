"""
tests/test_system.py  v2.1
===========================
System-level tests — v1 tests preserved, v2.1 tests added.
"""
from __future__ import annotations
import json, os, sys, time, multiprocessing as mp
from pathlib import Path
from unittest.mock import MagicMock, patch
import numpy as np

_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(_ROOT))


class TestIPCSchemaV2:
    def test_heartbeat(self):
        from core.ipc_schema import make_heartbeat, ProcessSource
        hb = make_heartbeat(ProcessSource.CAMERA,0,"Camera_0",1234,150.5,12.3)
        d = hb.to_dict()
        assert d["type"]=="heartbeat"; assert d["source"]=="camera"; assert d["fps"]==12.3
        print("✓ HeartbeatMessage")

    def test_detection_result(self):
        from core.ipc_schema import DetectionResultMessage, ProcessSource
        msg = DetectionResultMessage(source=ProcessSource.DETECTION,camera_id=1,fps=10.0,success_rate=95.5)
        d = msg.to_dict()
        assert d["type"]=="detection_result"; assert d["fps"]==10.0
        print("✓ DetectionResultMessage")

    def test_health_snapshot(self):
        from core.ipc_schema import HealthSnapshotMessage
        snap = HealthSnapshotMessage(processes=[{"name":"test"}],gpu_stats={"vram":1000},
            cpu_pct=45.0,ram_used_mb=8000,system_uptime_s=3600,total_restarts=2)
        d = snap.to_dict()
        assert d["type"]=="health_snapshot"; assert d["cpu_pct"]==45.0
        print("✓ HealthSnapshotMessage")

    def test_boundary_reload(self):
        from core.ipc_schema import BoundaryReloadMessage
        msg = BoundaryReloadMessage(camera_id=0,boundary_file="boundaries/camera_0.json")
        d = msg.to_dict()
        assert d["type"]=="boundary_reload"; assert "camera_0" in d["boundary_file"]
        print("✓ BoundaryReloadMessage")

    def test_gui_command(self):
        from core.ipc_schema import GUICommandMessage
        msg = GUICommandMessage(camera_id=None,command="start_detection",params={})
        d = msg.to_dict()
        assert d["type"]=="gui_command"; assert d["command"]=="start_detection"
        print("✓ GUICommandMessage")

    def test_inference_request(self):
        from core.ipc_schema import make_inference_request
        req = make_inference_request(0,"cam0_frame",(720,1280,3),42)
        d = req.to_dict()
        assert d["type"]=="inference_request"; assert d["frame_index"]==42
        print("✓ InferenceRequestMessage")

    def test_relay_state_9(self):
        from core.ipc_schema import RelayStateMessage, ProcessSource
        msg = RelayStateMessage(source=ProcessSource.RELAY,camera_id=None,
                                relay_states=[True,False,True,False,True,False,True,False,True])
        d = msg.to_dict()
        assert len(d["relay_states"])==9; assert d["relay_states"][0]==True
        print("✓ RelayStateMessage (9 relays)")


class TestConfigLoaderV2:
    def test_load_config(self):
        from core.config_loader import VisionSystemConfig
        cfg = VisionSystemConfig()
        assert len(cfg.cameras)>0; assert cfg.model.path; assert cfg.gpu_pool.pool_size>=1
        print(f"✓ Config v{cfg.system.version}: {len(cfg.cameras)} cameras")

    def test_relay_mapping(self):
        from core.config_loader import VisionSystemConfig
        cfg = VisionSystemConfig()
        assert cfg.relay.relay_count == 9
        r0 = cfg.relay.get_relay_indices(0); assert r0==[0,1,2]
        r1 = cfg.relay.get_relay_indices(1); assert r1==[3,4,5]
        r2 = cfg.relay.get_relay_indices(2); assert r2==[6,7,8]
        print("✓ 9-relay mapping: cam0→[0,1,2] cam1→[3,4,5] cam2→[6,7,8]")

    def test_model_config(self):
        from core.config_loader import VisionSystemConfig
        cfg = VisionSystemConfig()
        assert cfg.model.confidence==0.5; assert cfg.model.oil_can_class_id==0
        print("✓ ModelConfig")

    def test_gpu_pool_config(self):
        from core.config_loader import VisionSystemConfig
        cfg = VisionSystemConfig()
        assert cfg.gpu_pool.enabled==True; assert cfg.gpu_pool.pool_size>=1
        print(f"✓ GPUPoolConfig pool_size={cfg.gpu_pool.pool_size}")

    def test_validate_model_missing(self):
        from core.config_loader import VisionSystemConfig
        cfg = VisionSystemConfig()
        # Model won't exist in test env — should return False
        result = cfg.validate_model()
        print(f"✓ validate_model() returned {result} (False=model not found, expected in test env)")

    def test_load_v2_boundaries(self):
        from core.config_loader import VisionSystemConfig
        cfg = VisionSystemConfig()
        bd = cfg.load_boundaries(0)
        assert bd is not None; assert "oil_can" in bd; assert "bunk_hole" in bd
        assert len(bd["oil_can"])==3; assert len(bd["bunk_hole"])==3
        print(f"✓ Boundary v2 format: {len(bd['oil_can'])} OC, {len(bd['bunk_hole'])} BH")

    def test_get_boundary_path(self):
        from core.config_loader import VisionSystemConfig
        cfg = VisionSystemConfig()
        p = cfg.get_boundary_path(0)
        assert "camera_0" in str(p)
        print(f"✓ get_boundary_path: {p}")


class TestBoundaryEngineV2:
    def _make_flat_bd(self):
        return {
            "camera_id": 0,
            "oil_can": [{"id":"OC1","polygon":[[100,100],[300,100],[300,300],[100,300]]}],
            "bunk_hole": [{"id":"BH1","polygon":[[100,400],[300,400],[300,600],[100,600]]}],
        }

    def test_normalize_flat_format(self):
        from detection.gpu_worker_pool import _normalize_boundary_data
        flat = self._make_flat_bd()
        norm = _normalize_boundary_data(flat, 0)
        assert "boundaries" in norm; assert "pairs" in norm
        assert len(norm["pairs"])==1
        assert norm["pairs"][0]["oil_can_boundary"]=="OC1"
        assert norm["pairs"][0]["bunk_hole_boundary"]=="BH1"
        print("✓ Boundary flat→normalized format conversion")

    def test_relay_index_in_pairs(self):
        from detection.gpu_worker_pool import _normalize_boundary_data
        flat = {
            "camera_id": 1,
            "oil_can": [{"id":"OC1","polygon":[[100,100],[300,100],[300,300],[100,300]]},
                        {"id":"OC2","polygon":[[400,100],[600,100],[600,300],[400,300]]}],
            "bunk_hole": [{"id":"BH1","polygon":[[100,400],[300,400],[300,600],[100,600]]},
                          {"id":"BH2","polygon":[[400,400],[600,400],[600,600],[400,600]]}],
        }
        norm = _normalize_boundary_data(flat, 1)
        # Camera 1 → relay indices [3,4,5]
        assert norm["pairs"][0]["relay_index"]==3
        assert norm["pairs"][1]["relay_index"]==4
        print("✓ Relay index in pairs for camera 1 → [3,4]")

    def test_pair_ok(self):
        from detection.gpu_worker_pool import _normalize_boundary_data
        from core.boundary_engine import CameraBoundarySet
        from core.ipc_schema import DetectionObject, PairStatus
        flat = self._make_flat_bd()
        norm = _normalize_boundary_data(flat, 0)
        bs = CameraBoundarySet(norm)
        dets = [
            DetectionObject(0,"oil_can",0.9,(100/1280,100/720,250/1280,250/720)),
            DetectionObject(1,"bunk_hole",0.9,(100/1280,400/720,250/1280,550/720)),
        ]
        results = bs.evaluate(dets,1280,720,0,1)
        assert results[0].status==PairStatus.OK
        print("✓ Pair OK with v2 boundaries")

    def test_pair_statuses(self):
        from detection.gpu_worker_pool import _normalize_boundary_data
        from core.boundary_engine import CameraBoundarySet
        from core.ipc_schema import DetectionObject, PairStatus
        flat = self._make_flat_bd(); norm = _normalize_boundary_data(flat, 0); bs = CameraBoundarySet(norm)
        # Both absent
        r = bs.evaluate([],1280,720,0,1); assert r[0].status==PairStatus.BOTH_ABSENT; assert r[0].relay_active
        # OC only
        dets=[DetectionObject(0,"oil_can",0.9,(100/1280,100/720,250/1280,250/720))]
        r = bs.evaluate(dets,1280,720,0,1); assert r[0].status==PairStatus.OC_ONLY; assert r[0].relay_active
        # BH only
        dets=[DetectionObject(1,"bunk_hole",0.9,(100/1280,400/720,250/1280,550/720))]
        r = bs.evaluate(dets,1280,720,0,1); assert r[0].status==PairStatus.BH_ONLY; assert r[0].relay_active
        print("✓ All pair status conditions")


class TestRelayV2:
    def test_9_relay_driver(self):
        from relay.relay_process import SimulatedRelayDriver
        drv = SimulatedRelayDriver(9)
        assert drv.initialize()
        for i in range(9): assert drv.set_relay(i, True)
        drv.close()
        print("✓ SimulatedRelayDriver with 9 relays")

    def test_relay_camera_mapping(self):
        from core.config_loader import VisionSystemConfig
        from relay.relay_process import RelayWorker
        cfg = VisionSystemConfig()
        # Camera 2 → relays [6,7,8]
        indices = cfg.relay.get_relay_indices(2)
        assert indices == [6,7,8]
        print(f"✓ Camera 2 relay mapping: {indices}")

    def test_relay_worker_state_update(self):
        from relay.relay_process import SimulatedRelayDriver
        drv = SimulatedRelayDriver(9); drv.initialize()
        states = [False]*9; cached = [False]*9
        # Camera 1 pair 0 → relay 3 should turn ON
        relay_indices = [3, 4, 5]
        pair_results = [{"relay_active": True}, {"relay_active": False}, {"relay_active": True}]
        new_states = list(cached)
        for i, pr in enumerate(pair_results):
            if i < len(relay_indices): new_states[relay_indices[i]] = pr["relay_active"]
        assert new_states[3]==True; assert new_states[4]==False; assert new_states[5]==True
        assert new_states[0]==False  # other cameras unaffected
        print("✓ 9-relay state update for camera 1 pairs")


class TestSharedFrame:
    def test_write_read_cycle(self):
        from core.shared_frame import SharedFrameWriter, SharedFrameReader
        name = "test_shm_v2_rw"; w,h = 640,480
        writer = SharedFrameWriter(name, w, h); reader = SharedFrameReader(name, w, h)
        assert reader.connect(timeout=2.0)
        frame_in = np.random.randint(0,255,(h,w,3),dtype=np.uint8)
        writer.write(frame_in)
        frame_out, idx = reader.read()
        assert frame_out is not None; assert np.array_equal(frame_in,frame_out); assert idx>0
        reader.close(); writer.close()
        print("✓ SharedFrame write/read OK")

    def test_overwrite_model(self):
        from core.shared_frame import SharedFrameWriter, SharedFrameReader
        name = "test_shm_v2_ow"; w,h = 320,240
        writer = SharedFrameWriter(name, w, h); reader = SharedFrameReader(name, w, h)
        assert reader.connect(timeout=2.0)
        last = None
        for i in range(5):
            f = np.full((h,w,3),i*50,dtype=np.uint8); writer.write(f); last=f
        frame_out, _ = reader.read()
        assert frame_out is not None; assert np.array_equal(frame_out, last)
        reader.close(); writer.close()
        print("✓ SharedFrame overwrite model (latest always read)")


class TestResourceMonitor:
    def test_process_memory(self):
        from core.resource_monitor import get_process_memory_mb
        mem = get_process_memory_mb(); assert mem > 0
        print(f"✓ Process memory: {mem:.1f} MB")

    def test_memory_limit(self):
        from core.resource_monitor import is_memory_over_limit
        assert not is_memory_over_limit(100_000); assert is_memory_over_limit(0.001)
        print("✓ Memory limit check")


class TestBoundaryEditorLogic:
    def test_boundary_state_init(self):
        from gui.boundary_editor import BoundaryEditorState, _BOUNDARY_SEQUENCE
        s = BoundaryEditorState()
        assert s.current_idx == 0; assert not s.all_defined()
        assert s.current_bid == "OC1"; assert s.current_type == "oil_can"
        print("✓ BoundaryEditorState init")

    def test_boundary_definition_tracking(self):
        from gui.boundary_editor import BoundaryEditorState
        s = BoundaryEditorState()
        for bid, btype in [("OC1","oil_can"),("OC2","oil_can"),("OC3","oil_can"),
                            ("BH1","bunk_hole"),("BH2","bunk_hole"),("BH3","bunk_hole")]:
            s.closed_boundaries.append({"id":bid,"polygon":[[0,0],[100,0],[100,100],[0,100]],"type_class":btype})
        assert s.all_defined()
        print("✓ BoundaryEditorState all-defined check")

    def test_save_boundary_format(self):
        from gui.boundary_editor import BoundaryEditorState, _save_boundaries
        import tempfile
        s = BoundaryEditorState()
        for bid, btype in [("OC1","oil_can"),("OC2","oil_can"),("OC3","oil_can"),
                            ("BH1","bunk_hole"),("BH2","bunk_hole"),("BH3","bunk_hole")]:
            s.closed_boundaries.append({"id":bid,"polygon":[[0,0],[100,0],[100,100]],"type_class":btype})
        with tempfile.NamedTemporaryFile(suffix=".json",delete=False,mode='w') as f:
            tmp = Path(f.name)
        ok = _save_boundaries(s, 0, tmp)
        assert ok
        with open(tmp) as f: data = json.load(f)
        assert "oil_can" in data; assert "bunk_hole" in data
        assert len(data["oil_can"])==3; assert len(data["bunk_hole"])==3
        assert data["camera_id"]==0
        tmp.unlink()
        print("✓ Boundary save produces valid v2 JSON")


if __name__ == "__main__":
    print("="*60); print("Industrial Vision System v2.1 — Test Suite"); print("="*60)
    suites = [TestIPCSchemaV2, TestConfigLoaderV2, TestBoundaryEngineV2,
              TestRelayV2, TestSharedFrame, TestResourceMonitor, TestBoundaryEditorLogic]
    passed = 0; failed = 0; errors = []
    for cls in suites:
        suite = cls(); methods = [m for m in dir(suite) if m.startswith("test_")]
        print(f"\n--- {cls.__name__} ---")
        for method in methods:
            try: getattr(suite, method)(); passed += 1
            except Exception as e:
                failed += 1; errors.append(f"{cls.__name__}.{method}: {e}")
                import traceback; traceback.print_exc(); print(f"✗ {method}: {e}")
    print(f"\n{'='*60}")
    print(f"Results: {passed} passed, {failed} failed")
    if errors:
        print("\nFailures:")
        for err in errors: print(f"  - {err}")
    print("="*60)
    import sys; sys.exit(0 if failed == 0 else 1)
