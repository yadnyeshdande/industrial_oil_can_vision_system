"""
relay/backends/modbus_backend.py
=================================
Waveshare Modbus TCP Ethernet Relay backend.

Device:  Waveshare Modbus POE ETH Relay (16-channel tested; 9 used)
IP:      192.168.1.200  (configurable)
Port:    502            (standard Modbus TCP)
Slave:   1              (configurable)
Library: pymodbus       (pip install pymodbus)

INDUSTRIAL RULE — treat ALL Modbus calls as unreliable network I/O:
  • every call has a timeout
  • every write can fail mid-transaction
  • every connection can drop
  • network switch can reboot without warning

Coil addressing (0-based throughout, matching relay_idx):
    relay_idx 0  →  coil address 0  (Relay 1 on board)
    relay_idx 8  →  coil address 8  (Relay 9 on board)

pymodbus API note:
    The 'device_id' kwarg matches the Waveshare example code.
    If you have a newer pymodbus (3.7+) and see a TypeError, change
    device_id= to slave= on the write_coil / read_coils calls.
"""
from __future__ import annotations
import logging
import time
from typing import Optional, List

from relay.backends.base_backend import RelayBackend

logger = logging.getLogger(__name__)


class ModbusRelayBackend(RelayBackend):

    def __init__(
        self,
        ip:              str = "192.168.1.200",
        port:            int = 502,
        device_id:       int = 1,
        timeout_seconds: int = 2,
        relay_count:     int = 9,
        retry_attempts:  int = 3,
    ):
        self._ip             = ip
        self._port           = port
        self._device_id      = device_id
        self._timeout        = timeout_seconds
        self._relay_count    = relay_count
        self._retry_attempts = retry_attempts
        self._client         = None
        self._connected      = False
        self._last_err: str  = ""

    # ── public property ────────────────────────────────────────────────────
    @property
    def last_error(self) -> str:
        return self._last_err

    # ── RelayBackend interface ─────────────────────────────────────────────
    def connect(self) -> bool:
        try:
            from pymodbus.client import ModbusTcpClient
            self._client = ModbusTcpClient(
                host=self._ip,
                port=self._port,
                timeout=self._timeout,          # ← NEVER omit timeout
            )
            connected = self._client.connect()
            if not connected:
                self._last_err = (
                    f"Could not connect to Modbus relay at {self._ip}:{self._port}"
                )
                logger.error("[ModbusBackend] %s", self._last_err)
                self._connected = False
                return False
            self._connected = True
            self._last_err  = ""
            logger.info("[ModbusBackend] Connected to %s:%d  slave=%d",
                        self._ip, self._port, self._device_id)
            return True

        except ImportError:
            self._last_err = "pymodbus not installed — run: pip install pymodbus"
            logger.error("[ModbusBackend] %s", self._last_err)
            return False

        except Exception as e:
            self._last_err  = str(e)
            self._connected = False
            logger.error("[ModbusBackend] connect error: %s", e)
            return False

    def disconnect(self):
        try:
            if self._client and self._connected:
                self.all_off()
                self._client.close()
        except Exception as e:
            logger.warning("[ModbusBackend] disconnect error: %s", e)
        finally:
            self._client    = None
            self._connected = False
            logger.info("[ModbusBackend] Disconnected from %s:%d",
                        self._ip, self._port)

    def relay_on(self, relay_idx: int) -> bool:
        return self._write_coil(relay_idx, True)

    def relay_off(self, relay_idx: int) -> bool:
        return self._write_coil(relay_idx, False)

    def read_relay(self, relay_idx: int) -> Optional[bool]:
        try:
            if not self._ensure_connected():
                return None
            result = self._client.read_coils(
                address=relay_idx,
                count=1,
                device_id=self._device_id,
            )
            if hasattr(result, "isError") and result.isError():
                self._last_err = f"read_coils failed on relay idx {relay_idx}"
                logger.warning("[ModbusBackend] %s", self._last_err)
                return None
            self._last_err = ""
            return bool(result.bits[0])
        except Exception as e:
            self._last_err  = str(e)
            self._connected = False
            logger.warning("[ModbusBackend] read_relay idx %d error: %s",
                           relay_idx, e)
            return None

    def all_off(self) -> bool:
        """Write False to all relay coils in a single Modbus transaction."""
        try:
            if not self._ensure_connected():
                return False
            values = [False] * self._relay_count
            result = self._client.write_coils(
                address=0,
                values=values,
                device_id=self._device_id,
            )
            if hasattr(result, "isError") and result.isError():
                self._last_err = "write_coils (all_off) returned error"
                logger.error("[ModbusBackend] %s", self._last_err)
                return False
            self._last_err = ""
            logger.debug("[ModbusBackend] All %d relays turned OFF", self._relay_count)
            return True
        except Exception as e:
            self._last_err  = str(e)
            self._connected = False
            logger.error("[ModbusBackend] all_off error: %s", e)
            return False

    def health_check(self) -> bool:
        """Read coil 0 as a lightweight liveness probe."""
        try:
            if not self._ensure_connected():
                return False
            result = self._client.read_coils(
                address=0,
                count=1,
                device_id=self._device_id,
            )
            if hasattr(result, "isError") and result.isError():
                self._last_err = "health_check read_coils returned error"
                logger.warning("[ModbusBackend] %s", self._last_err)
                return False
            self._last_err = ""
            return True
        except Exception as e:
            self._last_err  = str(e)
            self._connected = False
            logger.warning("[ModbusBackend] health_check failed: %s", e)
            return False

    # ── internal helpers ───────────────────────────────────────────────────
    def _ensure_connected(self) -> bool:
        if self._connected and self._client is not None:
            return True
        return self._reconnect()

    def _write_coil(self, relay_idx: int, state: bool) -> bool:
        for attempt in range(self._retry_attempts):
            try:
                if not self._ensure_connected():
                    time.sleep(0.3)
                    continue
                result = self._client.write_coil(
                    address=relay_idx,
                    value=state,
                    device_id=self._device_id,
                )
                if hasattr(result, "isError") and result.isError():
                    raise IOError(
                        f"write_coil error: relay_idx={relay_idx} state={state}"
                    )
                self._last_err = ""
                return True
            except Exception as e:
                self._last_err  = str(e)
                self._connected = False
                logger.warning(
                    "[ModbusBackend] write_coil idx=%d state=%s attempt=%d: %s",
                    relay_idx, state, attempt + 1, e,
                )
                time.sleep(0.3 * (attempt + 1))

        logger.error(
            "[ModbusBackend] write_coil idx=%d state=%s failed after %d attempts",
            relay_idx, state, self._retry_attempts,
        )
        return False

    def _reconnect(self) -> bool:
        logger.warning("[ModbusBackend] Reconnecting to %s:%d...",
                       self._ip, self._port)
        self._connected = False
        try:
            if self._client:
                self._client.close()
        except Exception:
            pass
        self._client = None

        for attempt in range(self._retry_attempts):
            time.sleep(1.0 * (attempt + 1))
            if self.connect():
                logger.info("[ModbusBackend] Reconnected on attempt %d", attempt + 1)
                return True

        self._last_err = (
            f"Reconnect to {self._ip}:{self._port} failed "
            f"after {self._retry_attempts} attempts"
        )
        logger.error("[ModbusBackend] %s", self._last_err)
        return False