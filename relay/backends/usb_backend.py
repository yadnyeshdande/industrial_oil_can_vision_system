"""
relay/backends/usb_backend.py
==============================
USB relay backend using pyhid_usb_relay.

Wraps the existing PyhidRelayDriver logic from relay_process.py v2.2
and adds auto-reconnect on every write failure, so a Windows USB-stack
glitch or cable wobble does not take the whole relay process down.

pyhid_usb_relay API (>= 0.0.2):
    import pyhid_usb_relay
    device = pyhid_usb_relay.find()          # returns device object or None
    device.set_state(channel_1based, True)   # set channel ON  (1-based)
    device.set_state(channel_1based, False)  # set channel OFF (1-based)
    device.state                             # dict {1: bool, 2: bool, ...}

relay_idx here is always 0-based; we add 1 internally before calling set_state.
"""
from __future__ import annotations
import logging
import time
from typing import Optional

from relay.backends.base_backend import RelayBackend

logger = logging.getLogger(__name__)


class USBRelayBackend(RelayBackend):

    def __init__(self, relay_count: int = 9, reconnect_attempts: int = 3):
        self._relay_count     = relay_count
        self._reconnect_max   = reconnect_attempts
        self._device          = None
        self._last_err: str   = ""

    # ── public property ────────────────────────────────────────────────────
    @property
    def last_error(self) -> str:
        return self._last_err

    # ── RelayBackend interface ─────────────────────────────────────────────
    def connect(self) -> bool:
        try:
            import pyhid_usb_relay
            self._device = pyhid_usb_relay.find()
            if self._device is None:
                self._last_err = (
                    "No USB relay device found — check USB cable and driver. "
                    "Windows: also run fix_pyhid_libusb.py"
                )
                logger.error("[USBBackend] %s", self._last_err)
                return False
            self._last_err = ""
            logger.info("[USBBackend] Connected: %s", self._device)
            return True

        except ImportError:
            self._last_err = (
                "pyhid_usb_relay not installed. "
                "Run: pip install pyhid-usb-relay hid"
            )
            logger.error("[USBBackend] %s", self._last_err)
            return False

        except Exception as e:
            self._last_err = str(e)
            logger.error("[USBBackend] connect error: %s", e)
            return False

    def disconnect(self):
        try:
            if self._device is not None:
                self.all_off()
        except Exception as e:
            logger.warning("[USBBackend] all_off during disconnect failed: %s", e)
        finally:
            self._device = None
            logger.info("[USBBackend] Disconnected")

    def relay_on(self, relay_idx: int) -> bool:
        return self._write(relay_idx, True)

    def relay_off(self, relay_idx: int) -> bool:
        return self._write(relay_idx, False)

    def read_relay(self, relay_idx: int) -> Optional[bool]:
        try:
            if self._device is None:
                self._last_err = "Device not connected"
                return None
            state_map = self._device.state           # {1: bool, 2: bool, ...}
            return bool(state_map.get(relay_idx + 1, False))
        except Exception as e:
            self._last_err = str(e)
            logger.warning("[USBBackend] read_relay ch%d error: %s",
                           relay_idx + 1, e)
            return None

    def all_off(self) -> bool:
        ok = True
        for i in range(self._relay_count):
            if not self._write(i, False):
                ok = False
        return ok

    def health_check(self) -> bool:
        """Read device.state as a lightweight liveness probe."""
        try:
            if self._device is None:
                self._last_err = "Device not connected"
                return False
            _ = self._device.state          # probe — raises on dead handle
            self._last_err = ""
            return True
        except Exception as e:
            self._last_err = str(e)
            logger.warning("[USBBackend] health_check failed: %s", e)
            return False

    # ── internal helpers ───────────────────────────────────────────────────
    def _write(self, relay_idx: int, state: bool) -> bool:
        """
        Write relay state with one auto-reconnect attempt on failure.
        USB HID can freeze under Windows without raising at connect time.
        """
        try:
            if self._device is None:
                raise RuntimeError("Device not connected")
            self._device.set_state(relay_idx + 1, bool(state))  # 1-based
            self._last_err = ""
            return True

        except Exception as e:
            self._last_err = str(e)
            logger.warning(
                "[USBBackend] set_state ch%d=%s failed: %s — attempting reconnect",
                relay_idx + 1, state, e,
            )
            if self._reconnect():
                try:
                    self._device.set_state(relay_idx + 1, bool(state))
                    self._last_err = ""
                    return True
                except Exception as e2:
                    self._last_err = str(e2)
                    logger.error(
                        "[USBBackend] Retry set_state ch%d=%s failed: %s",
                        relay_idx + 1, state, e2,
                    )
            return False

    def _reconnect(self) -> bool:
        logger.warning("[USBBackend] Reconnecting USB device...")
        self._device = None
        for attempt in range(self._reconnect_max):
            time.sleep(0.5 * (attempt + 1))
            try:
                import pyhid_usb_relay
                self._device = pyhid_usb_relay.find()
                if self._device is not None:
                    self._last_err = ""
                    logger.info("[USBBackend] Reconnected on attempt %d", attempt + 1)
                    return True
            except Exception as e:
                logger.debug("[USBBackend] Reconnect attempt %d error: %s",
                             attempt + 1, e)
        self._last_err = (
            f"Reconnect failed after {self._reconnect_max} attempts"
        )
        logger.error("[USBBackend] %s", self._last_err)
        return False