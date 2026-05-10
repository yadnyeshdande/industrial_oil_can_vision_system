"""
relay/backends/base_backend.py
==============================
Abstract contract every relay backend must implement.
relay_idx is always 0-based throughout this interface.
"""
from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Optional


class RelayBackend(ABC):
    """
    Common interface for all relay hardware backends.
    All relay_idx arguments are 0-based integers.
    """

    @abstractmethod
    def connect(self) -> bool:
        """Establish connection to the hardware. Returns True on success."""

    @abstractmethod
    def disconnect(self):
        """Clean shutdown — turn everything off, release resources."""

    @abstractmethod
    def relay_on(self, relay_idx: int) -> bool:
        """Energise relay at 0-based index. Returns True on success."""

    @abstractmethod
    def relay_off(self, relay_idx: int) -> bool:
        """De-energise relay at 0-based index. Returns True on success."""

    @abstractmethod
    def read_relay(self, relay_idx: int) -> Optional[bool]:
        """
        Read current physical state (0-based index).
        Returns True (ON), False (OFF), or None on read failure.
        """

    @abstractmethod
    def all_off(self) -> bool:
        """Turn every relay OFF. Returns True if all writes succeeded."""

    @abstractmethod
    def health_check(self) -> bool:
        """
        Probe the hardware and confirm it is reachable and responsive.
        Returns True if healthy.
        """

    # ── Optional property (backends may override) ──────────────────────────
    @property
    def last_error(self) -> str:
        """Human-readable description of the most recent error."""
        return ""