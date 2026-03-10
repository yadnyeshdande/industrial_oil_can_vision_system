"""
core/logging_setup.py
=====================
Centralised logging configuration.
Each process calls setup_process_logging() with its own name
to get a rotating-file + console log handler.
"""

from __future__ import annotations
import logging
import logging.handlers
import os
import sys
import traceback
from pathlib import Path
from typing import Optional


def setup_process_logging(
    process_name: str,
    log_dir: str = "logs",
    log_level: str = "INFO",
    max_bytes: int = 10_485_760,
    backup_count: int = 5,
    fmt: str = "%(asctime)s [%(levelname)s] [%(name)s] %(message)s",
    date_fmt: str = "%Y-%m-%d %H:%M:%S",
    also_console: bool = True,
) -> logging.Logger:
    """
    Configure the root logger for a specific process.
    Returns a named logger for this process.
    """
    log_path = Path(log_dir)
    log_path.mkdir(parents=True, exist_ok=True)

    log_file = log_path / f"{process_name}.log"
    level = getattr(logging, log_level.upper(), logging.INFO)

    formatter = logging.Formatter(fmt=fmt, datefmt=date_fmt)

    # Rotating file handler
    file_handler = logging.handlers.RotatingFileHandler(
        str(log_file),
        maxBytes=max_bytes,
        backupCount=backup_count,
        encoding="utf-8",
    )
    file_handler.setFormatter(formatter)
    file_handler.setLevel(level)

    handlers: list = [file_handler]

    if also_console:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(formatter)
        console_handler.setLevel(level)
        handlers.append(console_handler)

    # Configure root logger for this process
    root = logging.getLogger()
    root.setLevel(level)
    # Remove existing handlers to avoid duplicates when called multiple times
    root.handlers.clear()
    for h in handlers:
        root.addHandler(h)

    logger = logging.getLogger(process_name)
    logger.info("Logging initialised → %s (level=%s)", log_file, log_level)
    return logger


def setup_crash_handler(process_name: str, log_dir: str = "logs"):
    """Install a global unhandled-exception handler that writes crash dumps."""
    crash_log = Path(log_dir) / f"{process_name}_crash.log"

    def handler(exc_type, exc_value, exc_tb):
        with open(crash_log, "a", encoding="utf-8") as f:
            import time
            f.write(f"\n{'='*80}\n")
            f.write(f"CRASH DUMP [{process_name}] @ {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write("".join(traceback.format_exception(exc_type, exc_value, exc_tb)))
            f.write(f"{'='*80}\n")
        logging.critical("UNHANDLED EXCEPTION in %s", process_name, exc_info=(exc_type, exc_value, exc_tb))
        sys.__excepthook__(exc_type, exc_value, exc_tb)

    sys.excepthook = handler


def get_logger(name: str) -> logging.Logger:
    return logging.getLogger(name)
