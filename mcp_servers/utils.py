"""Shared utilities for MCP servers."""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Dict

import yaml

# Project root
ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / "config.yml"
LOG_DIR = ROOT / "logs"
LOG_DIR.mkdir(exist_ok=True)
AUDIT_LOG = LOG_DIR / "audit.jsonl"


def load_config() -> Dict[str, Any]:
    """Load project configuration."""
    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def write_audit(entry: Dict[str, Any]) -> None:
    """Append an entry to the audit log."""
    entry.setdefault("timestamp", time.time())
    with open(AUDIT_LOG, "a", encoding="utf-8") as f:
        f.write(json.dumps(entry) + "\n")


__all__ = ["load_config", "write_audit", "CONFIG_PATH"]

