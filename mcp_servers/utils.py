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
MCP_SERVERS_PATH = ROOT / "mcp_servers" / "servers.json"


def load_config() -> Dict[str, Any]:
    """Load project configuration."""
    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_mcp_server_configs() -> Dict[str, Any]:
    """Load MCP server definitions from JSON."""
    with open(MCP_SERVERS_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


def write_audit(entry: Dict[str, Any]) -> None:
    """Append an entry to the audit log."""
    entry.setdefault("timestamp", time.time())
    with open(AUDIT_LOG, "a", encoding="utf-8") as f:
        f.write(json.dumps(entry) + "\n")


__all__ = [
    "load_config",
    "load_mcp_server_configs",
    "write_audit",
    "CONFIG_PATH",
    "MCP_SERVERS_PATH",
]

