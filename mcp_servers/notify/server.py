"""Notify MCP server wrapping Echonoti."""

from __future__ import annotations

import time
from collections import deque

from fastapi import HTTPException
from mcp.server.fastmcp import Context, FastMCP

from .lib.EchonotiWrapper import send_notification
from mcp_servers.utils import load_config, write_audit

config = load_config()
policy = config.get("mcp", {}).get("policy", {})
ALLOWED_CHANNELS = set(policy.get("notify_allow_channels", []))
RATE_LIMIT = policy.get("rate_limits", {}).get("notify_send", {}).get("per_minute", 0)

app = FastMCP("notify")
call_times: deque[float] = deque()


@app.custom_route("/healthz", methods=["GET"])
async def healthz(request):
    return {"ok": True}


@app.tool()
async def notify_channels(ctx: Context | None = None):
    rid = ctx.request_id if ctx else None
    write_audit({"request_id": rid, "tool": "notify_channels", "ok": True})
    return {"ok": True, "data": list(ALLOWED_CHANNELS), "meta": {}}


@app.tool()
async def notify_send(
    channel: str,
    message: str,
    priority: str = "normal",
    ctx: Context | None = None,
):
    start = time.time()
    rid = ctx.request_id if ctx else None

    if channel not in ALLOWED_CHANNELS:
        write_audit({"request_id": rid, "tool": "notify_send", "ok": False, "error": "channel_not_allowlisted"})
        raise HTTPException(403, "Channel not allowlisted")

    if RATE_LIMIT:
        now = time.time()
        while call_times and call_times[0] < now - 60:
            call_times.popleft()
        if len(call_times) >= RATE_LIMIT:
            write_audit({"request_id": rid, "tool": "notify_send", "ok": False, "error": "rate_limited"})
            raise HTTPException(429, "Rate limit exceeded")
        call_times.append(now)

    try:
        result = send_notification(headline=channel, summary=message, content=message, notification_type=priority)
        took = int((time.time() - start) * 1000)
        write_audit({"request_id": rid, "tool": "notify_send", "ok": True, "meta": {"took_ms": took}})
        return {"ok": True, "data": result, "meta": {"took_ms": took}}
    except Exception as exc:  # pragma: no cover
        took = int((time.time() - start) * 1000)
        write_audit({"request_id": rid, "tool": "notify_send", "ok": False, "error": str(exc)})
        return {
            "ok": False,
            "error": {"code": "notify_error", "message": str(exc)},
            "meta": {"took_ms": took},
        }


if __name__ == "__main__":  # pragma: no cover
    import uvicorn

    uvicorn.run(app.streamable_http_app(), host="0.0.0.0", port=7011)

