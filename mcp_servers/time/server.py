"""Time MCP server providing date and time utilities."""

from __future__ import annotations

from datetime import datetime
import pytz
from typing import Optional

from mcp.server.fastmcp import Context, FastMCP
from mcp_servers.utils import write_audit

app = FastMCP("time")


@app.custom_route("/healthz", methods=["GET"])
async def healthz():
    return {"ok": True}


@app.tool()
async def get_current_time(timezone: Optional[str] = None, ctx: Context | None = None):
    """Get the current time in the specified timezone or system timezone.
    
    Args:
        timezone: Timezone name (e.g., 'UTC', 'America/New_York', 'Europe/London').
                 If not provided, uses system local time.
        ctx: MCP context
        
    Returns:
        Formatted time string in YYYY-MM-DD HH:MM:SS format
    """
    rid = ctx.request_id if ctx else None
    try:
        if timezone:
            tz = pytz.timezone(timezone)
            current_time = datetime.now(tz).strftime("%Y-%m-%d %H:%M:%S")
        else:
            current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        write_audit({"request_id": rid, "tool": "get_current_time", "ok": True})
        return {"ok": True, "data": {"time": current_time, "timezone": timezone or "local"}, "meta": {}}
    except Exception as exc:
        write_audit({"request_id": rid, "tool": "get_current_time", "ok": False, "error": str(exc)})
        return {
            "ok": False,
            "error": {"code": "time_error", "message": str(exc)},
            "meta": {}
        }


@app.tool()
async def get_current_date(timezone: Optional[str] = None, ctx: Context | None = None):
    """Get the current date in the specified timezone or system timezone.
    
    Args:
        timezone: Timezone name (e.g., 'UTC', 'America/New_York', 'Europe/London').
                 If not provided, uses system local time.
        ctx: MCP context
        
    Returns:
        Formatted date string in YYYY-MM-DD format
    """
    rid = ctx.request_id if ctx else None
    try:
        if timezone:
            tz = pytz.timezone(timezone)
            current_date = datetime.now(tz).strftime("%Y-%m-%d")
        else:
            current_date = datetime.now().strftime("%Y-%m-%d")
        
        write_audit({"request_id": rid, "tool": "get_current_date", "ok": True})
        return {"ok": True, "data": {"date": current_date, "timezone": timezone or "local"}, "meta": {}}
    except Exception as exc:
        write_audit({"request_id": rid, "tool": "get_current_date", "ok": False, "error": str(exc)})
        return {
            "ok": False,
            "error": {"code": "date_error", "message": str(exc)},
            "meta": {}
        }


@app.tool()
async def get_datetime_info(timezone: Optional[str] = None, ctx: Context | None = None):
    """Get comprehensive datetime information including date, time, day of week, and timezone info.
    
    Args:
        timezone: Timezone name (e.g., 'UTC', 'America/New_York', 'Europe/London').
                 If not provided, uses system local time.
        ctx: MCP context
        
    Returns:
        Dictionary with complete datetime information
    """
    rid = ctx.request_id if ctx else None
    try:
        if timezone:
            tz = pytz.timezone(timezone)
            now = datetime.now(tz)
        else:
            now = datetime.now()
        
        info = {
            "date": now.strftime("%Y-%m-%d"),
            "time": now.strftime("%H:%M:%S"),
            "datetime": now.strftime("%Y-%m-%d %H:%M:%S"),
            "day_of_week": now.strftime("%A"),
            "day_of_month": now.day,
            "month": now.strftime("%B"),
            "year": now.year,
            "iso_format": now.isoformat(),
            "timezone": timezone or "local"
        }
        
        write_audit({"request_id": rid, "tool": "get_datetime_info", "ok": True})
        return {"ok": True, "data": info, "meta": {}}
    except Exception as exc:
        write_audit({"request_id": rid, "tool": "get_datetime_info", "ok": False, "error": str(exc)})
        return {
            "ok": False,
            "error": {"code": "datetime_error", "message": str(exc)},
            "meta": {}
        }


if __name__ == "__main__":  # pragma: no cover
    import uvicorn
    uvicorn.run(app.streamable_http_app(), host="0.0.0.0", port=7014)
