"""Scheduler MCP server orchestrating jobs across tools."""

from __future__ import annotations

import asyncio
import uuid
from datetime import datetime, timedelta

from apscheduler.schedulers.asyncio import AsyncIOScheduler
from apscheduler.triggers.cron import CronTrigger
from mcp.server.fastmcp import Context, FastMCP

from mcp.client.session import ClientSession
from mcp.client.streamable_http import streamablehttp_client
from mcp_servers.utils import load_config, write_audit

config = load_config()
servers = config.get("mcp", {}).get("servers", {})

app = FastMCP("scheduler")
scheduler = AsyncIOScheduler()


def ensure_scheduler() -> None:
    if not scheduler.running:
        scheduler.start()

# Map tool name -> server URL
tool_registry: dict[str, str] = {}


async def discover_tools() -> None:
    for info in servers.values():
        url = info.get("url")
        try:
            async with streamablehttp_client(url) as (r, w, _):
                session = ClientSession(r, w)
                await session.initialize()
                res = await session.list_tools()
                for t in res.tools:
                    tool_registry[t.name] = url
        except Exception:  # pragma: no cover - discovery failure
            continue


asyncio.get_event_loop().run_until_complete(discover_tools())


async def invoke_tool(url: str, name: str, args: dict) -> None:
    async with streamablehttp_client(url) as (r, w, _):
        session = ClientSession(r, w)
        await session.initialize()
        await session.call_tool(name, args)


async def run_job(job_id: str, tool: str, args: dict) -> None:
    url = tool_registry.get(tool)
    write_audit({"request_id": job_id, "tool": tool, "ok": True, "event": "job_start"})
    if url:
        await invoke_tool(url, tool, args)
    notify_url = servers.get("notify", {}).get("url")
    if notify_url:
        await invoke_tool(notify_url, "notify_send", {"channel": "me", "message": f"Job {job_id} finished"})
    write_audit({"request_id": job_id, "tool": tool, "ok": True, "event": "job_finish"})


@app.custom_route("/healthz", methods=["GET"])
async def healthz():
    return {"ok": True}


@app.tool()
async def schedule_in(delay_s: int, tool: str, args: dict, ctx: Context | None = None):
    ensure_scheduler()
    job_id = str(uuid.uuid4())
    scheduler.add_job(
        lambda: asyncio.create_task(run_job(job_id, tool, args)),
        trigger="date",
        run_date=datetime.utcnow() + timedelta(seconds=delay_s),
        id=job_id,
    )
    write_audit({"request_id": ctx.request_id if ctx else None, "tool": "schedule_in", "ok": True, "job_id": job_id})
    return {"ok": True, "data": {"job_id": job_id}, "meta": {}}


@app.tool()
async def schedule_cron(expr: str, tool: str, args: dict, ctx: Context | None = None):
    ensure_scheduler()
    job_id = str(uuid.uuid4())
    trigger = CronTrigger.from_crontab(expr)
    scheduler.add_job(
        lambda: asyncio.create_task(run_job(job_id, tool, args)),
        trigger=trigger,
        id=job_id,
    )
    write_audit({"request_id": ctx.request_id if ctx else None, "tool": "schedule_cron", "ok": True, "job_id": job_id})
    return {"ok": True, "data": {"job_id": job_id}, "meta": {}}


@app.tool()
async def list_jobs(ctx: Context | None = None):
    ensure_scheduler()
    jobs = [
        {"job_id": j.id, "next_run": j.next_run_time.isoformat() if j.next_run_time else None}
        for j in scheduler.get_jobs()
    ]
    write_audit({"request_id": ctx.request_id if ctx else None, "tool": "list_jobs", "ok": True})
    return {"ok": True, "data": jobs, "meta": {}}


@app.tool()
async def cancel(job_id: str, ctx: Context | None = None):
    ensure_scheduler()
    scheduler.remove_job(job_id)
    write_audit({"request_id": ctx.request_id if ctx else None, "tool": "cancel", "ok": True, "job_id": job_id})
    return {"ok": True, "data": {"job_id": job_id}, "meta": {}}


if __name__ == "__main__":  # pragma: no cover
    import uvicorn

    uvicorn.run(app.streamable_http_app(), host="0.0.0.0", port=7012)

