"""Web MCP server providing simple fetch and search tools."""

from __future__ import annotations

import time
from urllib.parse import urlparse

import httpx
from fastapi import HTTPException
from mcp.server.fastmcp import Context, FastMCP
try:  # pragma: no cover - optional dependency
    from tavily import TavilyClient
except Exception:  # pragma: no cover
    TavilyClient = None  # type: ignore

from mcp_servers.utils import load_config, write_audit

config = load_config()
policy = config.get("mcp", {}).get("policy", {})
ALLOWED_DOMAINS = set(policy.get("web_allow_domains", []))

app = FastMCP("web")
tavily_client = TavilyClient() if TavilyClient else None


@app.custom_route("/healthz", methods=["GET"])
async def healthz():
    return {"ok": True}


@app.tool()
async def web_fetch_json(
    url: str,
    params: dict | None = None,
    timeout_s: int = 10,
    ctx: Context | None = None,
):
    """Fetch JSON from allowlisted domains."""
    start = time.time()
    rid = ctx.request_id if ctx else None
    domain = urlparse(url).netloc
    if domain not in ALLOWED_DOMAINS:
        write_audit({"request_id": rid, "tool": "web_fetch_json", "ok": False, "error": "domain_not_allowlisted"})
        raise HTTPException(403, "Domain not allowlisted")

    try:
        async with httpx.AsyncClient(timeout=timeout_s) as client:
            resp = await client.get(url, params=params)
            resp.raise_for_status()
            data = resp.json()
        took = int((time.time() - start) * 1000)
        write_audit({"request_id": rid, "tool": "web_fetch_json", "ok": True, "meta": {"took_ms": took}})
        return {"ok": True, "data": data, "meta": {"took_ms": took}}
    except Exception as exc:  # pragma: no cover - network errors
        took = int((time.time() - start) * 1000)
        write_audit({"request_id": rid, "tool": "web_fetch_json", "ok": False, "error": str(exc)})
        return {
            "ok": False,
            "error": {"code": "fetch_error", "message": str(exc)},
            "meta": {"took_ms": took},
        }


@app.tool()
async def web_search(query: str, limit: int = 5, ctx: Context | None = None):
    """Search the web using Tavily."""
    start = time.time()
    rid = ctx.request_id if ctx else None
    if not tavily_client:
        write_audit({"request_id": rid, "tool": "web_search", "ok": False, "error": "unavailable"})
        raise HTTPException(503, "search backend unavailable")
    try:
        results = tavily_client.search(query, max_results=limit)
        took = int((time.time() - start) * 1000)
        write_audit({"request_id": rid, "tool": "web_search", "ok": True, "meta": {"took_ms": took}})
        return {"ok": True, "data": results, "meta": {"took_ms": took}}
    except Exception as exc:  # pragma: no cover
        took = int((time.time() - start) * 1000)
        write_audit({"request_id": rid, "tool": "web_search", "ok": False, "error": str(exc)})
        return {
            "ok": False,
            "error": {"code": "search_error", "message": str(exc)},
            "meta": {"took_ms": took},
        }


if __name__ == "__main__":  # pragma: no cover
    import uvicorn

    uvicorn.run(app.streamable_http_app(), host="0.0.0.0", port=7010)

