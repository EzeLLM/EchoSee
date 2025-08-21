"""LeetCode MCP server exposing problem retrieval tools."""

from __future__ import annotations

import time

import requests
from mcp.server.fastmcp import Context, FastMCP

from agent_management.helpers.LeetCode.LeetCodeAPI import LeetCodeAPI
from mcp_servers.utils import write_audit

app = FastMCP("leetcode")
api = LeetCodeAPI()


@app.custom_route("/healthz", methods=["GET"])
async def healthz():
    return {"ok": True}


@app.tool()
async def leetcode_problem(slug: str, ctx: Context | None = None):
    start = time.time()
    rid = ctx.request_id if ctx else None
    try:
        info = api.retrieve(slug)
        took = int((time.time() - start) * 1000)
        write_audit({"request_id": rid, "tool": "leetcode_problem", "ok": True})
        return {"ok": True, "data": info, "meta": {"took_ms": took}}
    except Exception as exc:  # pragma: no cover
        took = int((time.time() - start) * 1000)
        write_audit({"request_id": rid, "tool": "leetcode_problem", "ok": False, "error": str(exc)})
        return {
            "ok": False,
            "error": {"code": "leetcode_error", "message": str(exc)},
            "meta": {"took_ms": took},
        }


@app.tool()
async def leetcode_daily(ctx: Context | None = None):
    start = time.time()
    rid = ctx.request_id if ctx else None
    try:
        query = """
        query questionOfToday {
            activeDailyCodingChallengeQuestion {
                question {
                    titleSlug
                }
            }
        }
        """
        resp = requests.post("https://leetcode.com/graphql", json={"query": query})
        resp.raise_for_status()
        slug = resp.json()["data"]["activeDailyCodingChallengeQuestion"]["question"]["titleSlug"]
        info = api.retrieve(slug)
        took = int((time.time() - start) * 1000)
        write_audit({"request_id": rid, "tool": "leetcode_daily", "ok": True})
        return {"ok": True, "data": info, "meta": {"took_ms": took}}
    except Exception as exc:  # pragma: no cover
        took = int((time.time() - start) * 1000)
        write_audit({"request_id": rid, "tool": "leetcode_daily", "ok": False, "error": str(exc)})
        return {
            "ok": False,
            "error": {"code": "leetcode_error", "message": str(exc)},
            "meta": {"took_ms": took},
        }


if __name__ == "__main__":  # pragma: no cover
    import uvicorn

    uvicorn.run(app.streamable_http_app(), host="0.0.0.0", port=7013)

