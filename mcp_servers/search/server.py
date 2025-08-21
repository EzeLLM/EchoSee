"""Search MCP server providing advanced multi-source search functionality."""

from __future__ import annotations

import os
import time
import asyncio
from typing import List, Optional
from dataclasses import dataclass

import numpy as np
from sklearn.cluster import KMeans
from tavily import TavilyClient
from mcp.server.fastmcp import Context, FastMCP
from mcp_servers.utils import write_audit, load_config

# Configure embedding provider
try:
    import openai
    openai.api_key = os.environ.get("OPENAI_API_KEY")
    EMBEDDINGS_AVAILABLE = True
except Exception:
    EMBEDDINGS_AVAILABLE = False

# Configure high performance LLM
try:
    from langchain_openai import ChatOpenAI
    HIGH_PERF_LLM = ChatOpenAI(
        model="gpt-4o",
        temperature=0.2,
        max_tokens=4096
    )
    BASE_LLM = ChatOpenAI(
        model="gpt-4o-mini",
        temperature=0.2,
        max_tokens=1024
    )
    LLM_AVAILABLE = True
except Exception:
    LLM_AVAILABLE = False

app = FastMCP("search")
config = load_config()

# Initialize Tavily client
tavily_client = TavilyClient() if os.environ.get("TAVILY_API_KEY") else None


@dataclass
class SearchConfig:
    max_queries: int = 5
    oversample_factor: int = 20
    embedding_model: str = "text-embedding-ada-002"
    citation_limit: int = 6
    mmr_lambda: float = 0.6


# Embedding cache to avoid repeated calls
embedding_cache: dict[str, list[float]] = {}


def cosine_similarity(a, b):
    """Calculate cosine similarity between two vectors."""
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))


async def embed_texts(texts: List[str], model: str = "text-embedding-ada-002") -> List[List[float]]:
    """Get embeddings for texts using cache."""
    if not EMBEDDINGS_AVAILABLE:
        raise Exception("OpenAI embeddings not available")
    
    uncached = [t for t in texts if t not in embedding_cache]
    
    if uncached:
        resp = await asyncio.to_thread(
            openai.embeddings.create,
            input=uncached,
            model=model
        )
        
        for text, rec in zip(uncached, resp.data):
            embedding_cache[text] = rec.embedding
    
    return [embedding_cache[t] for t in texts]


async def generate_diverse_queries(user_query: str, config: SearchConfig) -> List[str]:
    """Generate diverse search queries using clustering."""
    if not LLM_AVAILABLE:
        return [user_query]  # Fallback to original query
    
    pool_size = config.oversample_factor * config.max_queries
    
    system_prompt = """You are a search query generator. Generate diverse variations of the user's search query.
    Output ONLY the queries, one per line, no numbering or bullets.
    Vary perspectives, phrasings, specificity levels, and angles.
    Generate queries that would collectively give the fullest picture of the topic."""
    
    user_prompt = f"Generate {pool_size} variations of: '{user_query}'"
    
    response = await asyncio.to_thread(
        lambda: BASE_LLM.invoke([
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ])
    )
    
    pool = [line.strip() for line in response.content.splitlines() if line.strip()]
    deduped = list(dict.fromkeys(pool))[:pool_size]
    
    if not EMBEDDINGS_AVAILABLE or len(deduped) <= config.max_queries:
        return deduped[:config.max_queries]
    
    # Cluster queries
    vectors = await embed_texts(deduped)
    kmeans = KMeans(n_clusters=config.max_queries, random_state=42)
    labels = kmeans.fit_predict(vectors)
    
    clusters = {i: [] for i in range(config.max_queries)}
    cluster_vectors = {i: [] for i in range(config.max_queries)}
    
    for query, vector, label in zip(deduped, vectors, labels):
        clusters[label].append(query)
        cluster_vectors[label].append(vector)
    
    # Pick query closest to centroid from each cluster
    final_queries = []
    for cluster_id in range(config.max_queries):
        if not clusters[cluster_id]:
            continue
        
        centroid = kmeans.cluster_centers_[cluster_id]
        distances = [cosine_similarity(vec, centroid) for vec in cluster_vectors[cluster_id]]
        best_idx = np.argmax(distances)
        final_queries.append(clusters[cluster_id][best_idx])
    
    return final_queries


async def search_tavily(queries: List[str], recent: bool = False) -> List[dict]:
    """Execute Tavily searches in parallel."""
    if not tavily_client:
        return []
    
    async def fetch(q: str) -> dict:
        search_kwargs = {"query": q}
        if recent:
            search_kwargs["time_range"] = "month"
        return await asyncio.to_thread(tavily_client.search, **search_kwargs)
    
    tasks = [fetch(q) for q in queries]
    return await asyncio.gather(*tasks)


def filter_snippets_mmr(snippets: List[str], user_query: str, config: SearchConfig) -> List[str]:
    """Select diverse relevant snippets using Maximum Marginal Relevance."""
    if not snippets or not EMBEDDINGS_AVAILABLE:
        return snippets[:config.citation_limit]
    
    # Get embeddings synchronously in async context
    loop = asyncio.new_event_loop()
    vectors = loop.run_until_complete(embed_texts([user_query] + snippets))
    loop.close()
    
    q_vec = vectors[0]
    doc_vecs = vectors[1:]
    
    relevance_scores = [cosine_similarity(v, q_vec) for v in doc_vecs]
    
    k = min(config.citation_limit, len(snippets))
    selected_indices: List[int] = []
    
    for _ in range(k):
        best_idx, best_score = None, -float("inf")
        
        for idx in range(len(doc_vecs)):
            if idx in selected_indices:
                continue
            
            if selected_indices:
                max_sim = max(cosine_similarity(doc_vecs[idx], doc_vecs[j]) for j in selected_indices)
            else:
                max_sim = 0.0
            
            mmr_score = config.mmr_lambda * relevance_scores[idx] - (1 - config.mmr_lambda) * max_sim
            
            if mmr_score > best_score:
                best_score = mmr_score
                best_idx = idx
        
        if best_idx is None:
            break
        
        selected_indices.append(best_idx)
    
    return [snippets[i] for i in selected_indices]


@app.custom_route("/healthz", methods=["GET"])
async def healthz():
    return {"ok": True}


@app.tool()
async def simple_search(
    query: str,
    time_range: Optional[str] = None,
    max_results: int = 5,
    ctx: Context | None = None
):
    """Perform a simple web search using Tavily.
    
    Args:
        query: The search query
        time_range: Optional time filter ('day', 'week', 'month', 'year')
        max_results: Maximum number of results to return
        ctx: MCP context
        
    Returns:
        Search results with title, URL, and content
    """
    start = time.time()
    rid = ctx.request_id if ctx else None
    
    if not tavily_client:
        write_audit({"request_id": rid, "tool": "simple_search", "ok": False, "error": "search_unavailable"})
        return {
            "ok": False,
            "error": {"code": "unavailable", "message": "Search backend not configured"},
            "meta": {}
        }
    
    try:
        search_kwargs = {"query": query, "max_results": max_results}
        if time_range:
            search_kwargs["time_range"] = time_range
        
        response = await asyncio.to_thread(tavily_client.search, **search_kwargs)
        
        results = []
        for item in response.get("results", []):
            results.append({
                "title": item.get("title", ""),
                "url": item.get("url", ""),
                "content": item.get("content", "")
            })
        
        took = int((time.time() - start) * 1000)
        write_audit({"request_id": rid, "tool": "simple_search", "ok": True})
        return {"ok": True, "data": {"results": results}, "meta": {"took_ms": took}}
        
    except Exception as exc:
        took = int((time.time() - start) * 1000)
        write_audit({"request_id": rid, "tool": "simple_search", "ok": False, "error": str(exc)})
        return {
            "ok": False,
            "error": {"code": "search_error", "message": str(exc)},
            "meta": {"took_ms": took}
        }


@app.tool()
async def robust_search(
    query: str,
    max_queries: int = 3,
    recent: bool = False,
    ctx: Context | None = None
):
    """Perform an advanced multi-query search with diversified results and AI synthesis.
    
    This tool generates multiple search variations, collects diverse evidence,
    and synthesizes a comprehensive answer.
    
    Args:
        query: The user's question or search topic
        max_queries: Number of diverse search queries to generate (default: 3)
        recent: If True, restrict results to past month
        ctx: MCP context
        
    Returns:
        Synthesized answer with evidence and sources
    """
    start = time.time()
    rid = ctx.request_id if ctx else None
    
    if not all([tavily_client, LLM_AVAILABLE]):
        write_audit({"request_id": rid, "tool": "robust_search", "ok": False, "error": "dependencies_missing"})
        return {
            "ok": False,
            "error": {"code": "unavailable", "message": "Required services not available"},
            "meta": {}
        }
    
    try:
        config = SearchConfig(max_queries=max_queries)
        
        # Generate diverse queries
        queries = await generate_diverse_queries(query, config)
        
        # Collect search results
        raw_results = await search_tavily(queries, recent)
        
        # Extract snippets
        snippets = []
        sources = []
        for result_set in raw_results:
            for item in result_set.get("results", []):
                title = item.get("title", "").strip()
                content = item.get("content", "").strip()
                url = item.get("url", "")
                if title and content:
                    snippets.append(f"{title}: {content}")
                    sources.append({"title": title, "url": url})
        
        # Filter using MMR if embeddings available
        if EMBEDDINGS_AVAILABLE and len(snippets) > config.citation_limit:
            filtered_snippets = filter_snippets_mmr(snippets, query, config)
        else:
            filtered_snippets = snippets[:config.citation_limit]
        
        # Synthesize answer
        evidence_blocks = [f"[{i+1}] \"{snippet}\"" for i, snippet in enumerate(filtered_snippets)]
        synthesis_prompt = f"""You are a fact-checking assistant providing comprehensive answers.
User asked: "{query}"

Evidence from {len(filtered_snippets)} sources:
{chr(10).join(evidence_blocks)}

Create a well-structured answer that:
1. Directly addresses the user's question
2. Cites evidence using [number] format
3. Provides a balanced view of the topic
4. Mentions any conflicting information
5. Stays factual and objective"""

        response = await asyncio.to_thread(
            lambda: HIGH_PERF_LLM.invoke([{"role": "user", "content": synthesis_prompt}])
        )
        
        answer = response.content.strip()
        
        took = int((time.time() - start) * 1000)
        write_audit({"request_id": rid, "tool": "robust_search", "ok": True})
        return {
            "ok": True,
            "data": {
                "answer": answer,
                "sources": sources[:len(filtered_snippets)],
                "queries_used": queries
            },
            "meta": {"took_ms": took}
        }
        
    except Exception as exc:
        took = int((time.time() - start) * 1000)
        write_audit({"request_id": rid, "tool": "robust_search", "ok": False, "error": str(exc)})
        return {
            "ok": False,
            "error": {"code": "search_error", "message": str(exc)},
            "meta": {"took_ms": took}
        }


if __name__ == "__main__":  # pragma: no cover
    import uvicorn
    uvicorn.run(app.streamable_http_app(), host="0.0.0.0", port=7016)
