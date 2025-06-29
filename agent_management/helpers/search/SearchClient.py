from __future__ import annotations

"""SearchClient
~~~~~~~~~~~~~~~~
A lightweight fact-checking agent that expands a user query into several
diversified search queries, runs them through Tavily, then distills the
raw snippets into a concise answer using a high-performance LLM.

Intended usage:
    >>> from agent_management.helpers.search.SearchClient import search_agent
    >>> answer = search_agent("origin of the universe", max_queries=4)

This helper keeps prompts minimal so that further prompt-engineering can
be performed by the caller if needed.
"""

import os
import random
from typing import List

from tavily import TavilyClient
from langchain_core.messages import HumanMessage

# Local util objects (normal and high-performance LLMs)
from utils.utils import llm, high_performance_llm  # noqa: E402


class SearchClient:
    """Diversified web-search agent built on Tavily and LangChain chat models."""

    def __init__(self, max_queries: int = 3):
        if max_queries < 1:
            raise ValueError("max_queries must be at least 1")

        self.max_queries = max_queries
        self.base_llm = llm  # normal LLM for query generation
        self.hp_llm = high_performance_llm  # high-performance LLM for synthesis
        self.tavily = TavilyClient(os.environ["TAVILY_API_KEY"])

    # ---------------------------------------------------------------------
    # Public API
    # ---------------------------------------------------------------------
    def run(self, user_query: str, *, max_queries: int | None = None) -> str:
        """Execute the full diversified search pipeline and return the answer."""
        if max_queries is not None:
            self.max_queries = max_queries

        diversified = self._generate_queries(user_query)
        print(diversified)
        print("--------------------------------")
        raw_evidence = self._collect_search_results(diversified)
        print(raw_evidence)
        print("--------------------------------")
        answer = self._summarise(user_query, raw_evidence)
        return answer.strip()

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def _generate_queries(self, user_query: str) -> List[str]:
        """Use *base_llm* to create diversified search queries."""
        target_count = self.max_queries * 2  # generate double

        prompt = (
            "Generate "
            f"{target_count} distinct web-search queries that could help answer "
            f"the following user request:\n\n'{user_query}'.\n\n"
            "Return each query on a separate line without numbering, additional "
            "comments, or quotation marks."
        )

        llm_response = self.base_llm.invoke([HumanMessage(content=prompt)])
        lines = [q.strip() for q in llm_response.content.split("\n") if q.strip()]

        # Remove duplicates while preserving order
        deduped: List[str] = []
        seen = set()
        for q in lines:
            if q not in seen:
                deduped.append(q)
                seen.add(q)

        # Randomly choose up to *max_queries* queries (without replacement)
        chosen = (
            random.sample(deduped, self.max_queries)
            if len(deduped) > self.max_queries
            else deduped
        )
        return chosen

    def _collect_search_results(self, queries: List[str]) -> str:
        """Run Tavily searches and concatenate raw results into one string."""
        snippets: List[str] = []

        for q in queries:
            response = self.tavily.search(query=q,return_raw_results=True)
            results = response.get("results", [])

            block_lines = [f"## Results for query: {q}"]
            for item in results:
                block_lines.append(
                    f"Title: {item.get('title')}\nURL: {item.get('url')}\nContent: {item.get('content')}"
                )
            snippets.append("\n".join(block_lines))

        return "\n\n".join(snippets)

    def _summarise(self, user_query: str, evidence: str) -> str:
        """Use *hp_llm* to write a short, fact-checked answer."""
        prompt = (
            f"You are a concise fact-checking assistant.\n\nUser request: '{user_query}'\n\n"
            "Your task: Based solely on the web search excerpts below, craft a short, "
            "accurate answer. Cite the most relevant URLs inline (e.g., [1]). "
            "If the information is insufficient, reply that you are not sure.\n\n"
            f"Web excerpts:\n{evidence}"
        )

        response = self.hp_llm.invoke([HumanMessage(content=prompt)])
        return response.content


# ---------------------------------------------------------------------------
# Convenience function for easy import into @tool wrappers or other modules
# ---------------------------------------------------------------------------

def search_agent(user_query: str, max_queries: int = 3) -> str:  # noqa: D401
    """Simple wrapper around `SearchClient.run` for one-off calls."""
    return SearchClient(max_queries=max_queries).run(user_query)

if __name__ == "__main__":
    print(search_agent("who is the current leader of the christian world?", max_queries=4))