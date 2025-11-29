from __future__ import annotations

import os
import asyncio
import logging
from dataclasses import dataclass
from typing import List

import numpy as np
import openai
from sklearn.cluster import KMeans
from core.clients import clients
from langchain_core.messages import HumanMessage, SystemMessage
from utils.utils import llm, high_performance_llm  # noqa: E402

logger = logging.getLogger(__name__)


@dataclass
class SearchConfig:
    max_queries: int = 5
    oversample_factor: int = 20        # how many candidates per final query
    embedding_model: str = "text-embedding-ada-002"
    citation_limit: int = 6            # number of evidence snippets to include in summary
    mmr_lambda: float = 0.6            # trade-off between relevance (1) and novelty (0). Lower means novelty gets more weight.


class SearchClient:
    """Diversified, embedding-driven search agent built on Tavily and LangChain."""

    def __init__(
        self,
        config: SearchConfig = SearchConfig(),
    ):
        if config.max_queries < 1:
            raise ValueError("max_queries must be at least 1")

        self.config = config
        self.max_queries = config.max_queries
        self.base_llm = llm
        self.hp_llm = high_performance_llm
        self.tavily = clients.tavily
        openai.api_key = os.environ.get("OPENAI_API_KEY")  # Still needed for openai.embeddings

        # Local cache to avoid repeated embedding calls for the same text.
        # Key   : str (raw text)
        # Value : List[float] (embedding vector)
        self._embedding_cache: dict[str, list[float]] = {}

    def run(self, user_query: str, *, max_queries: int | None = None, recent: bool = False) -> str:
        """Execute the full diversified search pipeline and return the answer."""
        if max_queries is not None:
            self.max_queries = max_queries

        # Pre-compute and cache the user-query embedding so later steps reuse it.
        self._embed([user_query])

        queries = self._generate_queries(user_query)
        logger.debug(f"Generated queries: {queries}")
        logger.debug("--------------------------------")
        raw_results = asyncio.run(self._collect_search_results(queries, recent))
        # print(raw_results)
        # print("--------------------------------")
        snippets = self._extract_snippets(raw_results)
        # print(snippets)
        # print("--------------------------------")
        filtered = self._filter_snippets(snippets, user_query)
        # print(filtered)
        # print("--------------------------------")
        return self._summarise(user_query, filtered).strip()

    def _generate_queries(self, user_query: str) -> List[str]:
        """Use embeddings + clustering to create diverse search queries, selecting the most representative query from each cluster."""
        pool_size = self.config.oversample_factor * self.max_queries
        system_prompt = f"""You are **Deep Research Assistant**, an LLM whose sole task is to take a **single user query** and expand it into a *diverse* list of Google-style search queries.  
Your output must be a **plain list**—one query per line, no numbering, no bullet symbols, nothing else.

---

### 1  Understand the user's need first
1. *Reflect*: What does the user really want to learn or resolve? Break the intent into its core sub-topics or possible angles.  
2. *Mind the nuance*: If the user expresses doubt, controversy, or comparison, note each tension point (e.g. benefit vs harm, official stance vs community experience).

---

### 2  Generate variations deliberately
Produce **N = max_queries × 2** queries unless otherwise instructed (e.g. max_queries is supplied by the caller).  
While writing each query, consciously vary at least one of:

| Variation lever          | Examples (for "Why does Tesla recommend charging LFP batteries to 100 %?")                           |
|--------------------------|-------------------------------------------------------------------------------------------------------|
| **Perspective / sentiment** | *"Tesla says charge LFP to 100 %, is that harmful?"*                                                |
| **Question ↔ statement**    | *"Harm to LFP batteries when charged to full"*                                                      |
| **Specific ↔ broad**        | *"LFP battery full-charge cycle life"* vs *"EV battery charging best practices"*                    |
| **Synonyms / paraphrases**  | *"lithium iron phosphate over-charge effects"*                                                      |
| **Stakeholder focus**       | *"Tesla owner forum advice on LFP charging"*                                                        |
| **Cause ↔ effect**          | *"Does 100 % charging improve LFP BMS calibration?"*                                                |

---

### 3  Balance focus and breadth
* Roughly **20 %** of queries should track the user's wording almost verbatim.  
  *Example:*  
  - *"Tesla recommends charging LFP to full—why?"*  
* The rest should explore wider or adjacent angles (chemistry, longevity studies, manufacturer guidelines, user anecdotes, etc.).

---

### 4  Quality checklist before you output
- **Same language** as the original query.  
- **One question per line.** If the user asks multiple questions, create separate clusters of variations for each.  
- **No near-duplicates.** Ensure wording, scope, or sentiment truly differs.  
- **Search-engine real.** Each line must read like something a person would actually type into Google.  
- **Coverage test:** Ask yourself, *If I ran these queries, would the combined results give me the fullest, most balanced picture possible?* If not, revise.

---

### 5  Output format
```text
query one
query two
query three

(Nothing before or after the list.)

Quick example
User: "Why does Tesla recommend charging LFP to full even though we know that's bad for batteries?"
Possible output (first 6 of ≈8):
Tesla recommends charging LFP to 100 percent why
Tesla recommends charging LFP to full but isn't that harmful
LFP battery longevity when charged to 100 percent
Tesla LFP battery full charge calibration benefits
Does full charging degrade lithium iron phosphate batteries
Tesla owner forum experiences with daily 100 percent LFP charge
Follow these rules exactly every time you are invoked.
"""
        prompt = f"""Generate {pool_size} variations of the following search engine query while obeying the instructions.
        User query: '{user_query}'
        """
        resp = self.hp_llm.invoke([SystemMessage(content=system_prompt),HumanMessage(content=prompt)])
        pool = [line.strip() for line in resp.content.splitlines() if line.strip()]
        logger.debug("The original queries are:")
        logger.debug(f"{pool}")
        logger.debug("--------------------------------")
        # dedupe while preserving order
        deduped = list(dict.fromkeys(pool))

        # embed candidates (uses cache under the hood)
        vectors = self._embed(deduped)

        # cluster into max_queries groups
        kmeans = KMeans(n_clusters=self.max_queries, random_state=42)
        labels = kmeans.fit_predict(vectors)
        clusters = {i: [] for i in range(self.max_queries)}
        cluster_vectors = {i: [] for i in range(self.max_queries)}
        
        for query, vector, label in zip(deduped, vectors, labels):
            clusters[label].append(query)
            cluster_vectors[label].append(vector)

        # pick the query closest to centroid from each cluster for best representation
        final_queries = []
        for cluster_id in range(self.max_queries):
            cluster_queries = clusters[cluster_id]
            cluster_vecs = cluster_vectors[cluster_id]
            
            if not cluster_queries:
                continue
                
            # Cluster centroid from KMeans
            cluster_centroid = kmeans.cluster_centers_[cluster_id]
            
            # Find closest to centroid
            centroid_distances = [cosine_similarity(vec, cluster_centroid) for vec in cluster_vecs]
            closest_to_centroid_idx = np.argmax(centroid_distances)
            
            final_queries.append(cluster_queries[closest_to_centroid_idx])

        logger.debug("The final queries are (using centroid method):")
        logger.debug(f"{final_queries}")
        logger.debug("--------------------------------")
        return final_queries

    async def _collect_search_results(self, queries: List[str], recent: bool) -> List[dict]:
        """Run Tavily searches in parallel and return results."""
        cache: dict[str, dict] = {}

        async def fetch(q: str) -> dict:
            if q in cache:
                return cache[q]
            # TavilyClient.search is synchronous; run it in a thread to avoid blocking the event loop
            search_kwargs = {"query": q, "return_raw_results": True}
            if recent:
                # filter to past month for recency
                search_kwargs["time_range"] = "month"

            res = await asyncio.to_thread(self.tavily.search, **search_kwargs)
            cache[q] = res
            return res

        tasks = [fetch(q) for q in queries]
        return await asyncio.gather(*tasks)

    def _extract_snippets(self, raw_results: List[dict]) -> List[str]:
        """Flatten Tavily results into text snippets."""
        snippets: List[str] = []
        for block in raw_results:
            for item in block.get("results", []):
                title = item.get("title", "").strip()
                content = item.get("content", "").strip()
                snippets.append(f"{title}: {content}")
        return snippets

    def _filter_snippets(self, snippets: List[str], user_query: str) -> List[str]:
        """Select a small set of diverse, high-relevance snippets via Max Marginal
        Relevance (MMR).

        MMR greedily balances two forces:
            relevance  – similarity to the user query
            novelty    – dissimilarity to snippets already selected

        The trade-off is controlled by ``self.config.mmr_lambda``.
        """

        if not snippets:
            return []

        # Embed user query + all snippets in one batch (cached query embedding makes this cheap)
        vectors = self._embed([user_query] + snippets)
        q_vec = vectors[0]
        doc_vecs = vectors[1:]

        # ── 2. Pre-compute relevance scores (similarity to the query)
        relevance_scores = [cosine_similarity(v, q_vec) for v in doc_vecs]

        λ = getattr(self.config, "mmr_lambda", 0.6)
        k = min(self.config.citation_limit, len(snippets))

        selected_indices: List[int] = []
        for _ in range(k):
            best_idx, best_score = None, -float("inf")

            for idx, vec in enumerate(doc_vecs):
                if idx in selected_indices:
                    continue  # already chosen

                # novelty term: max similarity to anything already selected
                if selected_indices:
                    max_sim_to_selected = max(
                        cosine_similarity(vec, doc_vecs[j]) for j in selected_indices
                    )
                else:
                    max_sim_to_selected = 0.0

                mmr_score = λ * relevance_scores[idx] - (1 - λ) * max_sim_to_selected

                if mmr_score > best_score:
                    best_score = mmr_score
                    best_idx = idx

            if best_idx is None:
                break  # no candidates left

            selected_indices.append(best_idx)

        return [snippets[i] for i in selected_indices]

    def _summarise(self, user_query: str, evidence: List[str]) -> str:
        """Use hp_llm to write a concise, fact-checked answer."""
        # build numbered citations
        blocks = [f"[{i+1}] \"{snippet}\"" for i, snippet in enumerate(evidence)]
        prompt = (
            f"You are a concise, fact-checking assistant.\n"
            f"User asked: \"{user_query}\"\n\n"
            f"Below are {len(evidence)} vetted snippets:\n\n"
            + "\n\n".join(blocks)
            + "\n\nTask: Using only the above, and if the request does not require recent information use what you know about the topic too."
            " The answer should be in the same language as the user query."
            " The answer should be in report format, mention links to the sources of the information (not citations with numbers, but real links), justify the facts with citations, and use the evidence to answer the user query, do not make up information."
            " The report should be progressive, meaning you should start with the most general information and then go into more specific details. Also, do not answer as 'there is no ...' but rather give all the information that an asker would want to know about the topic. You do not answer but inform!"

        )
        return self.hp_llm.invoke([HumanMessage(content=prompt)]).content

    # ── Embedding helper -------------------------------------------------
    def _embed(self, texts: List[str]) -> List[List[float]]:  # noqa: D401
        """Return embeddings for *texts*, using an internal cache to avoid
        duplicate calls to the OpenAI embeddings endpoint."""

        # Split input into cached and uncached texts
        uncached = [t for t in texts if t not in self._embedding_cache]

        if uncached:
            # Batch request for all uncached strings
            resp = openai.embeddings.create(
                input=uncached,
                model=self.config.embedding_model,
            )

            for text, rec in zip(uncached, resp.data):
                self._embedding_cache[text] = rec.embedding

        # Return embeddings in the same order as *texts*
        return [self._embedding_cache[t] for t in texts]


def search_agent(
    user_query: str,
    max_queries: int = 3,
    *,
    recent: bool = False,
    mmr_lambda: float | None = None,
) -> str:
    """Convenience wrapper.

    Parameters
    ----------
    user_query : str
        The search question.
    max_queries : int, default 3
        How many distinct search engine queries to generate.
    recent : bool, default False
        Restrict Tavily results to the past month.
    mmr_lambda : float, optional
        Weight for relevance vs. novelty in MMR. If None, uses the default
        from ``SearchConfig`` (currently 0.6).
    """

    cfg_kwargs = {"max_queries": max_queries}
    if mmr_lambda is not None:
        cfg_kwargs["mmr_lambda"] = mmr_lambda

    cfg = SearchConfig(**cfg_kwargs)
    return SearchClient(config=cfg).run(user_query, recent=recent)


# -- Utility: cosine similarity --
from numpy import dot
from numpy.linalg import norm

def cosine_similarity(a, b):
    return dot(a, b) / (norm(a) * norm(b))

if __name__ == "__main__":
    print(search_agent("why does tesla recommend chargin lfp to full even though we know thats bad for batteries?", max_queries=4,recent=True))