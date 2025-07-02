from __future__ import annotations

import os
import asyncio
from dataclasses import dataclass
from typing import List

import numpy as np
import openai
from sklearn.cluster import KMeans
from tavily import TavilyClient
from langchain_core.messages import HumanMessage, SystemMessage
from utils.utils import llm, high_performance_llm  # noqa: E402


@dataclass
class SearchConfig:
    max_queries: int = 5
    oversample_factor: int = 20        # how many candidates per final query
    embedding_model: str = "text-embedding-ada-002"
    citation_limit: int = 6            # number of evidence snippets to include in summary


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
        self.tavily = TavilyClient(os.environ["TAVILY_API_KEY"])
        openai.api_key = os.environ.get("OPENAI_API_KEY")

    def run(self, user_query: str, *, max_queries: int | None = None, recent: bool = False) -> str:
        """Execute the full diversified search pipeline and return the answer."""
        if max_queries is not None:
            self.max_queries = max_queries

        queries = self._generate_queries(user_query)
        print(queries)
        print("--------------------------------")
        raw_results = asyncio.run(self._collect_search_results(queries, recent))
        exit()
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
        system_prompt = f"""
        You are a deep research assistant, you are given a user query and you will generate a list of google search queries that will satisfy the rules and guidelines below.
        The variations must be generated with the following flow:
        - Think about what the user query is asking, make sure you understand the user query and what the user is asking for.
        - Think about the way variations should be generated.
        - Think on how the generated variations should be diverse.
        - Think on how to make those variations diverse, for example, if the user is asking a doubtfull question, the variations should be diverse in a way that they cover different aspects of the doubts, or if the user is asking a question about a specific topic, the variations should be diverse in a way that they cover different aspects of the topic.
        - Ask youself: Are these the querries that will give me the most diverse results to satisfy the user? If not improve the generated variations.
        Verify the following points for the generated variations:
        - The variations must be in the same language as the user query.
        - The variations must be real search engine querries and should cover different aspects of the topic, or topics.
        - The variations must be a pure list, just querries where each querry is a line, no numerations no bullet points etc.
        - If they user querry includes multiple questions, then generate queries for each question, do not include multiple questions in the same query.
        - The variations must be diverse, they must not be similar to each other, as different as possible.
        - Even though they must be diverse, around 20% of them must be fully focused on the user query, and some of them must be more general.
        """
        resp = self.hp_llm.invoke([SystemMessage(content=system_prompt),HumanMessage(content=prompt)])
        pool = [line.strip() for line in resp.content.splitlines() if line.strip()]
        print("The original queries are:")
        print(pool)
        print("--------------------------------")
        # dedupe while preserving order
        deduped = list(dict.fromkeys(pool))

        # embed candidates
        resp_embed = openai.embeddings.create(
            input=deduped,
            model=self.config.embedding_model
        )
        vectors = [item.embedding for item in resp_embed.data]

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

        print("The final queries are (using centroid method):")
        print(final_queries)
        print("--------------------------------")
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
        """
        Embed & score snippets for relevance + novelty.

        Relevance: how close the snippet is to the user query.
        Novelty: how different it is from already selected snippets.
        Combined scoring prevents redundancy and ensures diverse, on-point evidence.
        """
        # embed user query
        q_resp = openai.embeddings.create(
            input=[user_query], model=self.config.embedding_model
        )
        # q_vec is the embedding of the user query
        q_vec = q_resp.data[0].embedding

        # embed all snippets
        data_resp = openai.embeddings.create(
            input=snippets, model=self.config.embedding_model
        )
        # emb_list is the embeddings of the snippets
        emb_list = data_resp.data

        # scored is a list of tuples, each tuple contains a snippet and a score
        scored = []
        selected: List[str] = []
        for text, rec in zip(snippets, emb_list):
            emb = rec.embedding
            rel = cosine_similarity(emb, q_vec)
            nov = 1 - max((
                cosine_similarity(
                    emb,
                    openai.embeddings.create(input=[s], model=self.config.embedding_model).data[0].embedding,
                )
                for s in selected
            ), default=0)
            score = 0.7 * rel + 0.3 * nov
            scored.append((text, score))

        # pick top-K snippets by score
        scored.sort(key=lambda x: -x[1])
        top = [text for text, _ in scored[: self.config.citation_limit]]
        return top

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


def search_agent(user_query: str, max_queries: int = 3, *, recent: bool = False) -> str:
    """Convenience wrapper."""
    cfg = SearchConfig(max_queries=max_queries)
    return SearchClient(config=cfg).run(user_query, recent=recent)


# -- Utility: cosine similarity --
from numpy import dot
from numpy.linalg import norm

def cosine_similarity(a, b):
    return dot(a, b) / (norm(a) * norm(b))

if __name__ == "__main__":
    print(search_agent("why does tesla recommend chargin lfp to full even though we know thats bad for batteries?", max_queries=4,recent=True))