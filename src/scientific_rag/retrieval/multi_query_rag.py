"""
Multi-Query RAG Pipeline

Receives pre-expanded queries (JSON with rewritten_queries), runs each
through hybrid search + reranking, deduplicates, re-reranks the merged
pool against the original question, and generates a final answer.

Follows the same class conventions as retriever.py and retriever_rerank.py.
"""

from loguru import logger
from openai import OpenAI
from os import environ
from dotenv import load_dotenv

from ..vectorstore.indexer import VectorStore
from .retriever_rerank import RAGPipelineWithReranking

load_dotenv()


class MultiQueryRAG:
    """
    Orchestrates multi-query retrieval over pre-expanded queries.

    Workflow:
      1. Receive expanded queries JSON (original_query + rewritten_queries)
      2. Run each sub-query through RAGPipelineWithReranking (retrieve + rerank only, no LLM)
      3. Deduplicate chunks across all sub-queries, keep best rerank score
      4. Re-rerank the merged pool against the INTENT (fallback: original query)
      5. Build context and generate final answer

    This produces better recall than a single query because different
    phrasings activate different regions of the vector space.
    """

    def __init__(
        self,
        store: VectorStore,
        answer_prompt: str,
        model: str = "gpt-4.1-mini",
        reranker_model: str = "BAAI/bge-reranker-base",
    ):
        self.store = store
        self.client = OpenAI(api_key=environ["OPENAI_API_KEY"], timeout=160)
        self.model = model
        self.answer_prompt = answer_prompt

        # Reuse the existing reranking pipeline for each sub-query
        self.rag = RAGPipelineWithReranking(
            store=store,
            answer_prompt=answer_prompt,
            model=model,
            reranker_model=reranker_model,
        )

        logger.info("MultiQueryRAG initialized")

    def _retrieve_and_dedup(
        self,
        queries: list[str],
        top_k: int = 10,
        candidate_multiplier: int = 3,
        filters: dict = None,
    ) -> dict:
        """
        Runs retrieval + reranking for each sub-query.
        Deduplicates by stem + chunk_index, keeping the highest rerank score.

        Returns:
            dict keyed by chunk_id → chunk dict
        """
        all_chunks = {}

        for i, q in enumerate(queries, 1):
            logger.info(f"Sub-query [{i}/{len(queries)}]: {q[:80]}...")

            r = self.rag.ask(
                q,
                top_k=top_k,
                show_chunks=True,
                candidate_multiplier=candidate_multiplier,
                filters=filters,
                retrieve_only=True,
            )

            logger.info(f"  → {len(r['chunks'])} chunks retrieved")

            for c in r["chunks"]:
                cid = (
                    c["metadata"].get("stem", "")
                    + "_"
                    + c["metadata"].get("chunk_index", "")
                )
                score = c.get("rerank_score", c.get("score", 0))

                if cid not in all_chunks or score > all_chunks[cid]["rerank_score"]:
                    all_chunks[cid] = {
                        "text": c["text"],
                        "metadata": c["metadata"],
                        "rerank_score": score,
                        "original_score": c.get("original_score", c.get("score", 0)),
                        "score": c.get("score", 0),
                    }

        logger.info(f"Total unique chunks after dedup: {len(all_chunks)}")
        return all_chunks

    def ask(
        self,
        expanded_queries: dict,
        top_k: int = 10,
        max_chunks: int = 10,
        candidate_multiplier: int = 3,
        filters: dict = None,
        show_chunks: bool = False,
        conversation_context: str = "",
    ) -> dict:
        """
        Full multi-query RAG cycle.

        Args:
            expanded_queries:      Dict with keys: original_query, intent, rewritten_queries
            top_k:                 Chunks to retrieve per sub-query
            max_chunks:            Final chunks to keep after re-reranking
            candidate_multiplier:  Fetch top_k × multiplier before reranking each sub-query
            filters:               Optional Qdrant metadata filters
            show_chunks:           Include raw chunks in the response
            conversation_context:  Summary of previous conversation turns (for follow-ups)

        Returns:
            dict with: question, intent, sub_queries, answer, sources, chunks
        """
        original_query = expanded_queries["original_query"]
        sub_queries = expanded_queries["rewritten_queries"]
        intent = expanded_queries.get("intent", "—")

        logger.info(f"Original: {original_query}")
        logger.info(f"Intent: {intent}")
        logger.info(f"Sub-queries: {len(sub_queries)}")

        # Step 1: Retrieve per sub-query + deduplicate
        all_chunks = self._retrieve_and_dedup(
            sub_queries,
            top_k=top_k,
            candidate_multiplier=candidate_multiplier,
            filters=filters,
        )

        if not all_chunks:
            return {
                "question": original_query,
                "intent": intent,
                "sub_queries": sub_queries,
                "answer": "No relevant documents found for your question.",
                "sources": [],
                "chunks": [],
            }

        # Step 2: Re-rerank merged pool against the INTENT (better semantic
        # anchor than the raw query, especially for conversational follow-ups
        # where intent carries the full meaning).
        rerank_query = intent if intent and intent != "—" else original_query
        merged = list(all_chunks.values())
        reranked = self.rag.reranker.rerank(rerank_query, merged, top_k=max_chunks)

        logger.info(f"Final reranking: {len(reranked)} chunks selected")
        for i, c in enumerate(reranked[:3], 1):
            stem = c["metadata"].get("stem", "")[:50]
            logger.info(f"  [{i}] rerank={c['rerank_score']:.3f}  {stem}")

        # Step 3: Disambiguate same-author-year citations
        reranked = self.rag.disambiguate_same_author_year(reranked)

        # Step 4: Generate final answer
        context = self.rag.build_context(reranked)

        conv_block = ""
        if conversation_context:
            conv_block = f"""
            CONVERSATION CONTEXT (for continuity only, not a citable source):
            {conversation_context}
            """

        prompt = f"""Answer the following question using only the provided context chunks.

        QUESTION:
        {original_query}
        
        INTENT:
        {intent}
        {conv_block}
        CONTEXT:
        {context}
        """

        response = self.client.responses.create(
            model=self.model,
            input=[{"role": "user", "content": prompt}],
            instructions=self.answer_prompt,
            temperature=0,
        )

        answer = response.output_text

        # Step 5: Collect unique sources
        sources = []
        seen = set()
        for c in reranked:
            stem = c["metadata"].get("stem")
            if stem and stem not in seen:
                seen.add(stem)
                sources.append({
                    "title": c["metadata"].get("title"),
                    "year": c["metadata"].get("year"),
                    "authors": c["metadata"].get("authors"),
                    "citation": c["metadata"].get("citation"),
                    "rerank_score": c.get("rerank_score"),
                    "original_score": c.get("original_score"),
                })

        logger.info(f"Answer generated from {len(sources)} source(s)")

        return {
            "question": original_query,
            "intent": intent,
            "sub_queries": sub_queries,
            "answer": answer,
            "sources": sources,
            "chunks": reranked if show_chunks else [],
        }

