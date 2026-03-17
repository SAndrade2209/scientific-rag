"""
Enhanced RAG Retriever with Reranking

Adds a cross-encoder reranking stage after initial retrieval.
Cross-encoders are more accurate than bi-encoders (vector similarity)
because they jointly encode the question + chunk together.
"""

from loguru import logger
from openai import OpenAI
from os import environ
from dotenv import load_dotenv
from collections import defaultdict
from sentence_transformers import CrossEncoder
from ..vectorstore.indexer import VectorStore

load_dotenv()


class Reranker:
    """
    Uses a cross-encoder model to rerank retrieved chunks.

    How it works:
    1. Initial retrieval uses fast vector similarity (bi-encoder)
    2. Reranker uses slower but more accurate cross-encoder on top candidates
    3. Cross-encoder sees question + chunk together, not separately
    """

    def __init__(self, model_name: str = "BAAI/bge-reranker-base"):
        """
        Initialize reranker.

        Model options:
        - "cross-encoder/ms-marco-MiniLM-L-6-v2" (default): Fast, 80M params
        - "cross-encoder/ms-marco-MiniLM-L-12-v2": More accurate, 33M params
        - "BAAI/bge-reranker-base": Good for technical/scientific text
        - "BAAI/bge-reranker-large": Best quality but slower
        """
        logger.info(f"Loading reranker model: {model_name}")
        self.model = CrossEncoder(model_name, max_length=512)
        logger.info("Reranker ready")

    def rerank(self, question: str, results: list[dict], top_k: int = None) -> list[dict]:
        """
        Rerank results using cross-encoder scores.

        Args:
            question: The user's query
            results: List of retrieved chunks from initial search
            top_k: Return only top K after reranking (None = return all)

        Returns:
            Reranked results with additional 'rerank_score' field
        """
        if not results:
            return results

        # Prepare (question, chunk) pairs for cross-encoder
        pairs = [(question, r["text"]) for r in results]

        # Get cross-encoder relevance scores
        # Scores are NOT normalized - higher is better
        scores = self.model.predict(pairs)

        # Attach scores to results
        for i, r in enumerate(results):
            r["rerank_score"] = float(scores[i])
            r["original_score"] = r.get("score", 0.0)  # Keep original for comparison

        # Sort by rerank score (descending)
        results = sorted(results, key=lambda x: x["rerank_score"], reverse=True)

        if top_k:
            results = results[:top_k]

        logger.info(f"Reranked {len(results)} chunks")
        return results


class RAGPipelineWithReranking:
    """
    RAG pipeline with reranking enhancement.

    Workflow:
    1. Retrieve top_k * 2 candidates using hybrid search
    2. Rerank candidates using cross-encoder
    3. Take top_k for answer generation
    """

    def __init__(
        self,
        store: VectorStore,
        answer_prompt: str,
        model: str = "gpt-4.1-mini",
        reranker_model: str = "BAAI/bge-reranker-base"
    ):
        self.store = store
        self.client = OpenAI(api_key=environ["OPENAI_API_KEY"], timeout=160)
        self.model = model
        self.answer_prompt = answer_prompt
        self.reranker = Reranker(model_name=reranker_model)

        logger.info("RAG with reranking initialized")

    def disambiguate_same_author_year(self, results):
        """Same as original - adds letter suffixes for same author/year"""
        # 1. Collect one representative per document (stem)
        docs = {}
        for r in results:
            stem = r["metadata"].get("stem")
            if stem not in docs:
                docs[stem] = r

        # 2. Group documents by (authors, year)
        groups = defaultdict(list)
        for r in docs.values():
            meta = r["metadata"]
            key = (meta.get("authors"), meta.get("year"))
            groups[key].append(r)

        # 3. Assign suffixes per document
        stem_to_year = {}

        for (_, year), group in groups.items():
            if year and len(group) > 1:
                for idx, r in enumerate(group):
                    suffix = chr(97 + idx)
                    stem = r["metadata"]["stem"]
                    stem_to_year[stem] = f"{year}{suffix}"

        # 4. Propagate back to all chunks + rebuild citation
        for r in results:
            stem = r["metadata"].get("stem")
            if stem in stem_to_year:
                citation = r["metadata"]["citation"]
                r["metadata"]["citation"] = citation.replace(
                    f"{r['metadata']['year']}",
                    f"{stem_to_year[stem]}"
                )
                r["metadata"]["year"] = stem_to_year[stem]

        return results

    def build_context(self, results: list[dict]) -> str:
        """
        Formats retrieved chunks with rerank scores visible.
        """
        parts = []

        for i, r in enumerate(results, 1):
            meta = r["metadata"]

            fields = []

            if meta.get("authors"):
                fields.append(f"authors: {meta['authors']}")

            if meta.get("citation"):
                fields.append(f"citation: {meta['citation']}")

            section = meta.get("section_h2") or meta.get("section_h1")

            header = " ".join(fields)
            section_line = f"Section: {section}\n" if section else ""

            # Show both scores if reranked
            if 'rerank_score' in r:
                score_line = f"Relevance: {r['rerank_score']:.3f} (original: {r['original_score']:.3f})\n"
            else:
                score_line = f"Relevance score: {r['score']:.3f}\n"

            parts.append(
                f"[Chunk {i}] {header}\n"
                f"{section_line}"
                f"{score_line}\n"
                f"{r['text']}"
            )

        return "\n\n---\n\n".join(parts)

    def ask(
        self,
        question: str,
        top_k: int = 5,
        filters: dict = None,
        show_chunks: bool = False,
        candidate_multiplier: int = 2,
        retrieve_only: bool = False,
    ) -> dict:
        """
        RAG with reranking.

        Args:
            question: User's question
            top_k: Final number of chunks to use for answer
            filters: Optional Qdrant metadata filters
            show_chunks: Include raw chunks in response
            candidate_multiplier: Fetch top_k * multiplier for reranking
                                 Higher = better reranking quality but slower
            retrieve_only: If True, skip LLM answer generation and return
                          only the reranked chunks. Used by MultiQueryRAG
                          to avoid unnecessary API calls per sub-query.

        Returns:
            dict with answer, sources, and optionally chunks
        """
        logger.info(f"Question: {question[:100]}...")

        # Step 1: Retrieve more candidates than needed
        candidate_k = top_k * candidate_multiplier
        logger.info(f"Retrieving {candidate_k} candidates for reranking...")

        results = self.store.search(question, top_k=candidate_k, filters=filters)

        if not results:
            return {
                "answer": "No relevant documents found for your question.",
                "sources": [],
                "chunks": []
            }

        logger.info(f"Initial retrieval: {len(results)} chunks")
        top3_before = [f"{r['score']:.3f}" for r in results[:3]]
        logger.info(f"Top 3 scores before reranking: {top3_before}")

        # Step 2: Rerank with cross-encoder
        results = self.reranker.rerank(question, results, top_k=top_k)

        logger.info(f"After reranking: {len(results)} chunks")
        top3_after = [f"{r['rerank_score']:.3f}" for r in results[:3]]
        logger.info(f"Top 3 rerank scores: {top3_after}")

        # Early return: skip LLM call when only chunks are needed
        if retrieve_only:
            logger.info("retrieve_only=True — skipping LLM call")
            return {
                "answer": "",
                "sources": [],
                "chunks": results,
            }

        # Step 3: Prepare for LLM
        results = self.disambiguate_same_author_year(results)
        context = self.build_context(results)
        
        # Step 4: Generate answer
        prompt = f"""Answer the following question using only the provided context chunks.
        
        QUESTION:
        {question}
        
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
        for r in results:
            stem = r["metadata"].get("stem")
            if stem and stem not in seen:
                seen.add(stem)
                sources.append({
                    "title": r["metadata"].get("title"),
                    "year": r["metadata"].get("year"),
                    "authors": r["metadata"].get("authors"),
                    "citation": r["metadata"].get("citation"),
                    "rerank_score": r.get("rerank_score"),
                    "original_score": r.get("original_score"),
                })

        logger.info(f"Answer generated from {len(sources)} source(s)")

        return {
            "answer": answer,
            "sources": sources,
            "chunks": results if show_chunks else []
        }

