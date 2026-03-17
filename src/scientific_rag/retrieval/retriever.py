from loguru import logger
from openai import OpenAI
from os import environ
from dotenv import load_dotenv
from collections import defaultdict
from ..vectorstore.indexer import VectorStore

load_dotenv()


class RAGPipeline:
    """
    Ties everything together:
      1. Takes a user question
      2. Retrieves the most relevant chunks via hybrid search
      3. Passes them + the question to the LLM
      4. Returns the answer with sources
    """

    def __init__(self, store: VectorStore, answer_prompt, model: str = "gpt-4.1-mini"):
        self.store = store
        self.client = OpenAI(api_key=environ["OPENAI_API_KEY"], timeout=160)
        self.model = model
        self.answer_prompt = answer_prompt

    def disambiguate_same_author_year(self, results):
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
                r["metadata"]["citation"] = citation.replace(f"{r['metadata']['year']}", f"{stem_to_year[stem]}")
                r["metadata"]["year"] = stem_to_year[stem]

        return results

    def build_context(self, results: list[dict]) -> str:
        """
        Formats retrieved chunks into a structured context block for the LLM.
        Each chunk is labeled with its source so the model can cite properly.
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

            parts.append(
                f"[Chunk {i}] {header}\n"
                f"{section_line}"
                f"Relevance score: {r['score']}\n\n"
                f"{r['text']}"
            )

        return "\n\n---\n\n".join(parts)

    def ask(
        self,
        question: str,
        top_k: int = 5,
        filters: dict = None,
        show_chunks: bool = False
    ) -> dict:
        """
        Full RAG cycle.

        Args:
            question:    The user's question in natural language
            top_k:       How many chunks to retrieve (5 is usually a good balance)
            filters:     Optional Qdrant filters, e.g. {"year": 1996} or {"stem": "015..."}
            show_chunks: If True, includes the raw retrieved chunks in the return value
                         (useful for debugging retrieval quality)

        Returns a dict with:
            answer:   the LLM's response
            sources:  deduplicated list of source documents used
            chunks:   raw retrieved chunks (only if show_chunks=True)
        """

        # Step 1: retrieve
        results = self.store.search(question, top_k=top_k, filters=filters)

        if not results:
            return {
                "answer": "No relevant documents found for your question.",
                "sources": [],
                "chunks": []
            }

        # Step 2: build context
        results = self.disambiguate_same_author_year(results)
        context = self.build_context(results)

        # Step 3: generate
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

        # Step 4: collect unique sources
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
                    "score": r["score"],
                })

        logger.info(f"Answer generated from {len(sources)} source(s)")

        return {
            "answer": answer,
            "sources": sources,
            "chunks": results if show_chunks else []
        }
