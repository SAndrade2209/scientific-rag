# indexer.py
from fastembed import SparseTextEmbedding
import uuid
from loguru import logger
from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    VectorParams,
    SparseVectorParams,
    SparseIndexParams,
    PointStruct,
    SparseVector,
    FusionQuery,
    Prefetch,
    Fusion,
)
from qdrant_client.models import Filter, FieldCondition, MatchValue
from .embedders import BaseEmbedder


DENSE_VECTOR_NAME = "dense"
SPARSE_VECTOR_NAME = "sparse"
COLLECTION_NAME = "technical_docs"
QDRANT_URL = "http://localhost:6333"


# ---------------- SPARSE ENCODER ---------------- #
class BM25Encoder:
    """
    Wraps fastembed's BM25 sparse encoder.
    Sparse vectors are lists of (index, weight) pairs — only non-zero terms
    are stored, which is why they're called "sparse".

    BM25 is the same algorithm used by search engines like Elasticsearch.
    It scores terms based on frequency in the chunk vs rarity across all docs.
    Rare technical terms (e.g. "LTPP", "SMA", "TDR") get high weights,
    which is exactly what we want for scientific documents.
    """

    def __init__(self):
        logger.info("Loading BM25 sparse encoder...")
        self.model = SparseTextEmbedding(model_name="Qdrant/bm25")
        logger.info("BM25 encoder ready.")

    def encode(self, text: str) -> SparseVector:
        """
        Returns a Qdrant SparseVector with indices and values.
        indices = which vocabulary terms appear in the text
        values  = their BM25 weights
        """
        result = list(self.model.embed(text))[0]
        return SparseVector(
            indices=result.indices.tolist(),
            values=result.values.tolist()
        )


# ---------------- VECTOR STORE ---------------- #
class VectorStore:
    """
    Manages both dense (semantic) and sparse (BM25) vectors in Qdrant.
    At search time, runs both and merges results with RRF fusion.
    """

    def __init__(
            self,
            embedder: BaseEmbedder,
            collection_name: str = COLLECTION_NAME,
            url: str = QDRANT_URL,
            use_hybrid: bool = True,
            force_recreate: bool = False
    ):
        self.embedder = embedder
        self.collection_name = collection_name
        self.client = QdrantClient(url=url)
        self.use_hybrid = use_hybrid
        self.bm25 = BM25Encoder() if use_hybrid else None
        self._ensure_collection(force_recreate=force_recreate)

    def _ensure_collection(self, force_recreate: bool = False):
        existing = [c.name for c in self.client.get_collections().collections]

        if self.collection_name in existing:
            if force_recreate:
                logger.info(f"force_recreate=True — deleting and rebuilding collection.")
                self.client.delete_collection(self.collection_name)
            else:
                # Collection exists and we're not forcing — just connect, don't touch it
                logger.info(f"Collection '{self.collection_name}' found, connecting to existing.")
                return

        logger.info(f"Creating collection '{self.collection_name}'")
        if self.use_hybrid:
            self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config={
                    DENSE_VECTOR_NAME: VectorParams(
                        size=self.embedder.vector_size,
                        distance=Distance.COSINE
                    )
                },
                sparse_vectors_config={
                    SPARSE_VECTOR_NAME: SparseVectorParams(
                        index=SparseIndexParams(on_disk=False)
                    )
                }
            )
        else:
            self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config=VectorParams(
                    size=self.embedder.vector_size,
                    distance=Distance.COSINE
                )
            )

    def index_chunks(self, chunks: list[dict], batch_size: int = 32):
        total = len(chunks)
        logger.info(f"Indexing {total} chunks in batches of {batch_size}...")

        for batch_start in range(0, total, batch_size):
            batch = chunks[batch_start: batch_start + batch_size]
            points = []

            for chunk in batch:
                dense_vector = self.embedder.embed_document(chunk["text"])

                if self.use_hybrid:
                    sparse_vector = self.bm25.encode(chunk["text"])
                    point = PointStruct(
                        id=str(uuid.uuid4()),
                        vector={
                            DENSE_VECTOR_NAME: dense_vector,
                            SPARSE_VECTOR_NAME: sparse_vector,
                        },
                        payload={
                            "text": chunk["text"],
                            **chunk["metadata"]
                        }
                    )
                else:
                    point = PointStruct(
                        id=str(uuid.uuid4()),
                        vector=dense_vector,
                        payload={
                            "text": chunk["text"],
                            **chunk["metadata"]
                        }
                    )

                points.append(point)

            self.client.upsert(collection_name=self.collection_name, points=points)
            logger.info(f"  Indexed {min(batch_start + batch_size, total)}/{total}")

        logger.info("Indexing complete.")

    def search(self, query: str, top_k: int = 5, filters: dict = None) -> list[dict]:
        """
        Hybrid search: runs dense + sparse in parallel, merges with RRF.

        RRF (Reciprocal Rank Fusion) works by:
          1. Ranking results from dense search (1st, 2nd, 3rd...)
          2. Ranking results from sparse/BM25 search (1st, 2nd, 3rd...)
          3. Combining ranks: score = 1/(rank + 60) for each list, then summed

        The 60 is a smoothing constant — results that rank well in BOTH
        lists get a big boost. Results only good in one list get partial credit.
        No manual weight tuning needed.
        """

        qdrant_filter = None
        if filters:

            conditions = [
                FieldCondition(key=k, match=MatchValue(value=v))
                for k, v in filters.items()
            ]
            qdrant_filter = Filter(must=conditions)

        if self.use_hybrid:
            dense_vector = self.embedder.embed_query(query)
            sparse_vector = self.bm25.encode(query)

            results = self.client.query_points(
                collection_name=self.collection_name,
                prefetch=[
                    # First pass: fetch candidate results from each method independently
                    Prefetch(
                        query=dense_vector,
                        using=DENSE_VECTOR_NAME,
                        limit=top_k * 10,  # fetch more candidates than needed
                        filter=qdrant_filter  # apply filters inside each pass
                    ),
                    Prefetch(
                        query=sparse_vector,
                        using=SPARSE_VECTOR_NAME,
                        limit=top_k * 10,
                        filter=qdrant_filter
                    ),
                ],
                # Second pass: merge both candidate lists with RRF
                query=FusionQuery(fusion=Fusion.RRF),
                limit=top_k,
                with_payload=True
            ).points
        else:
            # Dense-only fallback
            results = self.client.query_points(
                collection_name=self.collection_name,
                query=self.embedder.embed_query(query),
                limit=top_k,
                query_filter=qdrant_filter,
                with_payload=True
            ).points

        return [
            {
                "text": r.payload.get("text", ""),
                "score": r.score,
                "metadata": {k: v for k, v in r.payload.items() if k != "text"}
            }
            for r in results
        ]

    def collection_info(self):
        info = self.client.get_collection(self.collection_name)
        logger.info(f"Collection '{self.collection_name}': {info.points_count} vectors stored")
        return info
