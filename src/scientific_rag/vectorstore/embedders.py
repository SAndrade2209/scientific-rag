from abc import ABC, abstractmethod
from sentence_transformers import SentenceTransformer
from openai import OpenAI
import boto3, json

# ---------------- BASE ---------------- #
class BaseEmbedder(ABC):
    """
    Every embedding provider must implement these three things.
    The rest of the pipeline only talks to this interface —
    it never imports LocalEmbedder or OpenAIEmbedder directly.
    That's what makes swapping providers a one-line change.
    """

    @abstractmethod
    def embed_document(self, text: str) -> list[float]:
        """
        Called when indexing a chunk into Qdrant.
        Some models behave differently for documents vs queries,
        so we keep them as separate methods even if they're identical right now.
        """
        pass

    @abstractmethod
    def embed_query(self, text: str) -> list[float]:
        """
        Called when embedding the user's question at search time.
        """
        pass

    @property
    @abstractmethod
    def vector_size(self) -> int:
        """
        The dimension of the vectors this model produces.
        Qdrant needs this number when you create a collection —
        and it must match exactly what you index, so we tie it to the embedder.
        """
        pass


# ---------------- LOCAL (FREE) ---------------- #

class LocalEmbedder(BaseEmbedder):
    """
    Runs entirely on machine using sentence-transformers.

    The model downloads once (~2GB for bge-m3) and is cached locally.
    bge-m3 is a strong choice for technical/scientific docs:
    - handles up to 8192 tokens per chunk (very long sections won't get cut off)
    - multilingual
    - performs well on retrieval benchmarks vs much larger paid models
    """

    def __init__(self, model_name: str = "BAAI/bge-m3"):
        logger_msg = f"Loading local embedder: {model_name}"
        print(logger_msg)  # swap for loguru logger if you prefer
        # This line downloads the model on first run, then loads from cache
        self.model = SentenceTransformer(model_name)
        self._model_name = model_name
        print("Embedder ready.")

    def embed_document(self, text: str) -> list[float]:
        # normalize_embeddings=True ensures vectors sit on the unit sphere,
        # which makes cosine similarity scores more consistent and comparable
        return self.model.encode(text, normalize_embeddings=True).tolist()

    def embed_query(self, text: str) -> list[float]:
        # bge-m3 doesn't require a different mode for queries,
        # but we keep this method separate in case you switch to a model that does
        return self.model.encode(text, normalize_embeddings=True).tolist()

    @property
    def vector_size(self) -> int:
        return 1024  # bge-m3 output dimension


# ---------------- OPENAI (ready when you need it) ---------------- #

class OpenAIEmbedder(BaseEmbedder):
    """
    Uses OpenAI's embedding API.
    To switch: just replace LocalEmbedder with this in your indexing script.
    Nothing else in the pipeline changes.

    Note: if you switch providers AFTER indexing, you must re-index everything.
    Vectors from different models are not comparable.

    Models:
      - text-embedding-3-small → 1536 dims, cheap, good quality
      - text-embedding-3-large → 3072 dims, best quality, higher cost
    """

    def __init__(self, model_name: str = "text-embedding-3-small", api_key: str = None):

        self.client = OpenAI(api_key=api_key)
        self._model_name = model_name

    def _embed(self, text: str) -> list[float]:
        response = self.client.embeddings.create(
            input=text,
            model=self._model_name
        )
        return response.data[0].embedding

    def embed_document(self, text: str) -> list[float]:
        return self._embed(text)

    def embed_query(self, text: str) -> list[float]:
        return self._embed(text)

    @property
    def vector_size(self) -> int:
        sizes = {
            "text-embedding-3-small": 1536,
            "text-embedding-3-large": 3072,
        }
        return sizes[self._model_name]


# ---------------- COHERE via AWS Bedrock (future option) ---------------- #

class BedrockCohereEmbedder(BaseEmbedder):
    """
    Uses Cohere Embed v3 via AWS Bedrock.
    Best option if you move to production on AWS.

    The key advantage over other providers: Cohere has explicit
    input_type separation — "search_document" vs "search_query".
    This asymmetric embedding meaningfully improves retrieval quality.
    """

    def __init__(self, model_id: str = "cohere.embed-english-v3", region: str = "us-east-1"):

        self.client = boto3.client("bedrock-runtime", region_name=region)
        self.model_id = model_id
        self._json = json

    def _embed(self, text: str, input_type: str) -> list[float]:
        body = self._json.dumps({"texts": [text], "input_type": input_type})
        response = self.client.invoke_model(
            modelId=self.model_id,
            body=body,
            contentType="application/json",
            accept="application/json"
        )
        result = self._json.loads(response["body"].read())
        return result["embeddings"][0]

    def embed_document(self, text: str) -> list[float]:
        return self._embed(text, input_type="search_document")

    def embed_query(self, text: str) -> list[float]:
        return self._embed(text, input_type="search_query")

    @property
    def vector_size(self) -> int:
        return 1024