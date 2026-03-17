"""
scientific_rag — Hybrid RAG pipeline for scientific and technical document retrieval.

Sub-packages:
    ingestion      — Document loading, chunking, and metadata preparation
    vectorstore    — Embedding models and Qdrant vector store
    retrieval      — Search, reranking, and answer generation
    orchestration  — Conversational pipeline (guardrail, expander, engine, sessions)
    full_load_scripts — Production batch pipeline

Root modules:
    config         — Centralised project paths and default settings
    utils          — LogCapture, create_memory
"""

from .ingestion import load_and_chunk_all, get_all_stems
from .vectorstore import LocalEmbedder, OpenAIEmbedder, BedrockCohereEmbedder, VectorStore
from .retrieval import RAGPipeline, RAGPipelineWithReranking, MultiQueryRAG
from .orchestration import (
    ChatEngine, TurnResult, init_pipeline,
    GuardrailAgent, QueryExpander,
    build_session_store, serialize_memory, deserialize_memory,
)
from .utils import LogCapture, create_memory

__all__ = [
    # ingestion
    "load_and_chunk_all",
    "get_all_stems",
    # vectorstore
    "LocalEmbedder",
    "OpenAIEmbedder",
    "BedrockCohereEmbedder",
    "VectorStore",
    # retrieval
    "RAGPipeline",
    "RAGPipelineWithReranking",
    "MultiQueryRAG",
    # orchestration
    "GuardrailAgent",
    "QueryExpander",
    "ChatEngine",
    "TurnResult",
    "init_pipeline",
    "build_session_store",
    "serialize_memory",
    "deserialize_memory",
    # utils
    "LogCapture",
    "create_memory",
]
