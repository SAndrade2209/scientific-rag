"""
ingestion — Document loading, chunking, and metadata preparation.
"""
from .chunker import load_and_chunk_all, get_all_stems, flatten_metadata, chunk_document
__all__ = ["load_and_chunk_all", "get_all_stems", "flatten_metadata", "chunk_document"]
