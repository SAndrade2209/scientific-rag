"""
full_load_scripts — Production-scale batch extraction & indexing.

Modules:
    storage             — Abstraction for local & S3 file I/O
    extract_pdfs        — Parallel OCR + LLM metadata extraction
    index_documents     — Streaming indexing into Qdrant with checkpoint/resume
    full_pipeline       — End-to-end pipeline (extract → metadata → chunk → embed → index)
                          processes one PDF at a time to keep RAM constant
"""

