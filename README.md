# scientific-rag

A production-ready **Conversational RAG** (Retrieval-Augmented Generation) system for querying collections of scientific and technical PDF documents using natural language.

---

## Overview

End-to-end pipeline that ingests technical PDFs, converts them to structured markdown, chunks and embeds them with a local **bge-m3** model, indexes into **Qdrant** with hybrid (dense + BM25 sparse) search, and answers questions through a multi-stage conversational pipeline:

```
User input
  → Guardrail (GPT-4.1)         — block or sanitize unsafe / off-topic input
  → Memory                       — load conversation summary
  → Query Expansion (GPT-4.1-mini) — generate intent + N sub-queries
  → Per sub-query: hybrid search → cross-encoder rerank (retrieve_only, no LLM)
  → Deduplicate → re-rerank merged pool against INTENT
  → Generate answer (GPT-4.1-mini) with academic citation prompt
  → Save to memory → persist session to SQLite / Redis
```

The Streamlit chat UI (`app.py`) provides a debug sidebar with guardrail verdicts, query expansion details, source citations, memory state, and session persistence diagnostics.

---

## Features

- 🔍 **Hybrid search** — dense vectors (bge-m3) + BM25 sparse vectors, fused with RRF
- 🔁 **Multi-query expansion** — GPT rewrites each query into semantically diverse sub-queries
- 📊 **Cross-encoder reranking** — bge-reranker-base for precise relevance scoring
- 🛡️ **Guardrail agent** — safety and relevance filter before retrieval
- 🧠 **Conversation memory** — LangChain `ConversationSummaryBufferMemory` with SQLite/Redis persistence
- 📄 **Structured citation** — APA-format inline citations from document metadata
- 🐳 **Docker-ready** — Qdrant runs via `docker compose up`

---

## Tech Stack

| Layer | Technology |
|---|---|
| LLM | OpenAI GPT-4.1 / GPT-4.1-mini |
| Embeddings | BAAI/bge-m3 (local, 8192 tokens) |
| Sparse encoding | fastembed BM25 |
| Vector store | Qdrant (hybrid dense + sparse, RRF fusion) |
| Reranker | BAAI/bge-reranker-base |
| Memory | LangChain ConversationSummaryBufferMemory |
| Session store | SQLite (local) / Redis (production) |
| PDF parsing | Docling |
| UI | Streamlit |

---

## Project Structure

```
scientific_rag/
├── app.py                            # Streamlit chat interface
├── docker-compose.yml                # Qdrant with persistent storage
├── pyproject.toml
├── requirements.txt
│
├── src/scientific_rag/               # Installable Python package
│   ├── __init__.py                   # Re-exports all public symbols
│   ├── config.py                     # Centralised paths & defaults
│   ├── utils.py                      # LogCapture, create_memory
│   ├── prompts/                      # LLM system prompts (plain text)
│   │
│   ├── ingestion/                    # ETL: document preparation
│   │   └── chunker.py               #   Markdown loading, splitting & metadata
│   │
│   ├── vectorstore/                  # Embeddings & vector storage
│   │   ├── embedders.py             #   LocalEmbedder (bge-m3), OpenAI, Bedrock
│   │   └── indexer.py               #   Qdrant VectorStore — hybrid dense+BM25
│   │
│   ├── retrieval/                    # Search & answer generation
│   │   ├── retriever.py             #   RAGPipeline — basic retrieve + generate
│   │   ├── retriever_rerank.py      #   + cross-encoder reranking
│   │   └── multi_query_rag.py       #   Multi-query retrieval + dedup + re-rerank
│   │
│   ├── orchestration/                # Conversational pipeline
│   │   ├── chat_engine.py           #   ChatEngine — orchestrates a full turn
│   │   ├── guardrail.py             #   GuardrailAgent — safety & relevance filter
│   │   ├── query_expander.py        #   QueryExpander — multi-query generation
│   │   └── session_store.py         #   SQLite/Redis session persistence
│   │
│   └── full_load_scripts/           # Production batch pipeline
│       ├── extract_pdfs.py          #   PDF → markdown + metadata
│       ├── index_documents.py       #   Chunk → embed → Qdrant
│       ├── full_pipeline.py         #   One-PDF-at-a-time full load
│       └── storage.py               #   Local/S3 storage abstraction
│
├── notebooks/
│   ├── pipeline/                     # Reproducible step-by-step pipeline
│   │   ├── 00_pdf_extraction.ipynb
│   │   ├── 01_indexing.ipynb
│   │   └── 02_retrieval.ipynb
│   ├── exploration/                  # Prototyping & interactive demos
│   │   ├── chat_rag.ipynb
│   │   ├── session_memory_demo.ipynb
│   │   └── full_load.ipynb
│   ├── testing/                      # Evaluation & benchmarks
│   │   ├── test_rag.ipynb
│   │   └── query_test.ipynb
│   └── archive/                      # Superseded notebooks
│
├── tests/
│   ├── test_quick.py                 # Quick smoke test
│   ├── test_reranking.py             # Reranking pipeline verification
│   └── test_questions_comprehensive.py
│
└── data/
    ├── raw/                          # Original PDFs (gitignored)
    ├── processed/
    │   ├── markdown/                 # Docling-converted markdown (gitignored)
    │   └── metadata/                 # Extracted reference JSON (gitignored)
    └── sessions.db                   # SQLite session store (auto-created, gitignored)
```

---

## Setup

### 1. Clone the repository

```bash
git clone https://github.com/<your-username>/scientific-rag.git
cd scientific-rag
```

### 2. Install dependencies

```bash
pip install -e ".[dev]"
# or
pip install -r requirements.txt
```

### 3. Configure environment

```bash
cp .env.example .env
# Edit .env and add your OPENAI_API_KEY
```

### 4. Start Qdrant

```bash
docker compose up -d
```

---

## Workflow

### 1. Add your PDFs

Place your PDF files in `data/raw/`.

### 2. Ingest documents

Run the notebooks in order:

```
notebooks/pipeline/00_pdf_extraction.ipynb   →  PDF → markdown + metadata JSON
notebooks/pipeline/01_indexing.ipynb         →  Chunk → embed → Qdrant
```

Or use the batch script:

```bash
python -m scientific_rag.full_load_scripts.full_pipeline
```

### 3. Start the chat UI

```bash
streamlit run app.py
```

Or explore with notebooks:

```
notebooks/pipeline/02_retrieval.ipynb       →  Single queries with reranking
notebooks/exploration/chat_rag.ipynb        →  Conversational RAG prototype
```

---

## Running Tests

```bash
pytest tests/
```

---

## Architecture

### Pipeline stages

| Stage | Component | Model |
|---|---|---|
| **Guardrail** | `GuardrailAgent` | GPT-4.1 |
| **Query expansion** | `QueryExpander` | GPT-4.1-mini |
| **Embedding** | `LocalEmbedder` | BAAI/bge-m3 |
| **Sparse encoding** | Built-in BM25 | fastembed |
| **Hybrid search** | `VectorStore` | Qdrant RRF fusion |
| **Reranking** | `Reranker` | BAAI/bge-reranker-base |
| **Answer generation** | `MultiQueryRAG` | GPT-4.1-mini |
| **Memory** | LangChain `ConversationSummaryBufferMemory` | GPT-4.1-mini |
| **Session persistence** | `SQLiteSessionStore` / `RedisSessionStore` | — |

### Key design decisions

| Choice | Rationale |
|---|---|
| `bge-m3` local embedder | Free, 8192 tokens, strong multilingual & scientific text |
| BM25 sparse vectors | Catches rare technical terms missed by dense vectors |
| RRF fusion | Merges dense + sparse without manual weight tuning |
| Multi-query expansion | Different phrasings activate different vector regions |
| `retrieve_only` flag | Sub-queries skip LLM call — only 1 LLM call total per turn |
| Re-rerank against **intent** | More accurate than raw query for conversational follow-ups |
| Cross-encoder reranking | Jointly encodes question + chunk for precise relevance scoring |
| SQLite → Redis session store | Zero infrastructure locally, auto-expiring in production |

---

## Module Reference

| Sub-package | Key exports | Purpose |
|---|---|---|
| `config` | `MD_DIR`, `QDRANT_URL`, `QA_MODEL`, … | Project-wide constants |
| `utils` | `LogCapture`, `create_memory` | Shared utilities |
| `ingestion` | `load_and_chunk_all`, `get_all_stems` | Document preparation |
| `vectorstore` | `LocalEmbedder`, `VectorStore` | Embeddings & Qdrant |
| `retrieval` | `RAGPipeline`, `RAGPipelineWithReranking`, `MultiQueryRAG` | Search & generation |
| `orchestration` | `ChatEngine`, `GuardrailAgent`, `QueryExpander`, `build_session_store` | Conversational pipeline |

All symbols are re-exported from the root `__init__.py`:

```python
from scientific_rag import ChatEngine, VectorStore, MultiQueryRAG
```

---

## License

MIT
