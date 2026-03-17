# Scientific RAG — Conversational Search over Technical Documents

> A production-grade Retrieval-Augmented Generation system that lets domain experts query dense scientific corpora in natural language and receive cited, precise answers in seconds.

![Python](https://img.shields.io/badge/Python-3.11-blue?logo=python)
![OpenAI](https://img.shields.io/badge/OpenAI-GPT--4.1-412991?logo=openai)
![Qdrant](https://img.shields.io/badge/Vector%20Store-Qdrant-orange)
![Docker](https://img.shields.io/badge/Docker-ready-2496ED?logo=docker)
![License](https://img.shields.io/badge/License-MIT-green)

---

## The Problem

Technical professionals need precise answers buried inside hundreds of dense PDFs. Keyword search breaks down when:

- A concept appears under multiple synonyms (`"skid resistance"`, `"friction coefficient"`, `"µ"`)
- The answer requires **synthesizing information across 3–4 separate papers**
- Follow-up questions depend on earlier turns (`"What about under wet conditions?"`)
- Rare domain terms (`CEL`, `LTPP`, `TDR`, `SMA`) have very low frequency and vanish inside dense embeddings

A single-vector, single-query RAG cannot handle this class of problem reliably. This project builds the multi-stage pipeline that does.

---

## Solution Overview

End-to-end pipeline from raw PDF to cited conversational answer:

```
User question
  ↓
[Guardrail]          — GPT-4.1 screens for safety, domain relevance, prompt injection
  ↓
[Memory]             — Load running summary of the conversation (LangChain CSBM)
  ↓
[Query Expansion]    — GPT-4.1-mini rewrites into N semantically diverse sub-queries
                       + extracts an explicit INTENT statement
  ↓
[Per sub-query]
  → Hybrid Search    — Dense (bge-m3 cosine) + Sparse (BM25) fused with RRF
  → Rerank           — Cross-encoder (bge-reranker-base), retrieve_only=True
  ↓
[Deduplication]      — Merge all sub-query chunks, keep best rerank score per chunk
  ↓
[Re-rank vs INTENT]  — Second pass against the extracted intent (not the raw query)
  ↓
[Generate Answer]    — GPT-4.1-mini with academic citation prompt → APA inline refs
  ↓
[Save Memory]        — Persist turn to ConversationSummaryBufferMemory → SQLite / Redis
```

The Streamlit chat UI (`app.py`) exposes a debug sidebar showing every stage: guardrail verdict, expanded sub-queries, retrieved chunks with scores, and memory state.

---

## Why Each Stage Exists

| Stage | Problem it solves |
|---|---|
| **Guardrail** | Blocks jailbreaks, off-topic queries, and prompt injections before they hit the expensive retrieval chain |
| **Query expansion** | A single embedding misses synonyms and related angles; N diverse sub-queries cover more of the vector space |
| **BM25 sparse vectors** | Dense embeddings compress rare terms (`LTPP`, `CEL`, `TDR`) into indistinguishable regions; BM25 preserves exact-term signal |
| **RRF fusion** | Merges dense + sparse rankings without manual weight tuning — items in both lists rise to the top |
| **`retrieve_only` flag** | Sub-queries skip the LLM call entirely; only **1 LLM completion** is made per turn regardless of query count |
| **Re-rank vs INTENT** | Conversational follow-ups (`"elaborate on that"`) have low lexical signal; ranking against the extracted intent gives better relevance than the raw utterance |
| **ConversationSummaryBufferMemory** | Avoids repeating full history in every prompt; LangChain summarizes older turns, keeping the context window bounded |

---

## Key Engineering Decisions

### 1. Local embedder over API embeddings

`BAAI/bge-m3` runs locally (no per-call cost), supports **8 192-token context windows**, and performs competitively with `text-embedding-3-large` on scientific text. Chunking long technical sections without truncation was a hard requirement.

### 2. Hybrid dense + BM25 in a single Qdrant collection

Both vector types live in the same collection. At query time, two `Prefetch` queries run in parallel and `FusionQuery(Fusion.RRF)` merges them — no infrastructure split, no synchronization problem.

### 3. Two reranking passes

- **Per sub-query** — `bge-reranker-base` scores each retrieved chunk against its own sub-query, filtering noise before the merge.
- **Global re-rank against INTENT** — after deduplication, the merged pool is re-ranked against the extracted intent (e.g., `"Understand hydroplaning physics"`) rather than the raw query (`"tell me more about that"`).

### 4. Stateless `ChatEngine` with external memory

`ChatEngine.process_turn()` is a pure function: `(user_input, memory) → TurnResult`. State lives in the caller (Streamlit `session_state`, a notebook variable). This makes the engine trivially testable and interface-agnostic.

### 5. SQLite-first, Redis-ready session store

Sessions persist to SQLite locally with zero infrastructure. Swapping to Redis (with TTL expiry) requires only a `build_session_store(backend="redis")` call — the interface is identical.

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

## Architecture

### Pipeline stages

| Stage | Component | Model |
|---|---|---|
| **Guardrail** | `GuardrailAgent` | GPT-4.1 |
| **Query expansion** | `QueryExpander` | GPT-4.1-mini |
| **Embedding** | `LocalEmbedder` | BAAI/bge-m3 |
| **Sparse encoding** | `BM25Encoder` | fastembed |
| **Hybrid search** | `VectorStore` | Qdrant RRF fusion |
| **Reranking (×2)** | `RAGPipelineWithReranking` | BAAI/bge-reranker-base |
| **Answer generation** | `MultiQueryRAG` | GPT-4.1-mini |
| **Memory** | LangChain `ConversationSummaryBufferMemory` | GPT-4.1-mini |
| **Session persistence** | `SQLiteSessionStore` / `RedisSessionStore` | — |

---

## Module Reference

| Sub-package | Key exports | Purpose |
|---|---|---|
| `config` | `MD_DIR`, `QDRANT_URL`, `QA_MODEL` | Project-wide paths & defaults |
| `utils` | `LogCapture`, `create_memory` | Shared utilities |
| `ingestion` | `load_and_chunk_all`, `get_all_stems` | Document preparation |
| `vectorstore` | `LocalEmbedder`, `VectorStore` | Embeddings & Qdrant |
| `retrieval` | `RAGPipeline`, `RAGPipelineWithReranking`, `MultiQueryRAG` | Search & generation |
| `orchestration` | `ChatEngine`, `GuardrailAgent`, `QueryExpander`, `build_session_store` | Conversational pipeline |

All public symbols are re-exported from the root package:

```python
from scientific_rag import (
    ChatEngine,           # top-level conversational orchestrator
    VectorStore,          # hybrid Qdrant store
    MultiQueryRAG,        # multi-query retrieval + answer generation
    GuardrailAgent,       # safety + relevance filter
    QueryExpander,        # intent + sub-query generation
    LocalEmbedder,        # bge-m3 local embedder
    build_session_store,  # SQLite or Redis session persistence
    create_memory,        # LangChain memory factory
)
```

---

## License

MIT
