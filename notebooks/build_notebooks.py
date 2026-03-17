"""
Generates project notebooks via nbformat so the JSON is always valid.

  00_pdf_extraction.ipynb  — PDF → markdown + reference JSON
  01_indexing.ipynb        — chunk → embed → Qdrant
  02_retrieval.ipynb       — RAG queries (reranking)

Run once: python _build_notebooks.py
"""

import nbformat
from nbformat.v4 import new_notebook, new_markdown_cell, new_code_cell
from pathlib import Path

BASE = Path(__file__).parent


# ─────────────────────────────────────────────────────────────────────────────
# helpers
# ─────────────────────────────────────────────────────────────────────────────
def md(src: str):
    return new_markdown_cell(src)

def code(src: str):
    return new_code_cell(src)

def save(nb, path: Path):
    nb.metadata.update({
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3.10.0"},
    })
    nbformat.write(nb, str(path))
    print(f"  wrote  {path.name}")


# ─────────────────────────────────────────────────────────────────────────────
# 00_pdf_extraction.ipynb
# ─────────────────────────────────────────────────────────────────────────────
nb0 = new_notebook()
nb0.cells = [

md("""\
# 00 — PDF Extraction

Converts PDFs into:
1. **Full markdown** (`md_docs/*.md`) — for chunking & embedding
2. **Reference metadata JSON** (`md_ref/*.json`) — title, authors, year, DOI, etc.

Uses:
- **Docling** for OCR + layout-aware markdown export
- **GPT-4.1-mini** to extract structured reference metadata from the first 5 pages

> Run this notebook **once per batch of new PDFs**.  
> Already-processed files (with existing `.json`) are skipped automatically."""),

md("## 1 — Imports & Config"),

code("""\
import time
import json
from os import environ
from pathlib import Path
from dotenv import load_dotenv
from loguru import logger
from openai import OpenAI

from docling.datamodel.accelerator_options import AcceleratorDevice, AcceleratorOptions
from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import PdfPipelineOptions, TableStructureOptions
from docling.document_converter import DocumentConverter, PdfFormatOption

load_dotenv()

API_KEY = environ["OPENAI_API_KEY"]

BASE_DIR    = Path(".").resolve().parent  # project root
PDF_DIR     = BASE_DIR / "pdf"
PROMPT_PATH = BASE_DIR / "prompts/extract_references_prompt"

MD_OUTPUT_DIR   = BASE_DIR / "md_docs"
JSON_OUTPUT_DIR = BASE_DIR / "md_ref"

MODEL  = "gpt-4.1-mini"
client = OpenAI(api_key=API_KEY, timeout=1500)

print(f"PDF source     : {PDF_DIR}")
print(f"Markdown output: {MD_OUTPUT_DIR}")
print(f"JSON output    : {JSON_OUTPUT_DIR}")\
"""),

md("## 2 — Helper Functions"),

code("""\
def read_txt(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def ensure_dirs():
    MD_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    JSON_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    print("Output directories ready.")\
"""),

md("## 3 — Docling OCR Converter"),

code("""\
def build_doc_converter():
    \"\"\"
    Configure Docling PDF converter with OCR and table extraction.
    \"\"\"
    options = PdfPipelineOptions()
    options.do_ocr = True
    options.do_table_structure = True
    options.table_structure_options = TableStructureOptions(do_cell_matching=True)
    options.ocr_options.lang = ["en"]
    options.accelerator_options = AcceleratorOptions(
        num_threads=4,
        device=AcceleratorDevice.AUTO
    )

    return DocumentConverter(
        format_options={
            InputFormat.PDF: PdfFormatOption(pipeline_options=options)
        }
    )


def extract_pdf(pdf_path: Path, converter: DocumentConverter):
    \"\"\"Run OCR on a single PDF and return the conversion result.\"\"\"
    start = time.time()
    result = converter.convert(pdf_path)
    logger.info(f"OCR finished in {time.time() - start:.2f}s")
    return result


def get_pages_1_to_5_markdown(conv_result) -> str:
    \"\"\"Export only the first 5 pages as markdown (for reference extraction).\"\"\"
    pages = []
    for page_no in sorted(conv_result.document.pages):
        if page_no > 5:
            break
        pages.append(conv_result.document.export_to_markdown(page_no=page_no))
    return "\\n\\n".join(pages)


print("OCR functions defined.")\
"""),

md("## 4 — LLM Reference Extraction"),

code("""\
def get_reference_json(ocr_text: str, prompt_template: str) -> dict:
    \"\"\"
    Send the first 5 pages to the LLM and extract structured reference metadata.
    \"\"\"
    prompt = prompt_template.replace("{OCR_TEXT}", ocr_text)

    response = client.responses.create(
        model=MODEL,
        input=prompt,
        temperature=0,
        text={"format": {"type": "json_object"}},
    )

    return json.loads(response.output_text)


print("LLM extraction function defined.")\
"""),

md("## 5 — Single-PDF Processing Pipeline"),

code("""\
def process_pdf(pdf_path: Path, converter, prompt_template):
    \"\"\"
    Full pipeline for one PDF:
      1. Skip if JSON already exists
      2. OCR → full markdown
      3. Extract first 5 pages → LLM → reference JSON
    \"\"\"
    stem = pdf_path.stem
    json_path = JSON_OUTPUT_DIR / f"{stem}.json"
    md_path   = MD_OUTPUT_DIR   / f"{stem}.md"

    if json_path.exists():
        logger.info(f"Skipping (already processed): {stem}")
        return

    logger.info(f"Processing: {pdf_path.name}")

    # OCR full document
    conv = extract_pdf(pdf_path, converter)

    # Save full markdown
    md_path.write_text(conv.document.export_to_markdown(), encoding="utf-8")

    # Extract reference metadata from first 5 pages
    ocr_text = get_pages_1_to_5_markdown(conv)
    reference = get_reference_json(ocr_text, prompt_template)

    # Save reference JSON
    json_path.write_text(
        json.dumps(reference, indent=2, ensure_ascii=False),
        encoding="utf-8"
    )

    logger.info(f"Completed: {stem}")


print("process_pdf() defined.")\
"""),

md("## 6 — Batch Process All PDFs"),

code("""\
ensure_dirs()

prompt_template = read_txt(PROMPT_PATH)
converter = build_doc_converter()

pdfs = sorted(PDF_DIR.glob("*.pdf"))
logger.info(f"Found {len(pdfs)} PDFs in {PDF_DIR}")

for pdf in pdfs:
    try:
        process_pdf(pdf, converter, prompt_template)
    except Exception:
        logger.exception(f"Failed on {pdf.name}")

logger.info("All documents processed.")\
"""),

md("## 7 — Verify Outputs"),

code("""\
md_files   = sorted(MD_OUTPUT_DIR.glob("*.md"))
json_files = sorted(JSON_OUTPUT_DIR.glob("*.json"))

print(f"Markdown files : {len(md_files)}")
print(f"JSON files     : {len(json_files)}")

# Show first few
print("\\n── Sample markdown files ──")
for f in md_files[:5]:
    print(f"  {f.name}")

print("\\n── Sample JSON files ──")
for f in json_files[:5]:
    print(f"  {f.name}")\
"""),

md("""\
## 8 — Inspect a Reference JSON

Pick any stem to preview its extracted metadata."""),

code("""\
import json

# Change this to any document stem you want to inspect
STEM = "001. The Joint Winter Runway Friction NASA Perspective"

json_path = JSON_OUTPUT_DIR / f"{STEM}.json"

if json_path.exists():
    meta = json.loads(json_path.read_text(encoding="utf-8"))
    print(f"Title   : {meta.get('title')}")
    print(f"Year    : {meta.get('year')}")
    print(f"Authors : {meta.get('authors')}")
    print(f"DOI     : {meta.get('doi')}")
    print(f"Type    : {meta.get('type')}")
else:
    print(f"File not found: {json_path}")\
"""),

md("""\
---

✅ **Done.** Next step: open `01_indexing.ipynb` to chunk, embed, and index into Qdrant."""),

]

save(nb0, BASE / "00_pdf_extraction.ipynb")


# ─────────────────────────────────────────────────────────────────────────────
# 01_indexing.ipynb
# ─────────────────────────────────────────────────────────────────────────────
nb1 = new_notebook()
nb1.cells = [

md("""\
# 01 — Indexing Pipeline

Everything that only needs to run **once** (or when documents change):

1. Load markdown docs and metadata from `md_docs/` and `md_ref/`
2. Chunk each document with header-aware splitting
3. Embed chunks with the local `bge-m3` model (dense vectors)
4. Encode with BM25 (sparse vectors)
5. Upsert into the Qdrant vector store

> Run this notebook once to populate the database.  
> After that, open `02_retrieval.ipynb` to run queries."""),

md("## 1 — Imports & Config"),

code("""\
from pathlib import Path
from dotenv import load_dotenv
from loguru import logger

from chunker import load_and_chunk_all, get_all_stems
from embedders import LocalEmbedder
from indexer import VectorStore

load_dotenv()

BASE_DIR = Path(".").resolve().parent  # project root

# ── Chunking ──────────────────────────────────────────────────────────────────
MAX_CHUNK_SIZE = 1000   # characters per chunk
CHUNK_OVERLAP  = 100    # overlap between consecutive chunks

# ── Qdrant ────────────────────────────────────────────────────────────────────
COLLECTION_NAME = "technical_docs"
QDRANT_URL      = "http://localhost:6333"
USE_HYBRID      = True   # dense + BM25 sparse vectors

print("Config ready.")
print(f"  Collection : {COLLECTION_NAME}")
print(f"  Hybrid     : {USE_HYBRID}")
print(f"  Chunk size : {MAX_CHUNK_SIZE} chars  |  overlap: {CHUNK_OVERLAP}")\
"""),

md("## 2 — Inspect Available Documents"),

code("""\
stems = get_all_stems()

print(f"Found {len(stems)} documents in md_docs/:\\n")
for s in stems:
    print(f"  • {s}")\
"""),

md("## 3 — Chunk All Documents"),

code("""\
all_chunks = load_and_chunk_all(
    max_chunk_size=MAX_CHUNK_SIZE,
    chunk_overlap=CHUNK_OVERLAP
)

print(f"\\nTotal chunks ready to embed: {len(all_chunks)}")

sample = all_chunks[0]
print("\\n── Sample chunk ─────────────────────────────────────────────────")
print(f"  stem    : {sample['metadata'].get('stem')}")
print(f"  title   : {sample['metadata'].get('title')}")
print(f"  section : {sample['metadata'].get('section_h2') or sample['metadata'].get('section_h1')}")
print(f"  chars   : {len(sample['text'])}")
print(f"  preview : {sample['text'][:200]}")\
"""),

md("## 4 — Chunk Statistics"),

code("""\
from collections import Counter

per_doc = Counter(c['metadata']['stem'] for c in all_chunks)

print(f"{'Document stem':<55} {'chunks':>6}")
print("-" * 65)
for stem, count in sorted(per_doc.items()):
    print(f"  {stem:<53} {count:>6}")
print("-" * 65)
print(f"  {'TOTAL':<53} {sum(per_doc.values()):>6}")

lengths = [len(c['text']) for c in all_chunks]
print(f"\\nChunk length — min: {min(lengths)}  max: {max(lengths)}  avg: {round(sum(lengths)/len(lengths))}")\
"""),

md("## 5 — Load Embedder"),

code("""\
# Downloads the model on first run (~2 GB), then loads from cache.
embedder = LocalEmbedder()

print(f"Embedder model : BAAI/bge-m3")
print(f"Vector size    : {embedder.vector_size}")\
"""),

md("""\
## 6 — Connect / Create Qdrant Collection

> ⚠️ Set `FORCE_RECREATE = True` **only** if you want to wipe the collection and rebuild.  
> Leave it as `False` to connect to the existing collection without re-indexing."""),

code("""\
FORCE_RECREATE = False   # ← change to True to rebuild from scratch

store = VectorStore(
    embedder=embedder,
    collection_name=COLLECTION_NAME,
    url=QDRANT_URL,
    use_hybrid=USE_HYBRID,
    force_recreate=FORCE_RECREATE
)

info = store.collection_info()
print(f"\\nCollection '{COLLECTION_NAME}' — {info.points_count} vectors currently stored")\
"""),

md("""\
## 7 — Index Chunks

Embeds every chunk (dense + sparse) and upserts into Qdrant.  
Skips automatically if the collection already has data and `FORCE_RECREATE = False`."""),

code("""\
import time

if info.points_count > 0 and not FORCE_RECREATE:
    print(f"Collection already has {info.points_count} vectors. Skipping indexing.")
    print("Set FORCE_RECREATE = True in cell 6 and re-run to rebuild.")
else:
    print(f"Indexing {len(all_chunks)} chunks...")
    t0 = time.time()
    store.index_chunks(all_chunks, batch_size=32)
    elapsed = round(time.time() - t0, 1)
    print(f"\\nDone in {elapsed}s")

    info = store.collection_info()
    print(f"Vectors in collection: {info.points_count}")\
"""),

md("## 8 — Sanity Check: Run a Test Query"),

code("""\
TEST_QUERY = "What methods are used to measure runway friction?"

results = store.search(TEST_QUERY, top_k=5)

print(f"Query: {TEST_QUERY}\\n")
for i, r in enumerate(results, 1):
    print(f"[{i}] score={round(r['score'], 4)}  stem={r['metadata'].get('stem', '')[:50]}")
    print(f"    section: {r['metadata'].get('section_h2') or r['metadata'].get('section_h1', '')}")
    print(f"    preview : {r['text'][:120]}")
    print()\
"""),

md("## 9 — Collection Summary"),

code("""\
info = store.collection_info()

print("=" * 60)
print("COLLECTION SUMMARY")
print("=" * 60)
print(f"  Name          : {COLLECTION_NAME}")
print(f"  Vectors stored: {info.points_count}")
print(f"  Hybrid mode   : {USE_HYBRID}")
print(f"  Documents     : {len(stems)}")
print(f"  Chunks built  : {len(all_chunks)}")
print("=" * 60)
print("\\n✓ Index is ready. Open 02_retrieval.ipynb to run queries.")\
"""),

]

save(nb1, BASE / "01_indexing.ipynb")


# ─────────────────────────────────────────────────────────────────────────────
# 02_retrieval.ipynb  (reranking only, no comparison)
# ─────────────────────────────────────────────────────────────────────────────
nb2 = new_notebook()
nb2.cells = [

md("""\
# 02 — Retrieval & QA

Assumes the vector store is already populated — run `01_indexing.ipynb` first.

Uses **hybrid search + cross-encoder reranking** (`bge-reranker-base`).

Sections:
1. Imports & config
2. Initialise pipeline
3. Single interactive query
4. Query with metadata filters
5. Batch test — all categories
6. Summary statistics"""),

md("## 1 — Imports & Config"),

code("""\
import time
from pathlib import Path
from dotenv import load_dotenv

from embedders import LocalEmbedder
from indexer import VectorStore
from retriever_rerank import RAGPipelineWithReranking
from test_questions_comprehensive import test_questions, categories

load_dotenv()

BASE_DIR      = Path(".").resolve().parent  # project root
ANSWER_PROMPT = (BASE_DIR / "prompts/answer_rag_prompt").read_text(encoding="utf-8")
QA_MODEL      = "gpt-4.1-mini"

print(f"Total questions loaded: {len(test_questions)}")
print("\\nQuestions per category:")
for cat, qs in categories.items():
    print(f"  {cat}: {len(qs)}")\
"""),

md("## 2 — Initialise Pipeline"),

code("""\
embedder = LocalEmbedder()
store    = VectorStore(embedder=embedder, use_hybrid=True)
rag      = RAGPipelineWithReranking(store=store, answer_prompt=ANSWER_PROMPT, model=QA_MODEL)

info = store.collection_info()
print(f"\\nPipeline ready.  Collection has {info.points_count} vectors.")\
"""),

md("""\
## 3 — Single Interactive Query

Edit `QUESTION` and re-run this cell."""),

code("""\
QUESTION           = "What methods are used to measure runway friction?"
TOP_K              = 5
CANDIDATE_MULT     = 3   # fetch TOP_K × CANDIDATE_MULT candidates before reranking

t0 = time.time()
result = rag.ask(QUESTION, top_k=TOP_K, show_chunks=True, candidate_multiplier=CANDIDATE_MULT)
elapsed = round(time.time() - t0, 2)

print(f"Question: {QUESTION}")
print(f"Time: {elapsed}s")
print("=" * 72)
print(result["answer"])

print("\\nSources:")
for s in result["sources"]:
    print(f"  • [{round(s.get('rerank_score', 0), 3)}]  {s['citation']}")

print(f"\\nChunks retrieved (after reranking):")
for i, c in enumerate(result["chunks"], 1):
    print(f"  [{i}] rerank={round(c.get('rerank_score', 0), 3)}  orig={round(c.get('original_score', c.get('score', 0)), 3)}  "
          f"stem={c['metadata'].get('stem', '')[:45]}")\
"""),

md("""\
## 4 — Query with Metadata Filters

Restrict retrieval to a specific document or year.  
Available filter keys: `stem`, `year`, `doc_type`, `authors`"""),

code("""\
FILTERED_QUESTION = "What is the role of aggregate gradation in SMA performance?"
FILTERS           = {"stem": "014. AGGREGATE SPECIFICATIONS FOR STONE MASTIC ASPHALT (SMA)"}

t0 = time.time()
r = rag.ask(FILTERED_QUESTION, top_k=5, filters=FILTERS, show_chunks=True)
elapsed = round(time.time() - t0, 2)

print(f"Q      : {FILTERED_QUESTION}")
print(f"Filter : {FILTERS}")
print(f"Time   : {elapsed}s")
print("=" * 72)
print(r["answer"])

print("\\nSources:")
for s in r["sources"]:
    print(f"  • {s['citation']}")\
"""),

md("""\
## 5 — Batch Test

Runs every question through the reranking pipeline.  
Each question prints its answer, retrieved sources, and key-term coverage."""),

code("""\
def run_question(q: dict, top_k: int = 5, candidate_mult: int = 2) -> dict:
    sep = "=" * 72
    print(f"\\n{sep}")
    print(f"[{q['id']}]  {q['category']}  |  difficulty: {q.get('difficulty', '—')}")
    print(f"Q: {q['question']}")
    print(sep)

    t0 = time.time()
    r  = rag.ask(q["question"], top_k=top_k, show_chunks=True, candidate_multiplier=candidate_mult)
    elapsed = round(time.time() - t0, 2)

    docs   = sorted({c["metadata"].get("stem", "")[:40] for c in r["chunks"]})
    scores = [round(c.get("rerank_score", c.get("score", 0)), 3) for c in r["chunks"][:3]]

    print(f"\\n  time   : {elapsed}s")
    print(f"  docs   : {docs}")
    print(f"  scores : {scores}")
    print(f"  answer : {r['answer']}")

    terms   = q.get("answer_should_include", [])
    hits    = [t for t in terms if t.lower() in r["answer"].lower()]
    if terms:
        print(f"\\n  key terms ({len(terms)}): {terms}")
        print(f"  found ({len(hits)}): {hits}")

    return dict(
        id=q["id"],
        category=q["category"],
        elapsed=elapsed,
        docs=docs,
        answer=r["answer"],
        sources=r["sources"],
        terms_hit=len(hits),
        terms_total=len(terms),
    )

print("Helper defined.")"""),

md("### 5.1 — Single Document (specific)"),

code("""\
cat1 = [q for q in test_questions if q["category"] == "single_document_detailed"]
results_cat1 = [run_question(q, top_k=10) for q in cat1]\
"""),

md("### 5.2 — Multi-document"),

code("""\
cat2 = [q for q in test_questions if q["category"] == "multi_document_integration"]
results_cat2 = [run_question(q, top_k=7) for q in cat2]\
"""),

md("### 5.3 — Comparative Analysis"),

code("""\
cat3 = [q for q in test_questions if q["category"] == "comparative_analysis"]
results_cat3 = [run_question(q, top_k=10) for q in cat3]\
"""),

md("### 5.4 — Technical Terminology"),

code("""\
cat4 = [q for q in test_questions if q["category"] == "technical_terminology"]
results_cat4 = [run_question(q, top_k=10) for q in cat4]\
"""),

md("### 5.5 — Operational / Design / Methodology"),

code("""\
cat567 = [q for q in test_questions if q["category"] in
          ("operational_complexity", "design_construction", "methodology")]
results_cat567 = [run_question(q, top_k=10) for q in cat567]\
"""),

md("""\
### 5.6 — Out of Scope (negative test)

These questions should **not** be answered from the corpus.  
A low rerank score confirms the system correctly finds no relevant content."""),

code("""\
cat8 = [q for q in test_questions if q["category"] == "out_of_scope"]

results_cat8 = []
for q in cat8:
    print(f"\\n{'='*72}")
    print(f"[{q['id']}] {q['question']}")

    r = rag.ask(q["question"], top_k=3, show_chunks=True)
    top_score = max((c.get("rerank_score", c.get("score", 0)) for c in r["chunks"]), default=0)

    print(f"  top rerank score : {round(top_score, 3)}  (low = correctly filtered)")
    print(f"  answer           : {r['answer'][:300]}")
    results_cat8.append({"id": q["id"], "top_score": round(top_score, 3)})\
"""),

md("### 5.7 — Ambiguous / Complex Reasoning"),

code("""\
cat910 = [q for q in test_questions if q["category"] in ("ambiguous", "complex_reasoning")]
results_cat910 = [run_question(q, top_k=7) for q in cat910]\
"""),

md("## 6 — Summary"),

code("""\
all_results = (
    results_cat1 + results_cat2 + results_cat3 +
    results_cat4 + results_cat567 + results_cat910
)

avg_t        = round(sum(r["elapsed"]     for r in all_results) / len(all_results), 2)
total_terms  = sum(r["terms_total"] for r in all_results)
total_hits   = sum(r["terms_hit"]   for r in all_results)
coverage_pct = round(total_hits / total_terms * 100) if total_terms else 0

print("=" * 72)
print("OVERALL SUMMARY")
print("=" * 72)
print(f"  Questions run     : {len(all_results)}")
print(f"  Avg response time : {avg_t}s")
print(f"  Key-term coverage : {total_hits}/{total_terms}  ({coverage_pct}%)")
print("=" * 72)

print(f"\\n{'ID':<22} {'category':<28} {'time':>6} {'terms':>7}")
print("-" * 72)
for r in all_results:
    denom = r['terms_total'] if r['terms_total'] else 1
    print(f"  {r['id']:<20} {r['category']:<28} {r['elapsed']:>5}s  {r['terms_hit']:>2}/{denom}")\
"""),

]

save(nb2, BASE / "02_retrieval.ipynb")

print("\\nDone. Both notebooks generated successfully.")


