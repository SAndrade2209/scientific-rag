"""
full_pipeline.py — Production-scale one-PDF-at-a-time full pipeline.

Processes each PDF end-to-end before moving to the next:
  1. OCR extraction → save markdown
  2. LLM metadata extraction → save JSON
  3. Chunking → embedding → upload to Qdrant
  4. Free RAM → next PDF

This keeps RAM constant regardless of corpus size.
Already-processed PDFs (checkpointed + in Qdrant) are skipped automatically.

Usage:
    # All local (default)
    python -m scientific_rag.full_load_scripts.full_pipeline

    # PDFs from S3, outputs to S3
    python -m scientific_rag.full_load_scripts.full_pipeline \\
        --source s3 --source-bucket my-bucket --source-prefix raw/ \\
        --output s3 --output-bucket my-bucket --output-prefix processed/

    # Reprocess a single document end-to-end
    python -m scientific_rag.full_load_scripts.full_pipeline --reprocess "015. Dual electromagnetic"

    # Force reprocess everything
    python -m scientific_rag.full_load_scripts.full_pipeline --force-all
"""

import gc
import json
import time
import argparse
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv
from loguru import logger
from tqdm import tqdm
from openai import OpenAI
from os import environ

# ── Docling ───────────────────────────────────────────────────────────────────
from docling.datamodel.accelerator_options import AcceleratorDevice, AcceleratorOptions
from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import PdfPipelineOptions, TableStructureOptions
from docling.document_converter import DocumentConverter, PdfFormatOption

# --- Monkey-patch: allow URIs without scheme (e.g. "www.example.com/...") ---
# Docling's PdfHyperlink uses Pydantic AnyUrl which rejects bare hostnames.
# This causes entire page batches to fail during preprocessing.
# Fix: relax uri field to Optional[str].  (Tracked in docling-core PR #520)
from docling_core.types.doc.page import PdfHyperlink, SegmentedPdfPage
PdfHyperlink.__annotations__["uri"] = Optional[str]
PdfHyperlink.model_fields["uri"].annotation = Optional[str]
PdfHyperlink.model_rebuild(force=True)
SegmentedPdfPage.model_rebuild(force=True)
# ---------------------------------------------------------------------------

# ── Project imports ───────────────────────────────────────────────────────────
from scientific_rag.config import (
    RAW_DIR, MD_DIR, REF_DIR, DATA_DIR,
    EXTRACT_PROMPT_PATH, MAX_CHUNK_SIZE, CHUNK_OVERLAP,
)
from scientific_rag.ingestion.chunker import flatten_metadata, chunk_document
from scientific_rag.vectorstore.embedders import LocalEmbedder
from scientific_rag.vectorstore.indexer import VectorStore
from scientific_rag.full_load_scripts.storage import BaseStorage, build_storage

load_dotenv()

MODEL = "gpt-4.1-mini"
CHECKPOINT_PATH = DATA_DIR / ".pipeline_checkpoint.json"


# ─────────────────────────────────────────────────────────────────────────────
# Checkpoint helpers
# ─────────────────────────────────────────────────────────────────────────────

def load_checkpoint() -> set:
    """Load the set of stems that have been fully processed (extracted + indexed)."""
    if CHECKPOINT_PATH.exists():
        data = json.loads(CHECKPOINT_PATH.read_text(encoding="utf-8"))
        return set(data.get("completed_stems", []))
    return set()


def save_checkpoint(completed_stems: set):
    """Persist the current set of fully completed stems."""
    CHECKPOINT_PATH.parent.mkdir(parents=True, exist_ok=True)
    CHECKPOINT_PATH.write_text(
        json.dumps({"completed_stems": sorted(completed_stems)}, indent=2),
        encoding="utf-8",
    )


# ─────────────────────────────────────────────────────────────────────────────
# Already-indexed stems from Qdrant
# ─────────────────────────────────────────────────────────────────────────────

def get_indexed_stems_from_qdrant(store: VectorStore) -> set:
    """Scroll Qdrant and collect every unique stem already stored."""
    stems = set()
    offset = None
    while True:
        results, offset = store.client.scroll(
            collection_name=store.collection_name,
            scroll_filter=None,
            limit=1000,
            offset=offset,
            with_payload=["stem"],
            with_vectors=False,
        )
        for point in results:
            stem = point.payload.get("stem")
            if stem:
                stems.add(stem)
        if offset is None:
            break
    return stems


# ─────────────────────────────────────────────────────────────────────────────
# Docling converter
# ─────────────────────────────────────────────────────────────────────────────

def build_converter(num_threads: int = 4) -> DocumentConverter:
    """Configure Docling PDF converter with OCR and table extraction."""
    options = PdfPipelineOptions()
    options.do_ocr = True
    options.do_table_structure = True
    options.table_structure_options = TableStructureOptions(do_cell_matching=True)
    options.ocr_options.lang = ["en"]
    options.accelerator_options = AcceleratorOptions(
        num_threads=num_threads,
        device=AcceleratorDevice.AUTO,
    )
    return DocumentConverter(
        format_options={InputFormat.PDF: PdfFormatOption(pipeline_options=options)}
    )


# ─────────────────────────────────────────────────────────────────────────────
# LLM metadata extraction
# ─────────────────────────────────────────────────────────────────────────────

def extract_metadata_llm(first_pages_md: str, prompt_template: str, client: OpenAI) -> dict:
    """Send the first 5 pages to the LLM and extract structured reference metadata."""
    prompt = prompt_template.replace("{OCR_TEXT}", first_pages_md)
    response = client.responses.create(
        model=MODEL,
        input=prompt,
        temperature=0,
        text={"format": {"type": "json_object"}},
    )
    return json.loads(response.output_text)


# ─────────────────────────────────────────────────────────────────────────────
# Single-PDF full pipeline
# ─────────────────────────────────────────────────────────────────────────────

def process_one_pdf_full(
    stem: str,
    pdf_storage: BaseStorage,
    md_storage: BaseStorage,
    ref_storage: BaseStorage,
    converter: DocumentConverter,
    prompt_template: str,
    client: OpenAI,
    store: VectorStore,
    max_chunk_size: int,
    chunk_overlap: int,
    batch_size: int,
) -> int:
    """
    Full end-to-end pipeline for one PDF:
      1. OCR → full markdown → save
      2. First 5 pages → LLM → reference JSON → save
      3. Chunk → embed → upload to Qdrant

    Returns the number of chunks indexed.
    Raises on error.
    """
    t0 = time.time()
    conv = full_md = pages_md = metadata = doc_meta = chunks = None

    try:
        # Step 1: OCR extraction
        pdf_path = pdf_storage.get_local_path(stem, ".pdf")
        conv = converter.convert(pdf_path)
        full_md = conv.document.export_to_markdown()
        logger.info(f"OCR done: {stem} ({len(full_md)} chars)")

        # Step 2: Save markdown
        md_storage.write_text(stem, ".md", full_md)

        # Step 3: LLM metadata extraction
        pages_md = "\n\n".join(
            conv.document.export_to_markdown(page_no=p)
            for p in sorted(conv.document.pages) if p <= 5
        )
        metadata = extract_metadata_llm(pages_md, prompt_template, client)

        # Step 4: Save metadata
        ref_storage.write_text(stem, ".json", json.dumps(metadata, indent=2, ensure_ascii=False))
        logger.info(f"Metadata saved: {metadata.get('title', '?')}")

        # Step 5: Chunk
        doc_meta = flatten_metadata(stem, metadata)
        chunks = chunk_document(stem, full_md, doc_meta, max_chunk_size, chunk_overlap)
        logger.info(f"Chunks: {len(chunks)}")

        # Step 6: Embed & upload to Qdrant
        n_chunks = len(chunks) if chunks else 0
        if chunks:
            store.index_chunks(chunks, batch_size=batch_size)

        elapsed = round(time.time() - t0, 1)
        logger.info(f"✅ {stem} — {n_chunks} chunks indexed in {elapsed}s")
        return n_chunks

    finally:
        # Free RAM — delete all heavy objects regardless of success/failure
        del conv, full_md, pages_md, metadata, doc_meta, chunks
        gc.collect()


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(
        description="Full pipeline: OCR → metadata → chunk → embed → Qdrant (one PDF at a time, RAM-friendly)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # All local
  python -m scientific_rag.full_load_scripts.full_pipeline

  # PDFs from S3, outputs to S3
  python -m scientific_rag.full_load_scripts.full_pipeline \\
      --source s3 --source-bucket docs --source-prefix raw/ \\
      --output s3 --output-bucket docs --output-prefix processed/

  # Reprocess one specific document
  python -m scientific_rag.full_load_scripts.full_pipeline \\
      --reprocess "015. Dual electromagnetic energy harvesting"

  # Force re-run everything
  python -m scientific_rag.full_load_scripts.full_pipeline --force-all
        """,
    )

    # Source (where PDFs live)
    p.add_argument("--source",         choices=["local", "s3"], default="local")
    p.add_argument("--source-bucket",  type=str, default=None)
    p.add_argument("--source-prefix",  type=str, default="raw/")
    p.add_argument("--source-dir",     type=str, default=None,
                   help=f"Local PDF directory (default: {RAW_DIR})")

    # Output (where .md and .json go)
    p.add_argument("--output",         choices=["local", "s3"], default="local")
    p.add_argument("--output-bucket",  type=str, default=None)
    p.add_argument("--output-md-prefix",  type=str, default="processed/markdown/")
    p.add_argument("--output-ref-prefix", type=str, default="processed/metadata/")
    p.add_argument("--output-md-dir",  type=str, default=None,
                   help=f"Local markdown output dir (default: {MD_DIR})")
    p.add_argument("--output-ref-dir", type=str, default=None,
                   help=f"Local metadata output dir (default: {REF_DIR})")

    # Processing
    p.add_argument("--batch-size",  type=int, default=64,   help="Qdrant upsert batch size (default: 64)")
    p.add_argument("--chunk-size",  type=int, default=None, help=f"Chunk size (default: {MAX_CHUNK_SIZE})")
    p.add_argument("--overlap",     type=int, default=None, help=f"Chunk overlap (default: {CHUNK_OVERLAP})")
    p.add_argument("--threads",     type=int, default=4,    help="Docling OCR threads (default: 4)")
    p.add_argument("--reprocess",   type=str, default=None, help="Force reprocess one stem")
    p.add_argument("--force-all",   action="store_true",    help="Re-run everything from scratch")
    p.add_argument("--region",      type=str, default="us-east-1", help="AWS region")

    return p.parse_args()


def main():
    args = parse_args()

    max_chunk = args.chunk_size or MAX_CHUNK_SIZE
    overlap   = args.overlap or CHUNK_OVERLAP

    # ── Build storage instances ───────────────────────────────────────────────
    pdf_storage = build_storage(
        source=args.source,
        local_dir=Path(args.source_dir) if args.source_dir else RAW_DIR,
        bucket=args.source_bucket,
        prefix=args.source_prefix,
        region=args.region,
    )
    md_storage = build_storage(
        source=args.output,
        local_dir=Path(args.output_md_dir) if args.output_md_dir else MD_DIR,
        bucket=args.output_bucket,
        prefix=args.output_md_prefix,
        region=args.region,
    )
    ref_storage = build_storage(
        source=args.output,
        local_dir=Path(args.output_ref_dir) if args.output_ref_dir else REF_DIR,
        bucket=args.output_bucket,
        prefix=args.output_ref_prefix,
        region=args.region,
    )

    logger.info(f"PDF source : {pdf_storage}")
    logger.info(f"MD output  : {md_storage}")
    logger.info(f"REF output : {ref_storage}")

    # ── Build heavy objects once ──────────────────────────────────────────────
    prompt_template = EXTRACT_PROMPT_PATH.read_text(encoding="utf-8")
    converter = build_converter(num_threads=args.threads)
    client = OpenAI(api_key=environ["OPENAI_API_KEY"], timeout=1500)
    embedder = LocalEmbedder()
    store = VectorStore(embedder=embedder, use_hybrid=True)

    info = store.collection_info()
    logger.info(f"Collection '{store.collection_name}': {info.points_count} vectors")

    # ── Determine what to skip ────────────────────────────────────────────────
    if args.force_all:
        logger.warning("--force-all: re-processing everything from scratch.")
        already_done = set()
        save_checkpoint(set())
    else:
        from_qdrant = get_indexed_stems_from_qdrant(store)
        from_checkpoint = load_checkpoint()
        already_done = from_qdrant | from_checkpoint
        logger.info(f"Already completed: {len(already_done)} stems")

    # ── List all PDFs ─────────────────────────────────────────────────────────
    all_stems = pdf_storage.list_files(".pdf")
    logger.info(f"Found {len(all_stems)} PDFs in {pdf_storage}")

    # ── Filter ────────────────────────────────────────────────────────────────
    to_process = []
    for stem in all_stems:
        if args.reprocess == stem or stem not in already_done:
            to_process.append(stem)

    logger.info(f"Skipping {len(all_stems) - len(to_process)} already-completed documents")
    logger.info(f"Processing {len(to_process)} documents (one at a time, RAM-friendly)")

    if not to_process:
        logger.info("Nothing to process. All PDFs already extracted and indexed.")
        return

    # ── Process ───────────────────────────────────────────────────────────────
    completed = set(already_done)
    total_chunks = 0
    failed = []

    with tqdm(to_process, desc="Full pipeline") as bar:
        for stem in bar:
            bar.set_postfix(stem=stem[:40])
            try:
                n = process_one_pdf_full(
                    stem=stem,
                    pdf_storage=pdf_storage,
                    md_storage=md_storage,
                    ref_storage=ref_storage,
                    converter=converter,
                    prompt_template=prompt_template,
                    client=client,
                    store=store,
                    max_chunk_size=max_chunk,
                    chunk_overlap=overlap,
                    batch_size=args.batch_size,
                )
                total_chunks += n
                completed.add(stem)
                save_checkpoint(completed)

            except Exception as e:
                logger.error(f"❌ FAILED {stem}: {e}")
                failed.append(stem)

    # ── Summary ───────────────────────────────────────────────────────────────
    info = store.collection_info()
    logger.info(f"{'='*60}")
    logger.info(f"DONE")
    logger.info(f"  Succeeded    : {len(to_process) - len(failed)}")
    logger.info(f"  Failed       : {len(failed)}")
    logger.info(f"  Chunks added : {total_chunks}")
    logger.info(f"  Total vectors: {info.points_count}")
    logger.info(f"{'='*60}")

    if failed:
        logger.warning("Failed documents:")
        for f in failed:
            logger.warning(f"  {f}")


if __name__ == "__main__":
    main()

