"""
extract_pdfs.py — Batch PDF extraction for production scale.

Replaces: notebooks/00_pdf_extraction.ipynb
Reuses:   scientific_rag.config (EXTRACT_PROMPT_PATH, RAW_DIR, MD_DIR, REF_DIR)

Runs Docling OCR + LLM metadata extraction in parallel.
Already-processed files (existing .json in ref_storage) are skipped automatically.

Usage:
    # All local (default)
    python full_load_scripts/extract_pdfs.py

    # PDFs in S3, outputs local
    python full_load_scripts/extract_pdfs.py \\
        --source s3 --source-bucket my-bucket --source-prefix raw/

    # PDFs in S3, outputs in S3
    python full_load_scripts/extract_pdfs.py \\
        --source s3 --source-bucket my-bucket --source-prefix raw/ \\
        --output s3 --output-bucket my-bucket --output-prefix processed/

    # More workers / reprocess one file
    python full_load_scripts/extract_pdfs.py --workers 8
    python full_load_scripts/extract_pdfs.py --reprocess "001. Some Doc"
"""

import sys
import json
import time
import argparse
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

from dotenv import load_dotenv
from loguru import logger
from tqdm import tqdm
from openai import OpenAI
from os import environ

from docling.datamodel.accelerator_options import AcceleratorDevice, AcceleratorOptions
from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import PdfPipelineOptions, TableStructureOptions
from docling.document_converter import DocumentConverter, PdfFormatOption

# --- Monkey-patch: allow URIs without scheme (e.g. "www.example.com/...") ---
# Docling's PdfHyperlink uses Pydantic AnyUrl which rejects bare hostnames.
# This causes entire page batches to fail during preprocessing.
# Fix: relax uri field to Optional[str].  (Tracked in docling-core PR #520)
from typing import Optional
from docling_core.types.doc.page import PdfHyperlink, SegmentedPdfPage
PdfHyperlink.__annotations__["uri"] = Optional[str]
PdfHyperlink.model_fields["uri"].annotation = Optional[str]
PdfHyperlink.model_rebuild(force=True)
SegmentedPdfPage.model_rebuild(force=True)
# ---------------------------------------------------------------------------

# ── Make package importable ───────────────────────────────────────────────────
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from scientific_rag.config import RAW_DIR, MD_DIR, REF_DIR, EXTRACT_PROMPT_PATH

from scientific_rag.full_load_scripts.storage import BaseStorage, build_storage

load_dotenv()

MODEL = "gpt-4.1-mini"


# ─────────────────────────────────────────────────────────────────────────────
# Docling converter
# ─────────────────────────────────────────────────────────────────────────────

def build_converter() -> DocumentConverter:
    """Configure Docling PDF converter with OCR and table extraction."""
    options = PdfPipelineOptions()
    options.do_ocr = True
    options.do_table_structure = True
    options.table_structure_options = TableStructureOptions(do_cell_matching=True)
    options.ocr_options.lang = ["en"]
    options.accelerator_options = AcceleratorOptions(
        num_threads=4,
        device=AcceleratorDevice.AUTO,
    )
    return DocumentConverter(
        format_options={InputFormat.PDF: PdfFormatOption(pipeline_options=options)}
    )


# ─────────────────────────────────────────────────────────────────────────────
# LLM metadata extraction
# ─────────────────────────────────────────────────────────────────────────────

def extract_metadata_llm(
    first_pages_md: str,
    prompt_template: str,
    client: OpenAI,
) -> dict:
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
# Single-PDF pipeline
# ─────────────────────────────────────────────────────────────────────────────

def process_one_pdf(
    stem: str,
    pdf_storage: BaseStorage,
    md_storage: BaseStorage,
    ref_storage: BaseStorage,
    converter: DocumentConverter,
    prompt_template: str,
    client: OpenAI,
) -> str:
    """
    Full pipeline for one PDF:
      1. Get local path to PDF (downloads from S3 if needed)
      2. OCR → full markdown
      3. First 5 pages → LLM → reference JSON
      4. Save both via storage abstraction

    Returns stem on success. Raises on error.
    """
    t0 = time.time()

    # Get a local file path (LocalStorage returns it directly; S3Storage downloads to cache)
    pdf_path = pdf_storage.get_local_path(stem, ".pdf")

    # OCR
    conv = converter.convert(pdf_path)
    full_markdown = conv.document.export_to_markdown()

    # Save markdown
    md_storage.write_text(stem, ".md", full_markdown)

    # Extract metadata from first 5 pages
    pages_md = "\n\n".join(
        conv.document.export_to_markdown(page_no=p)
        for p in sorted(conv.document.pages)
        if p <= 5
    )
    metadata = extract_metadata_llm(pages_md, prompt_template, client)

    # Save reference JSON
    ref_storage.write_text(stem, ".json", json.dumps(metadata, indent=2, ensure_ascii=False))

    elapsed = round(time.time() - t0, 1)
    logger.info(f"OK  {stem}  ({elapsed}s)")
    return stem


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(
        description="Batch PDF extraction (OCR + LLM metadata)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # All local
  python full_load_scripts/extract_pdfs.py

  # PDFs from S3, outputs to S3
  python full_load_scripts/extract_pdfs.py \\
      --source s3 --source-bucket docs --source-prefix raw/ \\
      --output s3 --output-bucket docs --output-prefix processed/

  # Reprocess a single document
  python full_load_scripts/extract_pdfs.py --reprocess "001. Doc Name"
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
    p.add_argument("--workers",     type=int, default=4,    help="Parallel workers (default: 4)")
    p.add_argument("--reprocess",   type=str, default=None, help="Force reprocess one stem")
    p.add_argument("--region",      type=str, default="us-east-1", help="AWS region")

    return p.parse_args()


def main():
    args = parse_args()

    # Build storage instances
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

    prompt_template = EXTRACT_PROMPT_PATH.read_text(encoding="utf-8")
    converter = build_converter()
    client = OpenAI(api_key=environ["OPENAI_API_KEY"], timeout=1500)

    # List all PDFs
    all_stems = pdf_storage.list_files(".pdf")
    logger.info(f"Found {len(all_stems)} PDFs in {pdf_storage}")

    # Filter: skip already processed unless --reprocess
    to_process = []
    skipped = 0
    for stem in all_stems:
        if ref_storage.exists(stem, ".json") and args.reprocess != stem:
            skipped += 1
        else:
            to_process.append(stem)

    logger.info(f"Skipping {skipped} already-processed documents")
    logger.info(f"Processing {len(to_process)} documents with {args.workers} workers")

    if not to_process:
        logger.info("Nothing to process. All PDFs already extracted.")
        return

    failed = []

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {
            pool.submit(
                process_one_pdf,
                stem, pdf_storage, md_storage, ref_storage,
                converter, prompt_template, client,
            ): stem
            for stem in to_process
        }
        with tqdm(total=len(to_process), desc="Extracting PDFs") as bar:
            for future in as_completed(futures):
                stem = futures[future]
                try:
                    future.result()
                except Exception as e:
                    logger.error(f"FAILED  {stem}: {e}")
                    failed.append(stem)
                finally:
                    bar.update(1)

    logger.info(f"Done. {len(to_process) - len(failed)} succeeded, {len(failed)} failed.")
    if failed:
        logger.warning("Failed files:")
        for f in failed:
            logger.warning(f"  {f}")


if __name__ == "__main__":
    main()

