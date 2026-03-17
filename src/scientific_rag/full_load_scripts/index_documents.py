"""
Reuses:   chunker.flatten_metadata, chunker.chunk_document,
          embedders.LocalEmbedder, indexer.VectorStore

Reads processed markdowns and metadata from local disk or S3,
chunks them, and indexes into the local Qdrant instance.

Features:
  - Skips already-indexed stems (checks Qdrant payloads)
  - Checkpoint file for resumable runs
  - Streams one document at a time — constant RAM
  - tqdm progress bar

Usage:
    # All local (default — reads from data/processed/)
    python full_load_scripts/index_documents.py

    # Read processed files from S3
    python full_load_scripts/index_documents.py \\
        --source s3 --source-bucket my-bucket

    # Custom batch size / re-index one doc
    python full_load_scripts/index_documents.py --batch-size 128
    python full_load_scripts/index_documents.py --reindex "001. Some Doc"
    python full_load_scripts/index_documents.py --force-all
"""

import sys
import json
import argparse
from pathlib import Path

from dotenv import load_dotenv
from loguru import logger
from tqdm import tqdm

# ── Make package importable ───────────────────────────────────────────────────
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from scientific_rag.ingestion.chunker import flatten_metadata, chunk_document
from scientific_rag.vectorstore.embedders import LocalEmbedder
from scientific_rag.vectorstore.indexer import VectorStore
from scientific_rag.config import MD_DIR, REF_DIR, DATA_DIR, MAX_CHUNK_SIZE, CHUNK_OVERLAP

from scientific_rag.full_load_scripts.storage import build_storage, BaseStorage

load_dotenv()

CHECKPOINT_PATH = DATA_DIR / ".index_checkpoint.json"


# ─────────────────────────────────────────────────────────────────────────────
# Checkpoint helpers
# ─────────────────────────────────────────────────────────────────────────────

def load_checkpoint() -> set:
    """Load the set of stems that have been successfully indexed."""
    if CHECKPOINT_PATH.exists():
        data = json.loads(CHECKPOINT_PATH.read_text(encoding="utf-8"))
        return set(data.get("indexed_stems", []))
    return set()


def save_checkpoint(indexed_stems: set):
    """Persist the current set of indexed stems."""
    CHECKPOINT_PATH.parent.mkdir(parents=True, exist_ok=True)
    CHECKPOINT_PATH.write_text(
        json.dumps({"indexed_stems": sorted(indexed_stems)}, indent=2),
        encoding="utf-8",
    )


# ─────────────────────────────────────────────────────────────────────────────
# Already-indexed stems from Qdrant
# ─────────────────────────────────────────────────────────────────────────────

def get_indexed_stems_from_qdrant(store: VectorStore) -> set:
    """
    Scroll through Qdrant and collect every unique stem already stored.
    This is the ground truth — checkpoint is a fast cache on top.
    """
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
# Chunk one document from storage
# ─────────────────────────────────────────────────────────────────────────────

def chunk_from_storage(
    stem: str,
    md_storage: BaseStorage,
    ref_storage: BaseStorage,
    max_chunk_size: int,
    chunk_overlap: int,
) -> list[dict]:
    """
    Read a single document's markdown and metadata from storage,
    then chunk it. Returns the list of chunk dicts ready for indexing.
    """
    if not md_storage.exists(stem, ".md"):
        logger.warning(f"No markdown found for: {stem}")
        return []

    if not ref_storage.exists(stem, ".json"):
        logger.warning(f"No metadata found for: {stem}")
        return []

    md_text = md_storage.read_text(stem, ".md")
    metadata = json.loads(ref_storage.read_text(stem, ".json"))

    doc_metadata = flatten_metadata(stem, metadata)
    return chunk_document(stem, md_text, doc_metadata, max_chunk_size, chunk_overlap)


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(
        description="Index processed documents into Qdrant",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # All local
  python full_load_scripts/index_documents.py

  # Read processed files from S3
  python full_load_scripts/index_documents.py \\
      --source s3 --source-bucket my-bucket

  # Force re-index everything
  python full_load_scripts/index_documents.py --force-all
        """,
    )

    # Source (where processed .md and .json live)
    p.add_argument("--source",         choices=["local", "s3"], default="local")
    p.add_argument("--source-bucket",  type=str, default=None)
    p.add_argument("--source-md-prefix",  type=str, default="processed/markdown/")
    p.add_argument("--source-ref-prefix", type=str, default="processed/metadata/")
    p.add_argument("--source-md-dir",  type=str, default=None,
                   help=f"Local markdown dir (default: {MD_DIR})")
    p.add_argument("--source-ref-dir", type=str, default=None,
                   help=f"Local metadata dir (default: {REF_DIR})")

    # Indexing
    p.add_argument("--batch-size",  type=int, default=64,   help="Upsert batch size (default: 64)")
    p.add_argument("--chunk-size",  type=int, default=None, help=f"Chunk size (default: {MAX_CHUNK_SIZE})")
    p.add_argument("--overlap",     type=int, default=None, help=f"Chunk overlap (default: {CHUNK_OVERLAP})")
    p.add_argument("--reindex",     type=str, default=None, help="Force re-index one stem")
    p.add_argument("--force-all",   action="store_true",    help="Re-index everything")
    p.add_argument("--region",      type=str, default="us-east-1", help="AWS region")

    return p.parse_args()


def main():
    args = parse_args()

    max_chunk = args.chunk_size or MAX_CHUNK_SIZE
    overlap   = args.overlap or CHUNK_OVERLAP

    # Build storage for reading processed files
    md_storage = build_storage(
        source=args.source,
        local_dir=Path(args.source_md_dir) if args.source_md_dir else MD_DIR,
        bucket=args.source_bucket,
        prefix=args.source_md_prefix,
        region=args.region,
    )
    ref_storage = build_storage(
        source=args.source,
        local_dir=Path(args.source_ref_dir) if args.source_ref_dir else REF_DIR,
        bucket=args.source_bucket,
        prefix=args.source_ref_prefix,
        region=args.region,
    )

    logger.info(f"MD source  : {md_storage}")
    logger.info(f"REF source : {ref_storage}")

    # Init embedder + vector store (connects to existing Qdrant collection)
    embedder = LocalEmbedder()
    store = VectorStore(embedder=embedder, use_hybrid=True)

    info = store.collection_info()
    logger.info(f"Collection '{store.collection_name}': {info.points_count} vectors")

    # Determine what to skip
    if args.force_all:
        logger.warning("--force-all: re-indexing everything.")
        already_indexed = set()
        save_checkpoint(set())
    else:
        logger.info("Checking which stems are already indexed...")
        from_qdrant = get_indexed_stems_from_qdrant(store)
        from_checkpoint = load_checkpoint()
        already_indexed = from_qdrant | from_checkpoint
        logger.info(f"Already indexed: {len(already_indexed)} stems")

    # List available stems from storage
    all_stems = md_storage.list_files(".md")
    logger.info(f"Available stems in storage: {len(all_stems)}")

    # Filter
    to_index = []
    for stem in all_stems:
        if args.reindex == stem or stem not in already_indexed:
            to_index.append(stem)

    logger.info(f"Stems to index: {len(to_index)} / {len(all_stems)} total")

    if not to_index:
        logger.info("Nothing to index. All documents already in Qdrant.")
        return

    indexed_ok = set(already_indexed)
    total_chunks = 0
    failed = []

    with tqdm(to_index, desc="Indexing documents") as bar:
        for stem in bar:
            bar.set_postfix(stem=stem[:40])
            try:
                chunks = chunk_from_storage(stem, md_storage, ref_storage, max_chunk, overlap)
                if not chunks:
                    continue

                store.index_chunks(chunks, batch_size=args.batch_size)
                indexed_ok.add(stem)
                total_chunks += len(chunks)

                # Checkpoint after every document
                save_checkpoint(indexed_ok)

            except Exception as e:
                logger.error(f"FAILED  {stem}: {e}")
                failed.append(stem)

    info = store.collection_info()
    logger.info(f"Done. Chunks indexed this run: {total_chunks}")
    logger.info(f"Collection now has {info.points_count} vectors total")

    if failed:
        logger.warning(f"{len(failed)} documents failed:")
        for f in failed:
            logger.warning(f"  {f}")


if __name__ == "__main__":
    main()

