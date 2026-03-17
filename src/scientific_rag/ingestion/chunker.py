import json
from loguru import logger
from langchain_text_splitters import MarkdownHeaderTextSplitter, RecursiveCharacterTextSplitter

from ..config import MD_DIR, REF_DIR


# ---------------- LOADERS ---------------- #

def load_markdown(stem: str) -> str | None:
    """
    Reads the full markdown text for a given document stem (filename without extension).
    Returns None if the file doesn't exist so the pipeline can skip it gracefully.
    """
    path = MD_DIR / f"{stem}.md"
    if not path.exists():
        logger.warning(f"No markdown found for: {stem}")
        return None
    return path.read_text(encoding="utf-8")


def load_metadata(stem: str) -> dict | None:
    """
    Reads the reference JSON for a given document stem.
    This is the structured metadata your LLM already extracted:
    title, authors, year, type, doi, etc.
    Returns None if the file doesn't exist.
    """
    path = REF_DIR / f"{stem}.json"
    if not path.exists():
        logger.warning(f"No metadata found for: {stem}")
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def get_all_stems() -> list[str]:
    """
    Scans md_docs/ and returns the stem of every markdown file.
    This drives the indexing loop — one stem = one document.
    """
    return [p.stem for p in sorted(MD_DIR.glob("*.md"))]


# ---------------- METADATA FLATTENING ---------------- #

def format_authors_apa(authors):
    if not authors:
        return None

    formatted = []
    for a in authors:
        family = (a.get("family") or "").strip()
        given = (a.get("given") or "").strip()

        if family and given:
            formatted.append(f"{family}, {given[0]}.")
        elif family:
            formatted.append(family)

    return ", ".join(formatted)


def build_citation(meta: dict) -> str | None:
    authors = format_authors_apa(meta.get("authors"))
    year = meta.get("year")
    title = meta.get("title")
    journal = meta.get("journal")
    doi = meta.get("doi") or meta.get("url")

    parts = []

    if authors:
        parts.append(authors)

    if year:
        parts.append(f"({year})")

    if title:
        parts.append(title)

    if journal:
        parts.append(journal)

    if doi:
        parts.append(doi)

    return ". ".join(parts) if parts else None


def flatten_metadata(stem: str, meta: dict) -> dict:
    """
    Qdrant stores metadata as a flat key-value payload alongside each vector.
    Your reference JSON has nested fields (e.g. authors is a list of dicts),
    which Qdrant can store but is awkward to filter on.

    We flatten it here into simple scalar fields that are easy to filter and display:
      - authors_str: "Yager, Thomas J.; Smith, John" (readable string)
      - year, title, type, doi, url as direct fields
      - stem: the filename without extension (links back to the source files)

    Anything that's None gets dropped so Qdrant payloads stay clean.
    """
    authors_str = format_authors_apa(meta.get("authors"))
    citation = build_citation(meta)

    flat = {
        "stem": stem,
        "doc_type": meta.get("type"),
        "title": meta.get("title"),
        "year": meta.get("year"),
        "journal": meta.get("journal"),
        "publisher": meta.get("publisher"),
        "doi": meta.get("doi"),
        "url": meta.get("url"),
        "authors": authors_str,
        "authors_raw": meta.get("authors"),
        "citation": citation.replace(".. (", ". ("),
    }

    # Drop None values — keeps Qdrant payloads clean
    return {k: v for k, v in flat.items() if v is not None}


# ---------------- CHUNKING ---------------- #

def chunk_document(stem: str, md_text: str, doc_metadata: dict, max_chunk_size: int = 1000, chunk_overlap: int = 100) -> list[dict]:
    """
    Splits one document's markdown into chunks ready for embedding.

    Two-stage process:
    ─────────────────
    Stage 1 — Header splitting:
        Splits by markdown headers (#, ##, ###).
        Each chunk inherits its section headers as metadata.
        This preserves document structure — you'll know a chunk came from
        the "Methodology" section of a specific paper, for example.

    Stage 2 — Size capping:
        Some sections are very long (dense methods, appendices).
        We cap at max_chunk_size characters with a small overlap so sentences
        aren't cut mid-thought at the boundary.

    Each final chunk is a dict with:
        "text"     → the actual content to embed
        "metadata" → everything Qdrant will store: doc info + section headers
    """

    # Stage 1: split by markdown headers
    header_splitter = MarkdownHeaderTextSplitter(
        headers_to_split_on=[
            ("#",   "section_h1"),
            ("##",  "section_h2"),
            ("###", "section_h3"),
        ],
        strip_headers=False  # keep headers inside the chunk text so context isn't lost
    )
    header_chunks = header_splitter.split_text(md_text)

    # Stage 2: cap size on any chunks that are still too large
    char_splitter = RecursiveCharacterTextSplitter(
        chunk_size=max_chunk_size,
        chunk_overlap=chunk_overlap,
    )

    final_chunks = []
    for i, chunk in enumerate(header_chunks):
        sub_chunks = char_splitter.split_text(chunk.page_content)

        for j, sub_text in enumerate(sub_chunks):
            # Skip empty or whitespace-only chunks
            if not sub_text.strip():
                continue

            final_chunks.append({
                "text": sub_text,
                "metadata": {
                    **doc_metadata,              # title, authors, year, doi, etc.
                    **chunk.metadata,            # section_h1, section_h2, section_h3
                    "chunk_index": f"{i}-{j}",  # useful for debugging retrieval issues
                }
            })

    return final_chunks


# ---------------- MAIN ENTRY POINT ---------------- #

def load_and_chunk_all(max_chunk_size: int = 1000, chunk_overlap: int = 100) -> list[dict]:
    """
    Iterates over every document in md_docs/, loads its markdown and metadata,
    and returns a flat list of all chunks across all documents.

    This is what the indexing script will call.
    """
    all_chunks = []
    stems = get_all_stems()
    logger.info(f"Found {len(stems)} documents to chunk")

    for stem in stems:
        md_text = load_markdown(stem)
        meta = load_metadata(stem)

        if md_text is None or meta is None:
            logger.warning(f"Skipping {stem} — missing markdown or metadata")
            continue

        doc_metadata = flatten_metadata(stem, meta)
        chunks = chunk_document(stem, md_text, doc_metadata, max_chunk_size, chunk_overlap)

        logger.info(f"{stem} → {len(chunks)} chunks")
        all_chunks.extend(chunks)

    logger.info(f"Total chunks across all documents: {len(all_chunks)}")
    return all_chunks