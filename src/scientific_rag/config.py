"""
config.py — Centralised project paths and default settings.

All other modules import paths from here so that moving the project
only requires updating this one file.
"""

from pathlib import Path
from dotenv import load_dotenv
from os import environ
import warnings

load_dotenv()

# ── Project root (two levels up from this file: src/scientific_rag/config.py)
PROJECT_ROOT = Path(__file__).resolve().parents[2]

# ── Data directories
DATA_DIR       = PROJECT_ROOT / "data"
RAW_DIR        = DATA_DIR / "raw"                     # original PDFs
MD_DIR         = DATA_DIR / "processed" / "markdown"  # converted markdown
REF_DIR        = DATA_DIR / "processed" / "metadata"  # reference JSON files

# ── Prompts
PROMPTS_DIR        = Path(__file__).parent / "prompts"
ANSWER_PROMPT_PATH          = PROMPTS_DIR / "answer_rag_prompt"
EXTRACT_PROMPT_PATH         = PROMPTS_DIR / "extract_references_prompt"
QUERY_EXPANSION_PROMPT_PATH      = PROMPTS_DIR / "query_expansion_prompt"
QUERY_EXPANSION_CONV_PROMPT_PATH = PROMPTS_DIR / "query_expansion_conversational_prompt"
GUARDRAIL_PROMPT_PATH            = PROMPTS_DIR / "guardrail_prompt"

# ── Qdrant defaults
QDRANT_URL      = "http://localhost:6333"
COLLECTION_NAME = "technical_docs"

# ── Chunking defaults
MAX_CHUNK_SIZE = 1000
CHUNK_OVERLAP  = 100

# ── LLM defaults
QA_MODEL = "gpt-4.1-mini"

OPENAI_API_KEY = environ.get("OPENAI_API_KEY", "")
if not OPENAI_API_KEY:
    warnings.warn(
        "OPENAI_API_KEY is not set. LLM-dependent features will not work. "
        "Copy .env.example to .env and add your key.",
        stacklevel=2,
    )
