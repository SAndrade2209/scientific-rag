"""
session_store.py — Persistent session storage for conversational memory.
Abstracts where session data lives so the same ChatEngine works with:
  - SQLite   (local development, zero infrastructure)
  - Redis    (production / AWS ElastiCache, auto-expiring sessions)
Each session stores:
  - messages:   full chat history [{role, content}, ...]
  - summary:    LangChain's moving_summary_buffer (compressed older turns)
  - buffer:     recent HumanMessage/AIMessage not yet summarised
  - debug_logs: per-turn debug info (guardrail, expansion, sources, logs)
  - created_at / updated_at timestamps
Usage:
    # Local development
    store = SQLiteSessionStore()
    # Production (Redis)
    store = RedisSessionStore(url="redis://localhost:6379")
    # Factory
    store = build_session_store()
"""
import json
import sqlite3
from abc import ABC, abstractmethod
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional
from loguru import logger
# ─────────────────────────────────────────────────────────────────────────────
# Session data schema
# ─────────────────────────────────────────────────────────────────────────────
def empty_session() -> dict:
    """Return a blank session dict with all expected keys."""
    now = datetime.now(timezone.utc).isoformat()
    return {
        "messages":   [],
        "summary":    "",
        "buffer":     [],
        "debug_logs": [],
        "created_at": now,
        "updated_at": now,
    }
# ─────────────────────────────────────────────────────────────────────────────
# Abstract base
# ─────────────────────────────────────────────────────────────────────────────
class BaseSessionStore(ABC):
    """
    Interface for session persistence backends.
    All methods use a string ``session_id`` as key.
    Data is always a plain dict (JSON-serializable).
    """
    @abstractmethod
    def save(self, session_id: str, data: dict) -> None:
        """Persist session data. Overwrites if exists."""
    @abstractmethod
    def load(self, session_id: str) -> Optional[dict]:
        """Load session data. Returns None if not found."""
    @abstractmethod
    def delete(self, session_id: str) -> None:
        """Delete a session."""
    @abstractmethod
    def exists(self, session_id: str) -> bool:
        """Check if a session exists."""
    def load_or_create(self, session_id: str) -> dict:
        """Load existing session or create a blank one."""
        data = self.load(session_id)
        if data is None:
            data = empty_session()
            self.save(session_id, data)
            logger.info(f"Created new session: {session_id}")
        return data
# ─────────────────────────────────────────────────────────────────────────────
# SQLite backend (local development)
# ─────────────────────────────────────────────────────────────────────────────
class SQLiteSessionStore(BaseSessionStore):
    """
    Stores sessions in a local SQLite database.
    Zero infrastructure — just a file on disk.
    Perfect for local development and single-user Streamlit.
    Args:
        db_path: Path to the SQLite database file.
                 Defaults to data/sessions.db
    """
    def __init__(self, db_path: Optional[Path] = None):
        if db_path is None:
            from ..config import DATA_DIR
            db_path = DATA_DIR / "sessions.db"
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()
        logger.info(f"SQLiteSessionStore ready: {self.db_path}")

    def _init_db(self):
        """Create the sessions table if it doesn't exist."""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS sessions (
                    session_id  TEXT PRIMARY KEY,
                    data        TEXT NOT NULL,
                    created_at  TEXT NOT NULL,
                    updated_at  TEXT NOT NULL
                )
            """)
            conn.commit()

    def _connect(self) -> sqlite3.Connection:
        return sqlite3.connect(self.db_path)

    def save(self, session_id: str, data: dict) -> None:
        now = datetime.now(timezone.utc).isoformat()
        data["updated_at"] = now
        json_str = json.dumps(data, ensure_ascii=False)
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO sessions (session_id, data, created_at, updated_at)
                VALUES (?, ?, ?, ?)
                ON CONFLICT(session_id) DO UPDATE SET
                    data = excluded.data,
                    updated_at = excluded.updated_at
                """,
                (session_id, json_str, data.get("created_at", now), now),
            )
            conn.commit()

    def load(self, session_id: str) -> Optional[dict]:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT data FROM sessions WHERE session_id = ?",
                (session_id,),
            ).fetchone()
        if row is None:
            return None
        return json.loads(row[0])

    def delete(self, session_id: str) -> None:
        with self._connect() as conn:
            conn.execute("DELETE FROM sessions WHERE session_id = ?", (session_id,))
            conn.commit()
        logger.info(f"Deleted session: {session_id}")

    def exists(self, session_id: str) -> bool:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT 1 FROM sessions WHERE session_id = ? LIMIT 1",
                (session_id,),
            ).fetchone()
        return row is not None

    def list_sessions(self) -> list[dict]:
        """List all sessions with metadata (for debugging)."""
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT session_id, created_at, updated_at FROM sessions ORDER BY updated_at DESC"
            ).fetchall()
        return [
            {"session_id": r[0], "created_at": r[1], "updated_at": r[2]}
            for r in rows
        ]

    def __repr__(self):
        return f"SQLiteSessionStore({self.db_path})"
# ─────────────────────────────────────────────────────────────────────────────
# Redis backend (production / AWS)
# ─────────────────────────────────────────────────────────────────────────────
class RedisSessionStore(BaseSessionStore):
    """
    Stores sessions in Redis with automatic TTL expiration.
    Ideal for production on AWS (ElastiCache) or any Redis instance.
    Sessions expire automatically after ``ttl_seconds``.
    Args:
        url:          Redis connection URL (e.g. "redis://localhost:6379")
        ttl_seconds:  Session time-to-live in seconds (default: 24h)
        key_prefix:   Prefix for Redis keys (namespace isolation)
    """
    def __init__(
        self,
        url: str = "redis://localhost:6379",
        ttl_seconds: int = 86400,
        key_prefix: str = "rag:session:",
    ):
        import redis
        self.client = redis.from_url(url, decode_responses=True)
        self.ttl = ttl_seconds
        self.prefix = key_prefix
        # Test connection
        self.client.ping()
        logger.info(f"RedisSessionStore ready: {url} (TTL={ttl_seconds}s)")

    def _key(self, session_id: str) -> str:
        return f"{self.prefix}{session_id}"

    def save(self, session_id: str, data: dict) -> None:
        now = datetime.now(timezone.utc).isoformat()
        data["updated_at"] = now
        json_str = json.dumps(data, ensure_ascii=False)
        self.client.setex(self._key(session_id), self.ttl, json_str)

    def load(self, session_id: str) -> Optional[dict]:
        raw = self.client.get(self._key(session_id))
        if raw is None:
            return None
        return json.loads(raw)

    def delete(self, session_id: str) -> None:
        self.client.delete(self._key(session_id))
        logger.info(f"Deleted session: {session_id}")

    def exists(self, session_id: str) -> bool:
        return bool(self.client.exists(self._key(session_id)))

    def __repr__(self):
        return f"RedisSessionStore(prefix={self.prefix}, ttl={self.ttl}s)"
# ─────────────────────────────────────────────────────────────────────────────
# Memory serialization helpers
# ─────────────────────────────────────────────────────────────────────────────
def serialize_memory(memory) -> dict:
    """
    Extract the serializable state from a LangChain
    ConversationSummaryBufferMemory instance.
    Returns a dict with:
      - summary: the compressed summary of older turns
      - buffer:  list of recent messages [{role, content}]
    """
    summary = getattr(memory, "moving_summary_buffer", "") or ""
    buffer = []
    if hasattr(memory, "chat_memory") and hasattr(memory.chat_memory, "messages"):
        for msg in memory.chat_memory.messages:
            role = "user" if msg.type == "human" else "assistant"
            buffer.append({"role": role, "content": msg.content})
    return {"summary": summary, "buffer": buffer}

def deserialize_memory(data: dict, api_key: str, model: str = "gpt-4.1-mini", max_token_limit: int = 1000):
    """
    Rebuild a LangChain ConversationSummaryBufferMemory from serialized data.
    Args:
        data:            dict with "summary" and "buffer" keys
        api_key:         OpenAI API key (needed for the summarisation LLM)
        model:           Model for summarisation
        max_token_limit: Token budget before summarisation kicks in
    Returns:
        ConversationSummaryBufferMemory with restored state
    """
    from langchain_classic.memory import ConversationSummaryBufferMemory
    from langchain_openai import ChatOpenAI
    from langchain_core.messages import HumanMessage, AIMessage
    llm = ChatOpenAI(model=model, api_key=api_key, temperature=0)
    memory = ConversationSummaryBufferMemory(
        llm=llm,
        max_token_limit=max_token_limit,
        human_prefix="User",
        ai_prefix="Assistant",
    )
    # Restore the compressed summary
    memory.moving_summary_buffer = data.get("summary", "")
    # Restore recent messages into the chat buffer
    for msg in data.get("buffer", []):
        if msg["role"] == "user":
            memory.chat_memory.add_message(HumanMessage(content=msg["content"]))
        else:
            memory.chat_memory.add_message(AIMessage(content=msg["content"]))
    return memory
# ─────────────────────────────────────────────────────────────────────────────
# Factory
# ─────────────────────────────────────────────────────────────────────────────
def build_session_store(
    backend: str = "sqlite",
    sqlite_path: Optional[Path] = None,
    redis_url: str = "redis://localhost:6379",
    redis_ttl: int = 86400,
) -> BaseSessionStore:
    """
    Factory to create a session store from configuration.
    Args:
        backend:     "sqlite" or "redis"
        sqlite_path: Path to SQLite DB file (only for sqlite)
        redis_url:   Redis connection URL (only for redis)
        redis_ttl:   Session TTL in seconds (only for redis)
    """
    if backend == "sqlite":
        return SQLiteSessionStore(db_path=sqlite_path)
    elif backend == "redis":
        return RedisSessionStore(url=redis_url, ttl_seconds=redis_ttl)
    else:
        raise ValueError(f"Unknown session backend: {backend}. Use 'sqlite' or 'redis'.")
