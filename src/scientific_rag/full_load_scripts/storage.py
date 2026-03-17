"""
storage.py — Unified file I/O abstraction for local disk and AWS S3.

Both LocalStorage and S3Storage implement the same BaseStorage interface,
so extract_pdfs.py and index_documents.py never care where files live.

Usage:
    # Local
    store = LocalStorage(Path("data/raw"))

    # S3
    store = S3Storage(bucket="my-bucket", prefix="raw/")

    # Same API for both
    stems = store.list_files(".pdf")
    text  = store.read_text("001. Some Doc", ".md")
    store.write_text("001. Some Doc", ".json", json_str)
"""

from abc import ABC, abstractmethod
from pathlib import Path

import boto3
from botocore.exceptions import ClientError
from loguru import logger


# ─────────────────────────────────────────────────────────────────────────────
# Base interface
# ─────────────────────────────────────────────────────────────────────────────

class BaseStorage(ABC):
    """
    Abstract file storage interface.

    Every method uses (stem, suffix) to identify a file:
        stem   = "001. The Joint Winter Runway Friction NASA Perspective"
        suffix = ".pdf", ".md", ".json"
    """

    @abstractmethod
    def list_files(self, suffix: str) -> list[str]:
        """Return all stems that have a file with the given suffix."""

    @abstractmethod
    def exists(self, stem: str, suffix: str) -> bool:
        """Check if a file exists."""

    @abstractmethod
    def read_bytes(self, stem: str, suffix: str) -> bytes:
        """Read raw bytes (for PDFs)."""

    @abstractmethod
    def read_text(self, stem: str, suffix: str, encoding: str = "utf-8") -> str:
        """Read as text (for .md, .json)."""

    @abstractmethod
    def write_text(self, stem: str, suffix: str, content: str, encoding: str = "utf-8"):
        """Write text content."""

    @abstractmethod
    def get_local_path(self, stem: str, suffix: str) -> Path:
        """
        Return a local filesystem path to the file.
        For S3 this downloads to a cache directory first.
        Needed because Docling requires a real file path for OCR.
        """


# ─────────────────────────────────────────────────────────────────────────────
# Local filesystem
# ─────────────────────────────────────────────────────────────────────────────

class LocalStorage(BaseStorage):
    """
    Reads/writes files from a local directory.

    Args:
        base_dir: Root directory (e.g. Path("data/raw"))
    """

    def __init__(self, base_dir: Path):
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(parents=True, exist_ok=True)

    def _path(self, stem: str, suffix: str) -> Path:
        return self.base_dir / f"{stem}{suffix}"

    def list_files(self, suffix: str) -> list[str]:
        return sorted(p.stem for p in self.base_dir.glob(f"*{suffix}"))

    def exists(self, stem: str, suffix: str) -> bool:
        return self._path(stem, suffix).exists()

    def read_bytes(self, stem: str, suffix: str) -> bytes:
        return self._path(stem, suffix).read_bytes()

    def read_text(self, stem: str, suffix: str, encoding: str = "utf-8") -> str:
        return self._path(stem, suffix).read_text(encoding=encoding)

    def write_text(self, stem: str, suffix: str, content: str, encoding: str = "utf-8"):
        path = self._path(stem, suffix)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding=encoding)

    def get_local_path(self, stem: str, suffix: str) -> Path:
        return self._path(stem, suffix)

    def __repr__(self):
        return f"LocalStorage({self.base_dir})"


# ─────────────────────────────────────────────────────────────────────────────
# AWS S3
# ─────────────────────────────────────────────────────────────────────────────

class S3Storage(BaseStorage):
    """
    Reads/writes files from an S3 bucket.

    Args:
        bucket:      S3 bucket name
        prefix:      Key prefix inside the bucket (e.g. "raw/", "processed/markdown/")
        local_cache: Local directory for caching downloaded files (default: /tmp/rag_cache)
        region:      AWS region (default: us-east-1)

    Authentication uses standard boto3 chain:
        - Environment variables (AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY)
        - ~/.aws/credentials profile
        - IAM role (if running on EC2/ECS/Lambda)
    """

    def __init__(
        self,
        bucket: str,
        prefix: str = "",
        local_cache: Path = Path("/tmp/rag_cache"),
        region: str = "us-east-1",
    ):
        self.bucket = bucket
        self.prefix = prefix.rstrip("/") + "/" if prefix else ""
        self.local_cache = Path(local_cache)
        self.local_cache.mkdir(parents=True, exist_ok=True)
        self.s3 = boto3.client("s3", region_name=region)
        logger.info(f"S3Storage: s3://{bucket}/{self.prefix}")

    def _key(self, stem: str, suffix: str) -> str:
        return f"{self.prefix}{stem}{suffix}"

    def _cache_path(self, stem: str, suffix: str) -> Path:
        return self.local_cache / f"{stem}{suffix}"

    def list_files(self, suffix: str) -> list[str]:
        stems = []
        paginator = self.s3.get_paginator("list_objects_v2")

        for page in paginator.paginate(Bucket=self.bucket, Prefix=self.prefix):
            for obj in page.get("Contents", []):
                key = obj["Key"]
                if key.endswith(suffix):
                    # Extract stem: remove prefix and suffix
                    name = key[len(self.prefix):]
                    stem = name[: -len(suffix)]
                    stems.append(stem)

        return sorted(stems)

    def exists(self, stem: str, suffix: str) -> bool:
        try:
            self.s3.head_object(Bucket=self.bucket, Key=self._key(stem, suffix))
            return True
        except ClientError:
            return False

    def read_bytes(self, stem: str, suffix: str) -> bytes:
        response = self.s3.get_object(Bucket=self.bucket, Key=self._key(stem, suffix))
        return response["Body"].read()

    def read_text(self, stem: str, suffix: str, encoding: str = "utf-8") -> str:
        return self.read_bytes(stem, suffix).decode(encoding)

    def write_text(self, stem: str, suffix: str, content: str, encoding: str = "utf-8"):
        self.s3.put_object(
            Bucket=self.bucket,
            Key=self._key(stem, suffix),
            Body=content.encode(encoding),
            ContentType="text/plain" if suffix == ".md" else "application/json",
        )
        logger.debug(f"Uploaded s3://{self.bucket}/{self._key(stem, suffix)}")

    def get_local_path(self, stem: str, suffix: str) -> Path:
        """Download to local cache if not already there, return local path."""
        cached = self._cache_path(stem, suffix)
        if not cached.exists():
            cached.parent.mkdir(parents=True, exist_ok=True)
            self.s3.download_file(
                Bucket=self.bucket,
                Key=self._key(stem, suffix),
                Filename=str(cached),
            )
            logger.debug(f"Downloaded to cache: {cached}")
        return cached

    def __repr__(self):
        return f"S3Storage(s3://{self.bucket}/{self.prefix})"


# ─────────────────────────────────────────────────────────────────────────────
# Factory — build storage from CLI args
# ─────────────────────────────────────────────────────────────────────────────

def build_storage(
    source: str,
    local_dir: Path = None,
    bucket: str = None,
    prefix: str = "",
    region: str = "us-east-1",
) -> BaseStorage:
    """
    Factory to create a storage instance from CLI arguments.

    Args:
        source:    "local" or "s3"
        local_dir: Path for LocalStorage (required if source == "local")
        bucket:    S3 bucket name (required if source == "s3")
        prefix:    S3 key prefix (only for S3)
        region:    AWS region (only for S3)
    """
    if source == "local":
        if local_dir is None:
            raise ValueError("local_dir is required for local storage")
        return LocalStorage(local_dir)
    elif source == "s3":
        if bucket is None:
            raise ValueError("bucket is required for S3 storage")
        return S3Storage(bucket=bucket, prefix=prefix, region=region)
    else:
        raise ValueError(f"Unknown source: {source}. Use 'local' or 's3'.")


