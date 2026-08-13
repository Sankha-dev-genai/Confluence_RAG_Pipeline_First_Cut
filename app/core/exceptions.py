"""Typed exceptions so failures are explicit and catchable by layer."""
from __future__ import annotations


class RagError(Exception):
    """Base class for all application errors."""


class ConfigError(RagError):
    """Missing or invalid configuration / environment."""


class SourceError(RagError):
    """Knowledge-source (Confluence / Wiki) ingestion failure."""


class IngestionError(SourceError):
    """A stage of the ingestion pipeline failed."""


class VectorStoreError(RagError):
    """Embedding or FAISS index failure."""


class RetrievalError(RagError):
    """Retrieval failed (e.g. index not built)."""


class GenerationError(RagError):
    """LLM answer generation failed."""