"""
Central configuration for the 3GPP RAG chatbot.

Every value that was previously hardcoded in the original `main.py`
(API key, model name, thresholds, paths, ...) now lives here and is
sourced from environment variables / a `.env` file. Nothing in the
rest of the codebase should read `os.environ` directly — import
`settings` from this module instead, so there is exactly one source
of truth and configuration is trivially testable (see tests/conftest.py).
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

PROJECT_ROOT = Path(__file__).resolve().parent.parent


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # ── LLM provider (Groq) ────────────────────────────────────────────
    groq_api_key: str | None = Field(default=None)
    groq_model: str = Field(default="openai/gpt-oss-20b")
    groq_endpoint: str = Field(default="https://api.groq.com/openai/v1/chat/completions")
    groq_timeout_seconds: int = Field(default=30)
    groq_max_retries: int = Field(default=3)

    # ── Models ───────────────────────────────────────────────────────────
    embed_model_name: str = Field(default="sentence-transformers/all-MiniLM-L6-v2")
    reranker_model_name: str = Field(default="cross-encoder/ms-marco-MiniLM-L-6-v2")
    nli_model_name: str = Field(default="cross-encoder/nli-deberta-v3-small")
    enable_reranker: bool = Field(default=True)
    enable_nli_verifier: bool = Field(default=True)

    # ── Retrieval tuning ─────────────────────────────────────────────────
    retrieval_top_n: int = Field(default=20, ge=1)
    retrieval_top_k: int = Field(default=4, ge=1)
    min_retrieval_score: float = Field(default=0.30, ge=-1.0, le=1.0)
    min_confidence_score: float = Field(default=0.55, ge=0.0, le=1.0)
    # Hard gate: a claim must independently clear this entailment/semantic
    # bar regardless of how high its retrieval or rerank score is — this is
    # what stops a topically-relevant-but-unsupported (or numerically wrong)
    # claim from being accepted just because the surrounding passage scored
    # well on retrieval. See app/verification/confidence.py.
    min_semantic_score: float = Field(default=0.50, ge=0.0, le=1.0)

    # ── Confidence score weights (must sum to ~1.0, validated below) ─────
    weight_retrieval: float = Field(default=0.25)
    weight_rerank: float = Field(default=0.35)
    weight_entailment: float = Field(default=0.40)

    # ── Conversation memory ──────────────────────────────────────────────
    max_history_turns: int = Field(default=6, ge=0)
    session_ttl_minutes: int = Field(default=120, ge=1)

    # ── Paths ────────────────────────────────────────────────────────────
    raw_pdf_dir: Path = Field(default=PROJECT_ROOT / "data" / "raw_pdfs")
    processed_dir: Path = Field(default=PROJECT_ROOT / "data" / "processed")
    index_dir: Path = Field(default=PROJECT_ROOT / "data" / "index")

    # ── Logging ──────────────────────────────────────────────────────────
    log_level: str = Field(default="INFO")
    log_file: str | None = Field(default="logs/app.log")
    log_json: bool = Field(default=False)

    # ── API server ───────────────────────────────────────────────────────
    cors_origins: str = Field(default="http://localhost:3000,http://localhost:8000")
    api_host: str = Field(default="0.0.0.0")
    api_port: int = Field(default=8000)

    @field_validator("raw_pdf_dir", "processed_dir", "index_dir", mode="before")
    @classmethod
    def _resolve_relative_to_project_root(cls, v: str | Path) -> Path:
        p = Path(v)
        return p if p.is_absolute() else (PROJECT_ROOT / p)

    @property
    def index_path(self) -> Path:
        return self.index_dir / "faiss_3gpp.index"

    @property
    def meta_path(self) -> Path:
        return self.processed_dir / "chunks_meta.json"

    @property
    def cors_origin_list(self) -> list[str]:
        return [o.strip() for o in self.cors_origins.split(",") if o.strip()]


@lru_cache
def get_settings() -> Settings:
    """Cached accessor so we parse the environment exactly once per process."""
    return Settings()


# Convenience module-level singleton used across the app: `from app.config import settings`
settings = get_settings()
