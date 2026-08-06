"""Centralised configuration — read from env vars, .env files, and defaults."""

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional


@dataclass
class LLMConfig:
    provider: str = "openai"                 # openai | anthropic | ollama
    model: str = "gpt-4o"
    api_key: Optional[str] = None
    base_url: Optional[str] = None
    temperature: float = 0.0
    max_tokens: int = 4096
    timeout_s: float = 120.0


@dataclass
class RAGConfig:
    chunk_size: int = 800
    chunk_overlap: int = 120
    embedding_model: str = "chroma-default"  # chroma-default | text-embedding-3-small | BAAI/bge-base-zh
    vector_store_path: str = "./logs/vector_store"
    top_k: int = 5


@dataclass
class MemoryConfig:
    working_memory_limit: int = 20           # max recent exchanges kept in-context
    long_term_path: str = "./logs/memory.json"


@dataclass
class SearchConfig:
    tavily_api_key: str = ""                  # set via TAVILY_API_KEY env var


@dataclass
class Config:
    system_prompt: str = "You are a helpful, thoughtful AI assistant."
    llm: LLMConfig = field(default_factory=LLMConfig)
    rag: RAGConfig = field(default_factory=RAGConfig)
    memory: MemoryConfig = field(default_factory=MemoryConfig)
    search: SearchConfig = field(default_factory=SearchConfig)
    max_steps: int = 20                       # hard cap on agent loop iterations
    verbose: bool = True


def load_config() -> Config:
    """Build Config from environment variables, falling back to dataclass defaults."""
    _load_dotenv()
    cfg = Config()

    # --- LLM overrides ---
    cfg.llm.provider = os.getenv("LLM_PROVIDER", cfg.llm.provider)
    cfg.llm.model = os.getenv("LLM_MODEL", cfg.llm.model)
    cfg.llm.api_key = os.getenv("LLM_API_KEY", cfg.llm.api_key)
    cfg.llm.base_url = os.getenv("LLM_BASE_URL", cfg.llm.base_url)
    cfg.llm.temperature = float(os.getenv("LLM_TEMPERATURE", cfg.llm.temperature))
    cfg.llm.max_tokens = int(os.getenv("LLM_MAX_TOKENS", cfg.llm.max_tokens))

    # --- RAG overrides ---
    cfg.rag.chunk_size = int(os.getenv("RAG_CHUNK_SIZE", cfg.rag.chunk_size))
    cfg.rag.chunk_overlap = int(os.getenv("RAG_CHUNK_OVERLAP", cfg.rag.chunk_overlap))
    cfg.rag.top_k = int(os.getenv("RAG_TOP_K", cfg.rag.top_k))

    # --- Search overrides ---
    cfg.search.tavily_api_key = os.getenv("TAVILY_API_KEY", cfg.search.tavily_api_key)

    # --- Paths ---
    Path(cfg.rag.vector_store_path).parent.mkdir(parents=True, exist_ok=True)
    Path(cfg.memory.long_term_path).parent.mkdir(parents=True, exist_ok=True)

    return cfg


def _load_dotenv() -> None:
    """Load .env file from the project root (if it exists)."""
    try:
        from dotenv import load_dotenv as _ld
        _ld(override=False)
    except ImportError:
        pass
