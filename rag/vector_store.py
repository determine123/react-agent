"""Vector store — Chroma-backed embedding index with configurable models.

Supports:
- ChromaDB's built-in embedding functions (default: all-MiniLM-L6-v2 via ONNX)
- OpenAI / DeepSeek embeddings via API
- Sentence-Transformers (opt-in, pip install sentence-transformers)
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from config import RAGConfig


class VectorStore:
    """Wrapper around ChromaDB for document indexing and semantic retrieval.

    Embedding function selection is controlled by ``RAGConfig.embedding_model``:

    - ``"chroma-default"`` — ChromaDB's built-in ONNX model (no extra deps)
    - ``"text-embedding-3-small"`` or ``"text-embedding-*"`` — OpenAI-compatible API
    - Any string containing ``"/"`` (e.g. ``"BAAI/bge-base-zh"``) — Sentence-Transformers
    """

    def __init__(self, cfg: RAGConfig) -> None:
        self.cfg = cfg
        self._client = None
        self._collection = None
        self._emb_fn = None

    # ------------------------------------------------------------------
    # Lazy init
    # ------------------------------------------------------------------

    @property
    def collection(self):
        if self._collection is None:
            import chromadb

            Path(self.cfg.vector_store_path).mkdir(parents=True, exist_ok=True)
            self._client = chromadb.PersistentClient(path=self.cfg.vector_store_path)
            self._collection = self._client.get_or_create_collection(
                name="documents",
                embedding_function=self._embedding_function(),
            )
        return self._collection

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def add(
        self,
        chunks: list[str],
        metadatas: list[dict] | None = None,
        ids: list[str] | None = None,
    ) -> None:
        """Embed and store *chunks*.

        Args:
            chunks: Text chunks to index.
            metadatas: Optional metadata dicts (same length as *chunks*).
            ids: Optional unique IDs (auto-generated if omitted).
        """
        if not chunks:
            return
        if ids is None:
            # Use content hash prefix + index for collision-free IDs
            ids = [f"chunk_{i}_{hash(c[:40])}" for i, c in enumerate(chunks)]
        self.collection.add(documents=chunks, metadatas=metadatas, ids=ids)

    def query(self, query: str, top_k: int | None = None) -> list[dict[str, Any]]:
        """Return the *top_k* most similar chunks for *query*.

        Returns:
            List of ``{"id", "text", "metadata", "distance"}`` dicts.
        """
        k = top_k or self.cfg.top_k
        results = self.collection.query(query_texts=[query], n_results=k)

        docs: list[dict[str, Any]] = []
        ids_batch = results.get("ids", [[]])[0]
        docs_batch = results.get("documents", [[]])[0]
        metas_batch = (results.get("metadatas") or [[]])[0]
        dists_batch = (results.get("distances") or [[]])[0]

        for i, doc_id in enumerate(ids_batch):
            docs.append(
                {
                    "id": doc_id,
                    "text": docs_batch[i] if i < len(docs_batch) else "",
                    "metadata": metas_batch[i] if i < len(metas_batch) else {},
                    "distance": dists_batch[i] if i < len(dists_batch) else None,
                }
            )
        return docs

    def count(self) -> int:
        """Return the number of documents in the collection."""
        return self.collection.count()

    def clear(self) -> None:
        """Wipe all documents from the store."""
        ids = self.collection.get()["ids"]
        if ids:
            self.collection.delete(ids=ids)

    # ------------------------------------------------------------------
    # Embedding function factory
    # ------------------------------------------------------------------

    def _embedding_function(self):
        """Resolve and cache the embedding function based on config.

        Strategy by model_name:
        - ``"chroma-default"`` → ChromaDB built-in ONNX, falls back to SentenceTransformer
        - ``"/"`` in name → Sentence-Transformers (e.g. BAAI/bge-base-zh)
        - ``"text-embedding-*"`` → OpenAI-compatible API
        """
        if self._emb_fn is not None:
            return self._emb_fn

        model_name = self.cfg.embedding_model

        # -- Sentence-Transformers (local model) --
        if "/" in model_name:
            try:
                from chromadb.utils.embedding_functions import (
                    SentenceTransformerEmbeddingFunction,
                )
                self._emb_fn = SentenceTransformerEmbeddingFunction(
                    model_name=model_name
                )
                return self._emb_fn
            except ImportError:
                raise ImportError(
                    "sentence-transformers is required for local embedding models. "
                    "Install with: pip install sentence-transformers"
                )

        # -- OpenAI / DeepSeek API embeddings --
        if model_name.startswith("text-embedding-"):
            try:
                from chromadb.utils.embedding_functions import (
                    OpenAIEmbeddingFunction,
                )
                from config import load_config

                llm_cfg = load_config().llm
                self._emb_fn = OpenAIEmbeddingFunction(
                    api_key=llm_cfg.api_key,
                    api_base=llm_cfg.base_url,
                    model_name=model_name,
                )
                return self._emb_fn
            except ImportError:
                raise ImportError(
                    "chromadb[openai] extra is required. "
                    "Install with: pip install chromadb"
                )

        # -- ChromaDB built-in default (ONNX all-MiniLM-L6-v2) --
        # Try SentenceTransformer first (more reliable), fall back to None (ONNX)
        try:
            from chromadb.utils.embedding_functions import (
                SentenceTransformerEmbeddingFunction,
            )
            self._emb_fn = SentenceTransformerEmbeddingFunction(
                model_name="all-MiniLM-L6-v2"
            )
            return self._emb_fn
        except (ImportError, Exception):
            # ONNX download may fail in some environments — None uses
            # ChromaDB's own default, which also attempts ONNX
            self._emb_fn = None
            return self._emb_fn
