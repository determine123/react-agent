"""RAG QA — retrieval-augmented generation pipeline with HyDE & Self-RAG.

Advanced strategies:
- **HyDE** (Hypothetical Document Embeddings): generate a hypothetical answer
  first, then use it as the query for vector retrieval — improves recall.
- **Self-RAG**: after generating an answer, reflect on whether it's well-supported
  by the retrieved context; if not, re-retrieve with a refined query.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from config import Config, RAGConfig

from llm.client import LLMClient
from rag.vector_store import VectorStore


# ---------------------------------------------------------------------------
# Prompt templates
# ---------------------------------------------------------------------------

_RAG_SYSTEM_PROMPT = (
    "You are a precise, helpful research assistant. "
    "Answer the user's question based **only** on the provided reference material. "
    "If the material does not contain enough information, say so honestly — "
    "do not fabricate or guess. "
    "When the material supports it, cite specific passages to justify your answer."
)

_RAG_USER_TEMPLATE = """## Reference Material

{context}

## Question

{question}

Please provide a clear, well-structured answer based on the reference material above."""

_HYDE_PROMPT = (
    "You are a knowledgeable assistant. Generate a brief, factual paragraph "
    "that answers the following question. Do NOT use any external knowledge "
    "beyond what a well-informed person would know. This is a hypothetical "
    "document to help with search retrieval — be specific and detailed.\n\n"
    "Question: {question}\n\n"
    "Hypothetical answer:"
)

_SELF_RAG_REFLECT_PROMPT = (
    "You are a critical evaluator. Given the question, the retrieved context, "
    "and the generated answer, decide if the answer is **fully supported** by "
    "the context.\n\n"
    "Question: {question}\n\n"
    "Retrieved Context:\n{context}\n\n"
    "Generated Answer: {answer}\n\n"
    'Reply with ONLY one word: "SUPPORTED" or "UNSUPPORTED". '
    "Then on the next line, if UNSUPPORTED, provide a better search query "
    "to find the missing information."
)

_REFINE_QUERY_PROMPT = (
    "The previous search did not retrieve sufficient information.\n"
    "Original question: {question}\n"
    "Previous answer (unsupported): {answer}\n"
    "Reflection: {reflection}\n\n"
    "Generate a refined search query (1 sentence) to find the missing information."
)


class RAGQA:
    """Retrieve-then-read QA pipeline with HyDE and Self-RAG support.

    Usage::

        from config import load_config
        from rag.qa import RAGQA

        cfg = load_config()
        qa = RAGQA(cfg)

        # Standard RAG
        answer = qa.answer("What is RAG?")

        # HyDE mode — better recall for vague questions
        answer = qa.answer("What is RAG?", strategy="hyde")

        # Self-RAG mode — validates answer against context
        answer = qa.answer("What is RAG?", strategy="self_rag")
    """

    def __init__(self, config: Config) -> None:
        self._cfg = config
        self._vector_store: VectorStore | None = None
        self._llm: LLMClient | None = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def answer(
        self,
        question: str,
        top_k: int | None = None,
        *,
        return_sources: bool = False,
        strategy: str = "default",
    ) -> str | dict[str, Any]:
        """Answer *question* using the indexed document corpus.

        Args:
            question: Natural-language question.
            top_k: Number of chunks to retrieve (defaults to RAGConfig.top_k).
            return_sources: When True, returns a dict with ``answer``,
                            ``sources``, ``context``, ``strategy`` keys.
            strategy: Retrieval strategy:
                - ``"default"`` — standard retrieve-then-read
                - ``"hyde"`` — generate hypothetical answer, use it as query
                - ``"self_rag"`` — validate answer, re-retrieve if unsupported

        Returns:
            Answer string, or dict with full metadata when return_sources=True.
        """
        k = top_k or self._cfg.rag.top_k

        if strategy == "hyde":
            return self._answer_with_hyde(question, k, return_sources)
        elif strategy == "self_rag":
            return self._answer_with_self_rag(question, k, return_sources)
        else:
            return self._answer_default(question, k, return_sources)

    # ------------------------------------------------------------------
    # Strategy: Default RAG
    # ------------------------------------------------------------------

    def _answer_default(
        self, question: str, top_k: int, return_sources: bool
    ) -> str | dict[str, Any]:
        docs = self.vector_store.query(question, top_k=top_k)
        if not docs:
            fb = "I couldn't find any relevant documents to answer that question."
            return {"answer": fb, "sources": [], "context": "", "strategy": "default"} if return_sources else fb

        context, sources = self._build_context_and_sources(docs)
        answer = self._generate(question, context)

        if return_sources:
            return {"answer": answer, "sources": sources, "context": context, "strategy": "default"}
        return answer

    # ------------------------------------------------------------------
    # Strategy: HyDE (Hypothetical Document Embeddings)
    # ------------------------------------------------------------------

    def _answer_with_hyde(
        self, question: str, top_k: int, return_sources: bool
    ) -> str | dict[str, Any]:
        """Generate a hypothetical answer → use it for retrieval → generate real answer.

        HyDE improves recall when the original question is short or uses
        vocabulary different from the indexed documents.
        """
        # Step 1: Generate hypothetical answer
        hypo_messages = [
            {"role": "user", "content": _HYDE_PROMPT.format(question=question)}
        ]
        hypo_raw = self.llm.chat(messages=hypo_messages)
        hypothetical = hypo_raw["choices"][0]["message"].get("content", question)

        # Step 2: Use hypothetical answer as search query
        docs = self.vector_store.query(hypothetical, top_k=top_k)
        if not docs:
            # Fall back to original question
            docs = self.vector_store.query(question, top_k=top_k)

        if not docs:
            fb = "I couldn't find any relevant documents to answer that question."
            return {"answer": fb, "sources": [], "context": "", "strategy": "hyde"} if return_sources else fb

        # Step 3: Generate final answer from real context
        context, sources = self._build_context_and_sources(docs)
        answer = self._generate(question, context)

        if return_sources:
            return {
                "answer": answer,
                "sources": sources,
                "context": context,
                "strategy": "hyde",
                "hypothetical_query": hypothetical[:300],
            }
        return answer

    # ------------------------------------------------------------------
    # Strategy: Self-RAG
    # ------------------------------------------------------------------

    def _answer_with_self_rag(
        self, question: str, top_k: int, return_sources: bool
    ) -> str | dict[str, Any]:
        """Generate an answer, then reflect on whether it's well-supported.

        If the answer is unsupported by the retrieved context, refine the
        search query and try again (up to 2 retries).
        """
        max_retries = 2
        all_sources: list[dict] = []

        for attempt in range(max_retries + 1):
            # Retrieve
            query = question if attempt == 0 else refined_query
            docs = self.vector_store.query(query, top_k=top_k)

            if not docs:
                fb = "I couldn't find any relevant documents to answer that question."
                return (
                    {"answer": fb, "sources": [], "context": "", "strategy": "self_rag"}
                    if return_sources
                    else fb
                )

            context, sources = self._build_context_and_sources(docs)
            for s in sources:
                if s not in all_sources:
                    all_sources.append(s)

            # Generate
            answer = self._generate(question, context)

            # Reflect
            is_supported, reflection = self._reflect(question, context, answer)
            if is_supported:
                break

            # Refine query for next attempt
            if attempt < max_retries:
                refined_query = self._refine_query(question, answer, reflection)

        if return_sources:
            return {
                "answer": answer,
                "sources": all_sources,
                "context": context,
                "strategy": "self_rag",
                "attempts": attempt + 1,
                "final_reflection": reflection,
            }
        return answer

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _build_context_and_sources(
        self, docs: list[dict[str, Any]]
    ) -> tuple[str, list[dict[str, Any]]]:
        context_parts: list[str] = []
        sources: list[dict[str, Any]] = []
        for i, doc in enumerate(docs, start=1):
            src = doc.get("metadata", {}).get("source", "unknown")
            chunk_text = doc.get("text", "")
            context_parts.append(f"[{i}] (source: {src})\n{chunk_text}")
            sources.append(
                {
                    "index": i,
                    "source": src,
                    "text_preview": chunk_text[:200],
                    "distance": doc.get("distance"),
                }
            )
        return "\n\n".join(context_parts), sources

    def _generate(self, question: str, context: str) -> str:
        messages = [
            {"role": "system", "content": _RAG_SYSTEM_PROMPT},
            {
                "role": "user",
                "content": _RAG_USER_TEMPLATE.format(context=context, question=question),
            },
        ]
        raw = self.llm.chat(messages=messages)
        return raw["choices"][0]["message"].get("content", "")

    def _reflect(self, question: str, context: str, answer: str) -> tuple[bool, str]:
        """Ask LLM to judge whether the answer is supported by the context."""
        messages = [
            {
                "role": "user",
                "content": _SELF_RAG_REFLECT_PROMPT.format(
                    question=question, context=context[:3000], answer=answer
                ),
            },
        ]
        raw = self.llm.chat(messages=messages)
        reflection = raw["choices"][0]["message"].get("content", "SUPPORTED")
        is_supported = reflection.strip().upper().startswith("SUPPORTED")
        return is_supported, reflection

    def _refine_query(self, question: str, answer: str, reflection: str) -> str:
        """Generate a refined search query based on reflection feedback."""
        messages = [
            {
                "role": "user",
                "content": _REFINE_QUERY_PROMPT.format(
                    question=question, answer=answer, reflection=reflection
                ),
            },
        ]
        raw = self.llm.chat(messages=messages)
        return raw["choices"][0]["message"].get("content", question)

    # ------------------------------------------------------------------
    # Lazy accessors
    # ------------------------------------------------------------------

    @property
    def vector_store(self) -> VectorStore:
        if self._vector_store is None:
            self._vector_store = VectorStore(self._cfg.rag)
        return self._vector_store

    @property
    def llm(self) -> LLMClient:
        if self._llm is None:
            self._llm = LLMClient(self._cfg.llm)
        return self._llm

    # ------------------------------------------------------------------
    # Index management
    # ------------------------------------------------------------------

    def index_documents(self, dir_path: str, glob: str = "**/*.*") -> int:
        from rag.loader import load_directory
        from rag.splitter import recursive_split

        docs = load_directory(dir_path, glob=glob)
        all_chunks: list[str] = []
        all_metadatas: list[dict] = []

        for doc in docs:
            chunks = recursive_split(
                doc["text"],
                chunk_size=self._cfg.rag.chunk_size,
                chunk_overlap=self._cfg.rag.chunk_overlap,
            )
            for chunk in chunks:
                all_chunks.append(chunk)
                all_metadatas.append(doc.get("metadata", {}))

        self.vector_store.add(all_chunks, metadatas=all_metadatas)
        return len(all_chunks)

    def index_count(self) -> int:
        return self.vector_store.count()
