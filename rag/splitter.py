"""Text splitter — recursive character splitting with overlap."""

from __future__ import annotations


def recursive_split(
    text: str,
    chunk_size: int = 800,
    chunk_overlap: int = 120,
    separators: list[str] | None = None,
) -> list[str]:
    """Split *text* into overlapping chunks, preferring the listed separators.

    Args:
        text: Raw document text.
        chunk_size: Maximum characters per chunk.
        chunk_overlap: Characters to overlap between consecutive chunks.
        separators: Priority list of separators (default: paragraph, sentence, space).

    Returns:
        List of text chunks.
    """
    if separators is None:
        separators = ["\n\n", "\n", "。", ". ", ".", " "]

    return _split_recursive(text, separators, chunk_size, chunk_overlap)


def _split_recursive(
    text: str,
    separators: list[str],
    chunk_size: int,
    overlap: int,
) -> list[str]:
    """Recurse through separators, splitting progressively finer."""
    # Find the first separator that actually appears
    sep = None
    for candidate in separators:
        if candidate in text:
            sep = candidate
            break

    if sep is None:
        # No separator found — just chunk by length
        return _chunk_by_length(text, chunk_size, overlap)

    fragments = text.split(sep)
    chunks: list[str] = []
    current = ""

    for frag in fragments:
        if len(current) + len(sep) + len(frag) <= chunk_size:
            current = (current + sep + frag) if current else frag
        else:
            if current:
                chunks.append(current)
            # If a single fragment exceeds chunk_size, recurse deeper
            if len(frag) > chunk_size:
                chunks.extend(_split_recursive(frag, separators[1:], chunk_size, overlap))
                current = ""
            else:
                current = frag

    if current:
        chunks.append(current)

    # Apply overlap between adjacent chunks (respect chunk_size limit)
    if overlap > 0 and len(chunks) > 1:
        overlapped = [chunks[0]]
        for c in chunks[1:]:
            # Take overlap chars from the END of the ORIGINAL previous chunk
            # (not the already-overlapped version) to prevent size creep
            prev_original = chunks[len(overlapped) - 1]
            overlap_text = prev_original[-overlap:] if len(prev_original) > overlap else prev_original
            merged = overlap_text + c
            # If the merged chunk exceeds chunk_size + overlap, trim
            max_len = chunk_size + overlap
            if len(merged) > max_len:
                merged = merged[:max_len]
            overlapped.append(merged)
        return overlapped

    return chunks


def _chunk_by_length(text: str, size: int, _overlap: int) -> list[str]:
    """Fallback: split *text* into fixed-size chunks."""
    return [text[i : i + size] for i in range(0, len(text), size)]
