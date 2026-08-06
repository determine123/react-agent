"""Document loader — reads PDFs, Markdown, TXT, and web pages."""

from __future__ import annotations

from pathlib import Path


def load_file(path: str | Path) -> list[dict]:
    """Load a single file and return a list of document dicts.

    Each dict: ``{"text": str, "metadata": {"source": path, ...}}``
    """
    path = Path(path)
    suffix = path.suffix.lower()

    if suffix == ".txt":
        return _load_txt(path)
    if suffix == ".md":
        return _load_txt(path)
    if suffix == ".pdf":
        return _load_pdf(path)
    raise ValueError(f"Unsupported file type: {suffix}")


def load_directory(dir_path: str | Path, glob: str = "**/*.*") -> list[dict]:
    """Recursively load all supported files in a directory."""
    docs: list[dict] = []
    supported = {".txt", ".md", ".pdf"}
    for fp in Path(dir_path).rglob(glob):
        if fp.suffix.lower() in supported and fp.is_file():
            docs.extend(load_file(fp))
    return docs


# ------------------------------------------------------------------
# Backends
# ------------------------------------------------------------------

def _load_txt(path: Path) -> list[dict]:
    text = path.read_text(encoding="utf-8")
    return [{"text": text, "metadata": {"source": str(path), "type": "text"}}]


def _load_pdf(path: Path) -> list[dict]:
    try:
        import pymupdf  # fitz
    except ImportError:
        raise ImportError("PyMuPDF (fitz) is required for PDF loading. pip install PyMuPDF")

    docs: list[dict] = []
    with pymupdf.open(str(path)) as pdf:
        for page_num, page in enumerate(pdf, start=1):
            text = page.get_text()
            if text.strip():
                docs.append(
                    {
                        "text": text,
                        "metadata": {
                            "source": str(path),
                            "page": page_num,
                            "type": "pdf",
                        },
                    }
                )
    return docs
