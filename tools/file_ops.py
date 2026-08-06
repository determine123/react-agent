"""File operations tool — read & write local files as agent "notes"."""

from __future__ import annotations

from pathlib import Path

# Sandbox the agent to a dedicated notes directory
_NOTES_DIR = Path("./logs/agent_notes")


def _ensure_notes_dir() -> Path:
    _NOTES_DIR.mkdir(parents=True, exist_ok=True)
    return _NOTES_DIR


def write_note(filename: str, content: str) -> dict:
    """Write content to a note file.

    Args:
        filename: File name (e.g. "meeting_notes.txt"). Saved under logs/agent_notes/.
        content: Text content to write.

    Returns:
        Dict with ``filename``, ``path``, ``size``, ``status``.
    """
    _ensure_notes_dir()
    # Prevent directory traversal
    safe_name = Path(filename).name
    filepath = _NOTES_DIR / safe_name

    try:
        filepath.write_text(content, encoding="utf-8")
        return {
            "filename": safe_name,
            "path": str(filepath),
            "size": len(content),
            "status": "written",
        }
    except Exception as exc:
        return {"error": f"Failed to write file: {exc}", "filename": safe_name}


def read_note(filename: str) -> dict:
    """Read a previously saved note.

    Args:
        filename: File name to read from logs/agent_notes/.

    Returns:
        Dict with ``filename``, ``content``, ``size``, ``status``.
    """
    _ensure_notes_dir()
    safe_name = Path(filename).name
    filepath = _NOTES_DIR / safe_name

    if not filepath.exists():
        return {
            "error": f"File '{safe_name}' does not exist.",
            "filename": safe_name,
        }

    try:
        content = filepath.read_text(encoding="utf-8")
        return {
            "filename": safe_name,
            "content": content,
            "size": len(content),
            "status": "read",
        }
    except Exception as exc:
        return {"error": f"Failed to read file: {exc}", "filename": safe_name}


def list_notes() -> dict:
    """List all saved notes.

    Returns:
        Dict with ``files`` (list of filenames) and ``count``.
    """
    _ensure_notes_dir()
    files = sorted(
        [
            {
                "name": f.name,
                "size": f.stat().st_size,
                "modified": f.stat().st_mtime,
            }
            for f in _NOTES_DIR.iterdir()
            if f.is_file()
        ],
        key=lambda x: x["modified"],
        reverse=True,
    )
    return {"files": files, "count": len(files), "notes_dir": str(_NOTES_DIR)}
