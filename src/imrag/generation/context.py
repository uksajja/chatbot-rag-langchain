from pathlib import Path
import uuid

from langchain_core.documents import Document

from imrag.core.schemas import SourceChunk


def _page_number(metadata: dict) -> int | None:
    """Return a 1-based page number, or None if the loader gave none.

    Loaders disagree: PyMuPDF and PyPDF store a 0-based "page", while loaders
    such as Unstructured and Docling store a 1-based "page_number". The
    difference is resolved here, once, so nothing else does page arithmetic.
    """
    if metadata.get("page_number") is not None:
        value, offset = metadata["page_number"], 0  # 1-based, so no offset
    elif metadata.get("page") is not None:
        value, offset = metadata["page"], 1         # 0-based, so add 1
    else:
        return None
    try:
        return int(value) + offset
    except (TypeError, ValueError):
        return None


def to_sources(docs: list[Document]) -> list[SourceChunk]:
    """Convert LangChain Documents into numbered, project-owned SourceChunks."""
    sources = []
    for i, d in enumerate(docs, start=1):
        meta = dict(d.metadata)                  # copy, so nothing mutates the Document
        source = meta.get("source")
        stable_key = "|".join([
            str(meta.get("source")),
            str(meta.get("page")),
            d.page_content,
        ])
        sources.append(
            SourceChunk(
                index=i,
                source_id=str(meta.get("file_hash") or source or "unknown"),
                chunk_id=str(uuid.uuid5(uuid.NAMESPACE_URL, stable_key)),
                text=d.page_content,
                source=Path(source).name if source else None,
                page=_page_number(meta),
                metadata=meta,
            )
        )
    return sources


def source_label(s: SourceChunk) -> str:
    """Human-readable location, e.g. 'PA66_GF30_datasheet.pdf, p.3'."""
    name = s.source or "unknown source"
    return f"{name}, p.{s.page}" if s.page is not None else name


def format_context(sources: list[SourceChunk]) -> str:
    """Render sources as numbered blocks for the prompt's {context} slot."""
    return "\n\n".join(
        f"[{s.index}] ({source_label(s)})\n{s.text}" for s in sources
    )