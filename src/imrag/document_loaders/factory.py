from __future__ import annotations

import hashlib
import json
import logging
from dataclasses import dataclass, field
from pathlib import Path

from langchain_core.documents import Document

from imrag.core.config import LoadingConfig
from imrag.core.registries import LOADERS

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------- loaders
# Every factory takes `path` plus loader-specific options and returns a
# LangChain BaseLoader. YAML params arrive as keyword arguments.

@LOADERS.register("pymupdf")
def pymupdf_loader(path: str, **kw):
    """Fast PDF text extraction, one Document per page (0-based "page").

    Tables come out as flattened text. Prefer docling or unstructured for
    table-heavy datasheets. Scanned PDFs without a text layer give empty pages.
    """
    from langchain_community.document_loaders import PyMuPDFLoader
    return PyMuPDFLoader(path, **kw)


@LOADERS.register("pypdf")
def pypdf_loader(path: str, **kw):
    """Pure-Python PDF loader, one Document per page."""
    from langchain_community.document_loaders import PyPDFLoader
    return PyPDFLoader(path, **kw)


@LOADERS.register("unstructured")
def unstructured_loader(path: str, **kw):
    """Layout-aware parsing (titles, tables, lists) for PDF, DOCX, HTML, and more.

    Requires `langchain-unstructured`. Extra options, such as the partition
    strategy, go in YAML params.
    """
    from langchain_unstructured import UnstructuredLoader
    return UnstructuredLoader(file_path=path, **kw)


@LOADERS.register("docling")
def docling_loader(path: str, export_type: str = "markdown", **kw):
    """Docling document conversion with table structure preserved.

    Requires `langchain-docling`. "markdown" returns one Document per file as
    Markdown (tables stay as Markdown tables) for your own chunker to split.
    "doc_chunks" returns Docling's own chunks, which bypasses your chunker.
    """
    from langchain_docling import DoclingLoader
    from langchain_docling.loader import ExportType

    export = ExportType.MARKDOWN if export_type == "markdown" else ExportType.DOC_CHUNKS
    return DoclingLoader(file_path=path, export_type=export, **kw)


@LOADERS.register("docx")
def docx_loader(path: str, **kw):
    """Word documents via docx2txt (text only, tables are flattened)."""
    from langchain_community.document_loaders import Docx2txtLoader
    return Docx2txtLoader(path, **kw)


@LOADERS.register("html")
def html_loader(path: str, **kw):
    """HTML via BeautifulSoup. Needs `beautifulsoup4` and `lxml`."""
    from langchain_community.document_loaders import BSHTMLLoader
    return BSHTMLLoader(path, **kw)


@LOADERS.register("text")
def text_loader(path: str, encoding: str = "utf-8", **kw):
    """Plain text and Markdown."""
    from langchain_community.document_loaders import TextLoader
    return TextLoader(path, encoding=encoding, **kw)


@LOADERS.register("csv")
def csv_loader(path: str, **kw):
    """CSV with one Document per row. Useful for parameter tables."""
    from langchain_community.document_loaders import CSVLoader
    return CSVLoader(file_path=path, **kw)


# ---------------------------------------------------------------- dispatch

@dataclass
class LoadResult:
    docs: list[Document] = field(default_factory=list)
    skipped: list[str] = field(default_factory=list)      # no loader for the extension
    empty: list[str] = field(default_factory=list)        # loaded, but no text
    failed: dict[str, str] = field(default_factory=dict)  # file -> error message


def _normalize_ext(ext: str) -> str:
    """
    Turn extension into canonical form (lowercase, no leading dot).
    >>> _normalize_ext(".PDF")
    'pdf'
    """

    return ext.lower().lstrip(".")


def _clean_metadata(meta: dict) -> dict:
    """Vector stores such as Chroma accept only str/int/float/bool values.
    Loaders like Unstructured emit lists and dicts, so those become JSON strings."""
    cleaned = {}
    for key, value in meta.items():
        if value is None:
            continue
        cleaned[key] = value if isinstance(value, (str, int, float, bool)) \
            else json.dumps(value, default=str)
    return cleaned


def _file_hash(path: Path) -> str:
    """
    This function takes a Path and returns a file hash.
    Take the first 16 chars of the SHA256 hash of the file's contents.
    >>> _file_hash(Path(__file__))  # doctest: +ELLIPSIS
    '...'
    """
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()[:16]


def load_documents(cfg: LoadingConfig) -> LoadResult:
    """Walk cfg.data_dir and load every file that has a loader configured."""
    root = Path(cfg.data_dir)
    if not root.is_dir():
        raise FileNotFoundError(f"data_dir not found: {root}")

    # Map file extensions to loader specs.
    routes = {_normalize_ext(ext): spec for ext, spec in cfg.loaders.items()}
    
    # Walk the data directory and load all files with a registered loader.
    candidates = root.rglob("*") if cfg.recursive else root.glob("*")

    result = LoadResult()

    for path in sorted(p for p in candidates if p.is_file()):
        # Skip hidden files and directories (including .git).

        rel = path.relative_to(root) # relative path from data_dir

        """
        Example:
        root = Path("documents/work")
        path = Path("documents/work/invoices/2026/october.pdf")
        rel = path.relative_to(root)
        print(rel)
        print(rel.parts)
        output: 
        invoices/2026/october.py
        ('invoices', '2026', 'october.py')
        """

        if any(part.startswith(".") for part in rel.parts):      # .DS_Store, .git, ...
            continue

        ext = _normalize_ext(path.suffix)
        spec = routes.get(ext)
        if spec is None:
            result.skipped.append(rel.as_posix())
            continue

        try:
            loader = LOADERS.build(spec.type, path=str(path), **spec.params)
            file_docs = loader.load()
        except Exception as exc:
            if cfg.on_error == "raise":
                raise
            logger.warning("failed to load %s: %r", rel, exc)
            result.failed[rel.as_posix()] = repr(exc)
            continue

        file_docs = [d for d in file_docs if d.page_content.strip()]
        if not file_docs:
            logger.warning("%s produced no text (scanned PDF? needs OCR)", rel)
            result.empty.append(rel.as_posix())
            continue

        # Add metadata to each document.
        """
        The hash is stored as file_hash in every document’s metadata, and from there it ends up on every chunk. 
        It exists to answer the question, “has this file changed?” by content instead of by filename or modification time.
        """
        digest = _file_hash(path)

        # Add doc_type based on the first directory level (e.g., "invoices", "manuals")
        doc_type = rel.parts[0] if len(rel.parts) > 1 else "general"
        for d in file_docs:
            d.metadata = _clean_metadata({
                **d.metadata,
                "source": rel.as_posix(),
                "file_name": path.name,
                "file_type": ext,
                "doc_type": doc_type,
                "loader": spec.type,
                "file_hash": digest,
            })
        result.docs.extend(file_docs)

    logger.info("loaded %d docs; %d skipped, %d empty, %d failed",
                len(result.docs), len(result.skipped),
                len(result.empty), len(result.failed))
    return result