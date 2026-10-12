from __future__ import annotations

import json
import logging
import uuid
from pathlib import Path


from langchain_core.documents import Document

from imrag.core.bootstrap import register_all
from imrag.core.config import ConfigLoader, Config
from imrag.document_loaders.factory import load_documents
from imrag.core.registries import CHUNKERS
from imrag.pipelines.factory import build_embedding, build_vectorstore

logger = logging.getLogger(__name__)

def chunk_id(chunk: Document) -> str:
    """Deterministic ID, so re-running ingestion updates instead of duplicating."""
    key = "|".join([
        str(chunk.metadata.get("source")),
        str(chunk.metadata.get("page")),
        chunk.page_content,
    ])
    return str(uuid.uuid5(uuid.NAMESPACE_URL, key))


def _persist_if_needed(store, params: dict) -> None:
    """FAISS keeps its index in memory, so it must be saved explicitly."""
    if not hasattr(store, "save_local"):
        return
    directory = params.get("persist_directory")
    if not directory:
        logger.warning("FAISS store has no persist_directory; the index will be lost")
        return
    store.save_local(directory, params.get("index_name", "index"))


def run_ingestion(cfg: Config) -> dict:
    """Load → chunk → embed → upsert, then write the chunk cache used by BM25."""
    register_all()
    embedding = build_embedding(cfg)
    store = build_vectorstore(cfg, embedding)
    chunker = CHUNKERS.build(
        cfg.chunking.type, embedding=embedding, **cfg.chunking.params
    )

    load_result = load_documents(cfg.loading)
    docs = load_result.docs
    if not docs:
        raise FileNotFoundError(f"No PDFs found under {cfg.loading.data_dir}")

    chunks = chunker.split_documents(docs)
    unique: dict[str, Document] = {}
    for c in chunks:
        unique.setdefault(chunk_id(c), c)       # repeated headers/footers collapse
    ids, items = list(unique), list(unique.values())
    logger.info("%d pages → %d chunks (%d unique)", len(docs), len(chunks), len(items))

    size = cfg.ingestion.batch_size
    for i in range(0, len(items), size):
        store.add_documents(items[i:i + size], ids=ids[i:i + size])
        logger.info("upserted %d/%d", min(i + size, len(items)), len(items))

    _persist_if_needed(store, cfg.vectorstore.params)

    cache = Path(cfg.ingestion.chunk_cache)
    cache.parent.mkdir(parents=True, exist_ok=True)
    with cache.open("w") as f:
        for cid, c in unique.items():
            f.write(json.dumps({"id": cid, "text": c.page_content,
                                "metadata": c.metadata}) + "\n")

    return {
        "docs": len(docs),
        "chunks": len(items),
        "skipped": len(load_result.skipped),
        "empty": len(load_result.empty),
        "failed": len(load_result.failed),
        "chunk_cache": str(cache),
    }


if __name__ == "__main__":
    import sys
    from dotenv import load_dotenv
    from imrag.core.config import load_config

    config_path = Path("configs/base_config.yaml")

    cfg_loader = ConfigLoader( config_path )
    cfg = cfg_loader.load_config()
    run_ingestion(cfg)