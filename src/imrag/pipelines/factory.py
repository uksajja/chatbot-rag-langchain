from __future__ import annotations

from langchain_core.embeddings import Embeddings
from langchain_core.language_models import BaseChatModel
from langchain_core.retrievers import BaseRetriever
from langchain_core.vectorstores import VectorStore

from imrag.core.bootstrap import register_all
from imrag.core.config import Config
from imrag.core.registries import (
    EMBEDDINGS, LLMS, RERANKERS, RETRIEVERS, VECTORSTORES,
)
from imrag.generation.prompts import load_prompt_template
from imrag.pipelines.rag_pipeline import RAGPipeline


def build_embedding(cfg: Config) -> Embeddings:
    register_all()
    return EMBEDDINGS.build(cfg.embedding.type, **cfg.embedding.params)


def build_vectorstore(cfg: Config, embedding: Embeddings) -> VectorStore:
    register_all()
    return VECTORSTORES.build(
        cfg.vectorstore.type, embedding=embedding, **cfg.vectorstore.params
    )


def build_pipeline(cfg: Config) -> RAGPipeline:
    """Build the online (query) pipeline from config."""
    print(f" Building RAG pipeline ")
    register_all()

    embedding: Embeddings = build_embedding(cfg)
    store: VectorStore = build_vectorstore(cfg, embedding)
    llm: BaseChatModel = LLMS.build(cfg.llm.type, **cfg.llm.params)
    retriever: BaseRetriever = RETRIEVERS.build(
        cfg.retrieval.type, store=store, llm=llm, **cfg.retrieval.params
    )
    reranker = (
        RERANKERS.build(cfg.reranker.type, **cfg.reranker.params)
        if cfg.reranker else None
    )

    return RAGPipeline(
        retriever=retriever,
        llm=llm,
        prompt=load_prompt_template(cfg.prompt.path),
        reranker=reranker,
        min_sources=cfg.guardrails.min_sources,
    )