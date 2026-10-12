import json

from langchain_core.documents import Document

from imrag.core.registries import RETRIEVERS


@RETRIEVERS.register("dense")
def dense_retriever(
    store,
    llm=None,
    k: int = 5,
    search_type: str = "similarity",
    fetch_k: int | None = None,
    lambda_mult: float | None = None,
    score_threshold: float | None = None,
    filter: dict | None = None,
):
    """Vector-store retriever (replaces both "similarity" and "dense_retriever").

    Args:
        store: VectorStore instance (live dependency).
        llm: Accepted and ignored, so every retriever factory has the same signature.
        k: Number of chunks to return.
        search_type: "similarity", "mmr", or "similarity_score_threshold".
        fetch_k: mmr only. Candidates fetched before diversity re-ranking.
        lambda_mult: mmr only. 0 = max diversity, 1 = max relevance.
        score_threshold: similarity_score_threshold only. Minimum relevance score.
        filter: Metadata filter, passed to the vector store.

    YAML example:
        retrieval:
          type: dense
          params:
            k: 5
            search_type: mmr
            fetch_k: 20
            lambda_mult: 0.5
    """
    search_kwargs = {"k": k}
    for key, value in {"fetch_k": fetch_k, "lambda_mult": lambda_mult,
                       "score_threshold": score_threshold, "filter": filter}.items():
        if value is not None:
            search_kwargs[key] = value
    return store.as_retriever(search_type=search_type, search_kwargs=search_kwargs)


@RETRIEVERS.register("bm25")
def bm25_retriever(
    chunks_path: str,
    store=None,
    llm=None,
    k: int = 5,
    bm25_params: dict | None = None,
):
    """BM25 keyword retriever over the chunk cache written during ingestion.

    Args:
        chunks_path: JSONL file with one {"text", "metadata"} object per chunk.
        store, llm: Accepted and ignored (same signature as the other retrievers).
        k: Number of chunks to return.
        bm25_params: BM25 tuning, e.g. {"k1": 1.5, "b": 0.75}.

    YAML example:
        retrieval:
          type: bm25
          params:
            chunks_path: data/processed/chunks.jsonl
            k: 5
            bm25_params: { k1: 1.5, b: 0.75 }
    """
    from langchain_community.retrievers import BM25Retriever

    docs = []
    with open(chunks_path) as f:
        for line in f:
            record = json.loads(line)
            docs.append(Document(page_content=record["text"],
                                 metadata=record["metadata"]))
    return BM25Retriever.from_documents(docs, k=k, bm25_params=bm25_params)


@RETRIEVERS.register("self_query")
def self_query_retriever(
    store,
    llm,
    document_contents: str,
    metadata_field_info: list[dict],
    k: int = 5,
    **kw,
):
    """LLM turns the question into a semantic query plus a metadata filter.

    Requires `langchain-classic` and a vector store that supports structured
    queries (Chroma and Qdrant do; FAISS does not).

    Args:
        store: VectorStore instance.
        llm: Chat model that writes the filter (live dependency).
        document_contents: Short description of what the chunks contain.
        metadata_field_info: List of {name, description, type} dicts.
        k: Number of chunks to return.

    YAML example:
        retrieval:
          type: self_query
          params:
            document_contents: Injection molding datasheets and troubleshooting guides
            metadata_field_info:
              - { name: material_family, description: Resin family such as PA66 or PC/ABS, type: string }
              - { name: doc_type, description: datasheet, troubleshooting, or SOP, type: string }
    """
    from langchain_classic.chains.query_constructor.schema import AttributeInfo
    from langchain_classic.retrievers.self_query.base import SelfQueryRetriever

    fields = [AttributeInfo(**field) for field in metadata_field_info]
    return SelfQueryRetriever.from_llm(
        llm, store, document_contents, fields,
        search_kwargs={"k": k}, **kw,
    )


@RETRIEVERS.register("contextual_compression")
def contextual_compression_retriever(store, llm, base: dict, **kw):
    """Wrap another retriever and let the LLM trim each chunk to the relevant part.

    Requires `langchain-classic`. This makes one LLM call per retrieved chunk,
    so it adds latency and cost.

    Args:
        store: VectorStore instance, passed to the base retriever.
        llm: Chat model used by the extractor (live dependency).
        base: Nested spec for the retriever being wrapped, {type, params}.

    YAML example:
        retrieval:
          type: contextual_compression
          params:
            base:
              type: dense
              params: { k: 10 }
    """
    from langchain_classic.retrievers.contextual_compression import (
        ContextualCompressionRetriever,
    )
    from langchain_classic.retrievers.document_compressors import LLMChainExtractor

    base_retriever = RETRIEVERS.build(
        base["type"], store=store, llm=llm, **base.get("params", {})
    )
    return ContextualCompressionRetriever(
        base_compressor=LLMChainExtractor.from_llm(llm),
        base_retriever=base_retriever,
        **kw,
    )