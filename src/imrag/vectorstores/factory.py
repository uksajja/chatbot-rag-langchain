import os

from imrag.core.registries import VECTORSTORES


@VECTORSTORES.register("chroma")
def chroma_vectorstore(
    embedding,
    collection_name: str,
    persist_directory: str | None = None,
    **kw,
):
    """Chroma vector store, persisted to disk when persist_directory is set.

    Args:
        embedding: Embeddings instance (live dependency, not from YAML).
        collection_name: Chroma collection to use. Created if it doesn't exist.
        persist_directory: Folder where Chroma stores its data. If None,
            the store is in-memory and is lost when the process exits.
        **kw: Passed through to Chroma, e.g. collection_metadata or client_settings.

    YAML example:
        vectorstore:
          type: chroma
          params:
            collection_name: injection_molding_v1
            persist_directory: data/processed/chroma
    """
    from langchain_chroma import Chroma
    return Chroma(
        collection_name=collection_name,
        embedding_function=embedding,
        persist_directory=persist_directory,
        **kw,
    )


@VECTORSTORES.register("faiss")
def faiss_vectorstore(
    embedding,
    persist_directory: str | None = None,
    index_name: str = "index",
    allow_dangerous_deserialization: bool = True,
    **kw,
):
    """FAISS vector store (local, in-memory index with optional save/load).

    Loads an existing index from persist_directory if one is there, otherwise
    creates an empty one. FAISS does not save itself: after ingestion, call
    store.save_local(persist_directory, index_name).

    Requires the `faiss-cpu` package.

    Args:
        embedding: Embeddings instance (live dependency).
        persist_directory: Folder holding <index_name>.faiss and <index_name>.pkl.
        index_name: File name stem for the saved index.
        allow_dangerous_deserialization: Loading uses pickle, so only load
            files that your own ingestion wrote.
        **kw: Passed through to the FAISS constructor when creating a new index.

    YAML example:
        vectorstore:
          type: faiss
          params:
            persist_directory: data/processed/faiss
            index_name: injection_molding_v1
    """
    import faiss
    from langchain_community.docstore.in_memory import InMemoryDocstore
    from langchain_community.vectorstores import FAISS

    if persist_directory and os.path.exists(
        os.path.join(persist_directory, f"{index_name}.faiss")
    ):
        return FAISS.load_local(
            persist_directory,
            embedding,
            index_name=index_name,
            allow_dangerous_deserialization=allow_dangerous_deserialization,
        )

    dim = len(embedding.embed_query("dimension probe"))
    return FAISS(
        embedding_function=embedding,
        index=faiss.IndexFlatL2(dim),
        docstore=InMemoryDocstore(),
        index_to_docstore_id={},
        **kw,
    )


@VECTORSTORES.register("qdrant")
def qdrant_vectorstore(
    embedding,
    collection_name: str,
    url: str = "http://localhost:6333",
    **kw,
):
    """Qdrant vector store. Creates the collection if it doesn't exist.

    Args:
        embedding: Embeddings instance (live dependency).
        collection_name: Qdrant collection to use.
        url: Qdrant server URL.
        **kw: Passed through to QdrantVectorStore.

    YAML example:
        vectorstore:
          type: qdrant
          params:
            url: http://localhost:6333
            collection_name: injection_molding_v1
    """
    from langchain_qdrant import QdrantVectorStore
    from qdrant_client import QdrantClient
    from qdrant_client.models import Distance, VectorParams

    client = QdrantClient(url=url)
    if not client.collection_exists(collection_name):
        dim = len(embedding.embed_query("dimension probe"))
        client.create_collection(
            collection_name,
            vectors_config=VectorParams(size=dim, distance=Distance.COSINE),
        )
    return QdrantVectorStore(
        client=client, collection_name=collection_name, embedding=embedding, **kw
    )