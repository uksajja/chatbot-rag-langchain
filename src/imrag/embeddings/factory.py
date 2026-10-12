from imrag.core.registries import EMBEDDINGS


@EMBEDDINGS.register("huggingface")
def huggingface_embedding(model_name: str = "sentence-transformers/all-MiniLM-L6-v2", **kw):
    """Create a HuggingFace (sentence-transformers) embedding model.

    Args:
        model_name: HuggingFace model id or local path.
        **kw: Passed through to HuggingFaceEmbeddings. Common options:
            model_kwargs: dict passed to the SentenceTransformer constructor,
                e.g. {"device": "cpu"}.
            encode_kwargs: dict passed to encode(),
                e.g. {"normalize_embeddings": True, "batch_size": 64}.
            cache_folder: where downloaded models are cached.

    Returns:
        HuggingFaceEmbeddings

    YAML example:
        embedding:
          type: huggingface
          params:
            model_name: BAAI/bge-base-en-v1.5
            model_kwargs: { device: cpu }
            encode_kwargs: { normalize_embeddings: true }
            cache_folder: ./cache
    """
    from langchain_huggingface import HuggingFaceEmbeddings
    return HuggingFaceEmbeddings(model_name=model_name, **kw)


@EMBEDDINGS.register("openai")
def openai_embedding(model: str = "text-embedding-3-small", **kw):
    """Create an OpenAI embedding model.

    The API key is read from the OPENAI_API_KEY environment variable,
    so it does not belong in YAML.

    Args:
        model: OpenAI embedding model name.
        **kw: Passed through to OpenAIEmbeddings. Common options:
            dimensions: output size for text-embedding-3 models, e.g. 512.

    Returns:
        OpenAIEmbeddings

    YAML example:
        embedding:
          type: openai
          params:
            model: text-embedding-3-small
            dimensions: 512
    """
    from langchain_openai import OpenAIEmbeddings
    return OpenAIEmbeddings(model=model, **kw)

@EMBEDDINGS.register("nvidia")
def nvidia_embedding(model: str= "nemotron-3-embed-1b", **kw ):
    
    """Create an NVIDIA embedding model.

    The API key is read from the NVIDIA_API_KEY environment variable.

    Args:
        model: NVIDIA embedding model name.
        **kw: Passed through to NVIDIAEmbeddings. Common options:
            model_type: type of model (e.g. "passage", "query").

    Returns:
        NVIDIAEmbeddings

    YAML example:
        embedding:
          type: nvidia
          params:
            model: nvidia/NV-Embed-v2
    """
    from langchain_nvidia_ai_endpoints import NVIDIAEmbeddings
    return NVIDIAEmbeddings(model=model, **kw)