from imrag.core.registries import CHUNKERS


@CHUNKERS.register("recursive")
def recursive_text_splitter(
    chunk_size: int = 800,
    chunk_overlap: int = 120,
    separators: list[str] | None = None,
    embedding=None,
    **kw,
):
    """Split on a priority list of separators, falling back to the next one.

    Args:
        chunk_size: Maximum chunk length (characters by default).
        chunk_overlap: Overlap between consecutive chunks.
        separators: Split priority. Default is ["\\n\\n", "\\n", " ", ""].
        embedding: Accepted and ignored (see note below).
        **kw: Passed through to RecursiveCharacterTextSplitter,
            e.g. keep_separator, add_start_index, strip_whitespace.

    YAML example:
        chunking:
          type: recursive
          params:
            chunk_size: 800
            chunk_overlap: 120
    """
    from langchain_text_splitters import RecursiveCharacterTextSplitter
    return RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        **({"separators": separators} if separators else {}),
        **kw,
    )


@CHUNKERS.register("fixed")
def fixed_text_splitter(
    chunk_size: int = 800,
    chunk_overlap: int = 0,
    separator: str = "\n\n",
    embedding=None,
    **kw,
):
    """Split on a single separator, then merge pieces up to chunk_size.

    A piece longer than chunk_size is kept whole (with a warning), so chunks can
    exceed chunk_size if the text has no separator inside long passages.

    Args:
        chunk_size: Target maximum chunk length.
        chunk_overlap: Overlap between consecutive chunks.
        separator: The one separator used for splitting.
        embedding: Accepted and ignored.
        **kw: Passed through to CharacterTextSplitter.
    """
    from langchain_text_splitters import CharacterTextSplitter
    return CharacterTextSplitter(
        chunk_size=chunk_size, chunk_overlap=chunk_overlap, separator=separator, **kw
    )


@CHUNKERS.register("semantic")
def semantic_chunker(
    embedding,
    breakpoint_threshold_type: str = "percentile",
    breakpoint_threshold_amount: float | None = None,
    min_chunk_size: int | None = None,
    **kw,
):
    """Split where the embedding similarity between adjacent sentences drops.

    Requires `langchain-experimental`. SemanticChunker has no chunk_size or
    chunk_overlap: chunk length depends on the text, so very long or very short
    chunks are possible. Use min_chunk_size to avoid tiny ones.

    Args:
        embedding: Embeddings instance (live dependency, required).
        breakpoint_threshold_type: "percentile", "standard_deviation",
            "interquartile", or "gradient".
        breakpoint_threshold_amount: Threshold for the chosen type. If None,
            the library default for that type is used.
        min_chunk_size: Minimum chunk length.
        **kw: Passed through to SemanticChunker, e.g. buffer_size.

    YAML example:
        chunking:
          type: semantic
          params:
            breakpoint_threshold_type: percentile
            breakpoint_threshold_amount: 90
    """
    from langchain_experimental.text_splitter import SemanticChunker

    options = {"breakpoint_threshold_type": breakpoint_threshold_type, **kw}
    if breakpoint_threshold_amount is not None:
        options["breakpoint_threshold_amount"] = breakpoint_threshold_amount
    if min_chunk_size is not None:
        options["min_chunk_size"] = min_chunk_size
    return SemanticChunker(embedding, **options)


@CHUNKERS.register("spacy")
def spacy_text_splitter(
    chunk_size: int = 800,
    chunk_overlap: int = 0,
    pipeline: str = "en_core_web_sm",
    embedding=None,
    **kw,
):
    """Sentence-aware splitting using spaCy.

    Requires `spacy` and the model, installed with:
        python -m spacy download en_core_web_sm

    Args:
        chunk_size, chunk_overlap: Length limits for merged sentences.
        pipeline: spaCy model name.
        embedding: Accepted and ignored.
        **kw: Passed through to SpacyTextSplitter, e.g. separator.
    YAML example:
        chunking:
          type: spacy
          params:
            pipeline: en_core_web_sm
            chunk_size: 800
            chunk_overlap: 0
    """
    from langchain_text_splitters import SpacyTextSplitter
    return SpacyTextSplitter(
        pipeline=pipeline, chunk_size=chunk_size, chunk_overlap=chunk_overlap, **kw
    )


@CHUNKERS.register("nltk")
def nltk_text_splitter(
    chunk_size: int = 800,
    chunk_overlap: int = 0,
    language: str = "english",
    embedding=None,
    **kw,
):
    """Sentence-aware splitting using NLTK.

    Requires `nltk` and its sentence tokenizer data (the "punkt" resource,
    downloaded once with nltk.download).

    Args:
        chunk_size, chunk_overlap: Length limits for merged sentences.
        language: Language of the sentence tokenizer.
        embedding: Accepted and ignored.
        **kw: Passed through to NLTKTextSplitter, e.g. separator.
    YAML example:
        chunking:
          type: nltk
          params:
            language: english
            chunk_size: 800
            chunk_overlap: 0
    """
    from langchain_text_splitters import NLTKTextSplitter
    return NLTKTextSplitter(
        language=language, chunk_size=chunk_size, chunk_overlap=chunk_overlap, **kw
    )