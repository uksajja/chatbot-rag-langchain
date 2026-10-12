# core/bootstrap.py


import importlib

_PLUGIN_MODULES = [
    "imrag.document_loaders.factory",
    "imrag.chunking.factory",
    "imrag.embeddings.factory",
    "imrag.vectorstores.factory",
    "imrag.retrieval.factory",
    "imrag.retrieval.reranker",
    "imrag.llms.factory"
]


def register_all() -> None:
    """
    Registers all the plugins in the _PLUGIN_MODULES list.
    This function imports each module in the list, which triggers the registration of
    the corresponding components (chunking, embeddings, vectorstores, retrievers, llms).
    """
    for module_name in _PLUGIN_MODULES:
        importlib.import_module(module_name)
