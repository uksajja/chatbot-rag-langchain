
from langchain_core.document_loaders import BaseLoader
from langchain_core.documents import BaseDocumentTransformer
from langchain_core.embeddings import Embeddings
from langchain_core.vectorstores import VectorStore
from langchain_core.retrievers import BaseRetriever
from langchain_core.language_models import BaseChatModel

from imrag.core.registry import Registry

"""
Registries for different components of the system.
Use these registries to register and retrieve different types of components, such as chunkers, embeddings, vectorstores, retrievers, and language models.
The registries are used to map string names to factory functions that create instances of the corresponding components.
NOTE: 
    Each registry has a name and an expected type.
    The expected_type parameter in the Registry class is used for type checking to ensure that the objects created by the factory functions are of the expected type. 
    If an expected type is provided, the registry will check if the created object is an instance of that type and raise a TypeError if it is not. 
    This helps to catch errors early and ensures that the components being registered and retrieved are of the correct type.
    For custom components, you can set expected_type to None to skip type checking, but it is recommended to provide the expected type for safety.
    example: 
    If you are registering a custom chunker, 
    and your custom chunker class is MyCustomChunker, then in the registry you should do:

    ```python
    CHUNKERS.register("my_custom_chunker", MyCustomChunker)
    ```
    if MyCustomChunker is a subclass of BaseDocumentTransformer, then the expected_type for the CHUNKERS registry should be BaseDocumentTransformer.
    Otherwise, if you set expected_type to None, the registry will not perform type checking and you may encounter runtime errors if the object is not of the expected type.
"""

LOADERS      = Registry("document_loader", BaseLoader)
CHUNKERS     = Registry("chunker", BaseDocumentTransformer)
EMBEDDINGS   = Registry("embedding", Embeddings)
VECTORSTORES = Registry("vectorstore", VectorStore)
RETRIEVERS   = Registry("retriever", BaseRetriever)
RERANKERS    = Registry("reranker")
LLMS         = Registry("llm", BaseChatModel)