from imrag.core.registries import RERANKERS


class CrossEncoderReranker:
    def __init__(self, model_name: str, top_n: int):
        from sentence_transformers import CrossEncoder
        self.model = CrossEncoder(model_name)
        self.top_n = top_n

    def rerank(self, query, docs):
        pairs = [[query, doc.page_content] for doc in docs]
        scores = self.model.predict(pairs)
        scored_docs = zip(scores, docs)
        sorted_docs = sorted(scored_docs, key=lambda x: x[0], reverse=True)
        return [doc for score, doc in sorted_docs[: self.top_n]]

@RERANKERS.register("cross-encoder")
def cross_encoder_reranker(model_name: str, top_n: int, **kw):
    """
    Local cross-encoder reranker (sentence-transformers)
    """
    return CrossEncoderReranker(model_name, top_n)

class NemotronReranker:
    def __init__(self, model_name: str, top_n: int):
        from langchain_nvidia_ai_endpoints import NVIDIARerank
        self.reranker = NVIDIARerank(model=model_name, top_n=top_n)

    def rerank(self, query, docs):
        return list(self.reranker.compress_documents(documents=docs, query=query))

@RERANKERS.register("nemotron")
def nemotron_reranker(model_name: str, top_n: int, **kw):
    """
    Nemotron reranker
    """
    return NemotronReranker(model_name, top_n)

