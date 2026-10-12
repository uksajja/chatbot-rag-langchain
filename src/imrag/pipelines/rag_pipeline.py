from __future__ import annotations

import logging
import time
from typing import Iterator

from langchain_core.language_models import BaseChatModel
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.retrievers import BaseRetriever

from imrag.retrieval.reranker import CrossEncoderReranker, NemotronReranker
from imrag.core.schemas import RAGResponse, SourceChunk, Timings
from imrag.generation.context import format_context, to_sources

logger = logging.getLogger(__name__)

NOT_FOUND = (
    "I could not find this in the provided documents. "
    "Please rephrase the question or check the source manuals."
)


class RAGPipeline:
    """The only class the UI, API, and eval harness should call."""

    def __init__(
        self,
        retriever: BaseRetriever,
        llm: BaseChatModel,
        prompt: ChatPromptTemplate,
        reranker: CrossEncoderReranker | NemotronReranker | None = None,
        min_sources: int = 1,
    ):
        self.retriever = retriever
        self.reranker = reranker
        self.min_sources = min_sources
        self.chain = prompt | llm | StrOutputParser()      # LCEL

    def retrieve(self, question: str) -> tuple[list[SourceChunk], Timings]:
        """Retrieve, optionally rerank, then number the results.

        Everything that changes which chunks reach the model must happen
        before to_sources(), so the [n] numbers match the UI and the evals.
        """
        t0 = time.perf_counter()
        docs = self.retriever.invoke(question)
        t1 = time.perf_counter()

        rerank_ms = 0.0
        if self.reranker and docs:
            docs = self.reranker.rerank(question, docs)
            rerank_ms = (time.perf_counter() - t1) * 1000

        return to_sources(docs), Timings(
            retrieve_ms=(t1 - t0) * 1000, rerank_ms=rerank_ms
        )

    def answer(self, question: str) -> RAGResponse:
        question = self._check(question)
        t_start = time.perf_counter()
        sources, timings = self.retrieve(question)

        if len(sources) < self.min_sources:
            timings.total_ms = (time.perf_counter() - t_start) * 1000
            return RAGResponse(question=question, answer=NOT_FOUND,
                               sources=[], timings=timings)

        t_gen = time.perf_counter()
        text = self.chain.invoke(
            {"context": format_context(sources), "question": question}
        )
        now = time.perf_counter()
        timings.generate_ms = (now - t_gen) * 1000
        timings.total_ms = (now - t_start) * 1000
        logger.info("answered in %.0f ms (%s)", timings.total_ms, timings.model_dump())
        return RAGResponse(question=question, answer=text,
                           sources=sources, timings=timings)

    def stream(self, question: str) -> tuple[list[SourceChunk], Iterator[str]]:
        """Return (sources, token iterator) so a UI can show sources and stream text."""
        question = self._check(question)
        sources, _ = self.retrieve(question)
        if len(sources) < self.min_sources:
            return [], iter([NOT_FOUND])
        tokens = self.chain.stream(
            {"context": format_context(sources), "question": question}
        )
        return sources, tokens

    @staticmethod
    def _check(question: str) -> str:
        question = question.strip()
        if not question:
            raise ValueError("Question is empty")
        return question