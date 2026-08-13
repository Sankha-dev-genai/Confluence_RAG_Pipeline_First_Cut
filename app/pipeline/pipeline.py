from __future__ import annotations

import time
from typing import Any

import app.core.config as config
from app.retrieval.retriever import Retriever
from app.retrieval.citation_builder import CitationBuilder


class RAGPipeline:
    """
    End-to-end query pipeline.

        question
           -> Retriever (FAISS semantic search + metadata rerank + filter)
           -> CitationBuilder (evidence)
           -> LLMGenerator (grounded answer)   [optional]
           -> structured response (answer + citations + metrics)

    If no OpenAI key is configured (or use_llm=False), the pipeline still
    runs retrieval and returns citations — so the app is always demoable.
    """

    HIGH_CONFIDENCE = 0.55
    MEDIUM_CONFIDENCE = 0.40

    def __init__(
        self,
        top_k: int = 20,
        min_score: float = 0.20,
        context_k: int = 5,
        use_llm: bool = True,
        model: str = "gpt-4.1-mini",
        use_hybrid: bool | None = None,
        hybrid_alpha: float | None = None,
    ) -> None:
        self.context_k = context_k
        self.retriever = Retriever(
            top_k=top_k, min_score=min_score,
            use_hybrid=use_hybrid, hybrid_alpha=hybrid_alpha,
        )

        self.use_llm = bool(use_llm and config.OPENAI_API_KEY)
        self._generator = None
        self._model = model

    # ---- lazy LLM so retrieval/eval never require an API key ----
    @property
    def generator(self):
        if self._generator is None:
            from app.llm.generator import LLMGenerator
            self._generator = LLMGenerator(model=self._model)
        return self._generator

    def _confidence(self, top_similarity: float) -> str:
        if top_similarity >= self.HIGH_CONFIDENCE:
            return "HIGH"
        if top_similarity >= self.MEDIUM_CONFIDENCE:
            return "MEDIUM"
        return "LOW"

    def retrieve(self, question: str) -> list[dict[str, Any]]:
        """Retrieval only (used by the evaluation harness — no LLM cost)."""
        return self.retriever.retrieve(question)

    def answer(self, question: str) -> dict[str, Any]:
        t0 = time.perf_counter()
        retrieved = self.retriever.retrieve(question)
        retrieval_time = time.perf_counter() - t0

        context = retrieved[: self.context_k]
        citations = CitationBuilder.build(context)

        similarities = [c["similarity"] for c in citations] or [0.0]
        top_similarity = max(similarities)

        answer_text = ""
        answer_generated = False
        generation_time = 0.0
        error = None

        if not context:
            answer_text = (
                "I could not find anything relevant to this question in the "
                "Confluence documentation."
            )
        elif self.use_llm:
            t1 = time.perf_counter()
            result = self.generator.generate(question, context)
            generation_time = time.perf_counter() - t1
            if result.get("success"):
                answer_text = result["answer"]
                answer_generated = True
            else:
                error = result.get("error")
                answer_text = (
                    "Answer generation failed; showing retrieved sources only. "
                    f"({error})"
                )
        else:
            answer_text = (
                "LLM generation is disabled (no OpenAI key configured). "
                "The most relevant sources are shown below."
            )

        return {
            "question": question,
            "answer": answer_text,
            "answer_generated": answer_generated,
            "error": error,
            "confidence": self._confidence(top_similarity),
            "citations": citations,
            "retrieved": retrieved,
            "metrics": {
                "chunks_retrieved": len(retrieved),
                "chunks_used": len(context),
                "top_similarity": round(top_similarity, 4),
                "avg_similarity": round(sum(similarities) / len(similarities), 4),
                "unique_pages": len({c["page_id"] for c in citations}),
                "retrieval_time_ms": round(retrieval_time * 1000, 2),
                "generation_time_ms": round(generation_time * 1000, 2),
                "total_time_ms": round(
                    (retrieval_time + generation_time) * 1000, 2
                ),
            },
        }
