from typing import Any


class ResponseParser:

    @staticmethod
    def build_response(
        question: str,
        answer: str,
        retrieved_chunks: list[dict],
        retrieval_time: float,
        generation_time: float,
    ) -> dict[str, Any]:

        scores = [
            chunk["score"]
            for chunk in retrieved_chunks
        ]

        highest = max(scores) if scores else 0.0
        lowest = min(scores) if scores else 0.0
        average = (
            sum(scores) / len(scores)
            if scores
            else 0.0
        )

        unique_pages = len(
            {
                chunk["page_id"]
                for chunk in retrieved_chunks
            }
        )

        sources = []

        for chunk in retrieved_chunks:

            sources.append(
                {
                    "parent_page": chunk.get(
                        "parent_page",
                        "Unknown",
                    ),
                    "child_page": chunk.get(
                        "title",
                        "",
                    ),
                    "section": chunk.get(
                        "section",
                        "",
                    ),
                    "chunk_id": chunk.get(
                        "chunk_id",
                    ),
                    "score": round(
                        chunk["score"],
                        4,
                    ),
                    "source_url": chunk.get(
                        "source_url",
                        "",
                    ),
                }
            )

        confidence = "LOW"

        if highest >= 0.85:
            confidence = "HIGH"

        elif highest >= 0.65:
            confidence = "MEDIUM"

        return {

            "question": question,

            "answer": answer,

            "confidence": confidence,

            "sources": sources,

            "metrics": {

                "chunks_retrieved": len(
                    retrieved_chunks
                ),

                "highest_score": round(
                    highest,
                    4,
                ),

                "lowest_score": round(
                    lowest,
                    4,
                ),

                "average_score": round(
                    average,
                    4,
                ),

                "unique_pages": unique_pages,

                "retrieval_time_ms": round(
                    retrieval_time * 1000,
                    2,
                ),

                "generation_time_ms": round(
                    generation_time * 1000,
                    2,
                ),

                "total_time_ms": round(
                    (
                        retrieval_time
                        + generation_time
                    )
                    * 1000,
                    2,
                ),
            },
        }