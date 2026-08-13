class PromptBuilder:
    """
    Builds the prompt sent to the LLM.
    """

    @staticmethod
    def build_context(chunks: list[dict]) -> str:
        """
        Convert retrieved chunks into a formatted context block.
        """

        context_parts = []

        for i, chunk in enumerate(chunks, start=1):
            context_parts.append(
                f"""
SOURCE {i}

Title:
{chunk.get("title", "")}

Section:
{chunk.get("section", "")}

Chunk ID:
{chunk.get("chunk_id", "")}

Content:
{chunk.get("text", "")}
"""
            )

        return "\n".join(context_parts)

    @staticmethod
    def build_prompt(
        question: str,
        chunks: list[dict],
    ) -> str:
        """
        Build the complete prompt for the LLM.
        """

        context = PromptBuilder.build_context(chunks)

        return f"""
You are a Confluence Documentation Assistant.

Answer ONLY using the supplied documentation.

Rules:
- Do not invent information.
- If the answer is not present in the documentation, clearly say so.
- Keep the answer concise and accurate.
- Mention the source numbers you used.

====================================================

QUESTION

{question}

====================================================

DOCUMENTATION

{context}

====================================================

Provide your response in this format:

Answer:

Sources Used:
"""