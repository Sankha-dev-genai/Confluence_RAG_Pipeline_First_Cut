from openai import OpenAI
from openai import (
    APIConnectionError,
    AuthenticationError,
    RateLimitError,
    APIStatusError,
)

import app.core.config as config
from app.llm.prompt_builder import PromptBuilder


class LLMGenerator:
    """
    Handles communication with OpenAI.
    """

    def __init__(
        self,
        model: str = "gpt-4.1-mini",
        temperature: float = 0.0,
        max_tokens: int = 700,
    ):

        self.client = OpenAI(
            api_key=config.OPENAI_API_KEY,
        )

        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens

    def generate(
        self,
        question: str,
        chunks: list[dict],
    ) -> dict:

        prompt = PromptBuilder.build_prompt(
            question=question,
            chunks=chunks,
        )

        try:

            response = self.client.chat.completions.create(
                model=self.model,
                temperature=self.temperature,
                max_tokens=self.max_tokens,
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "You answer ONLY from the supplied "
                            "Confluence documentation."
                        ),
                    },
                    {
                        "role": "user",
                        "content": prompt,
                    },
                ],
            )

            return {
                "success": True,
                "answer": response.choices[0].message.content.strip(),
            }

        except AuthenticationError:

            return {
                "success": False,
                "error": "Invalid OpenAI API Key.",
            }

        except RateLimitError:

            return {
                "success": False,
                "error": "OpenAI rate limit exceeded.",
            }

        except APIConnectionError:

            return {
                "success": False,
                "error": "Unable to connect to OpenAI.",
            }

        except APIStatusError as e:

            return {
                "success": False,
                "error": f"OpenAI Error: {e.status_code}",
            }

        except Exception as e:

            return {
                "success": False,
                "error": str(e),
            }