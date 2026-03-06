"""
llm.py
------
Groq LLM client wrapper.

Why Groq?
  - Free tier with very fast inference (runs LLaMA on custom hardware)
  - Compatible with the OpenAI SDK interface for easy switching
  - We already use it for LearnOS, so the team knows it well

This module keeps all LLM interaction in one place so it's easy to swap
the provider (e.g., to OpenAI or Anthropic) without changing other modules.
"""

from groq import Groq
from agent.config import GROQ_API_KEY, GROQ_MODEL, LLM_TEMPERATURE, LLM_MAX_TOKENS
from agent.prompt import SYSTEM_PROMPT


class GroqLLMClient:
    """
    A thin wrapper around the Groq Python SDK.

    Responsibilities:
      - Manage the Groq client connection
      - Format messages correctly for the chat completions API
      - Return clean text responses (strip whitespace, handle errors)
    """

    def __init__(self):
        if not GROQ_API_KEY:
            raise ValueError(
                "GROQ_API_KEY is not set. Please add it to your .env file."
            )
        self.client = Groq(api_key=GROQ_API_KEY)
        self.model = GROQ_MODEL
        print(f"[LLM] Groq client initialised with model: {self.model}")

    def generate(self, user_prompt: str, system_prompt: str = SYSTEM_PROMPT) -> str:
        """
        Send a prompt to the LLM and return the text response.

        Args:
            user_prompt:   The full prompt including retrieved context and query.
            system_prompt: Instructions that define the assistant's persona and rules.

        Returns:
            The model's text response as a string.
        """
        try:
            chat_completion = self.client.chat.completions.create(
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user",   "content": user_prompt},
                ],
                model=self.model,
                temperature=LLM_TEMPERATURE,   # Low = factual, High = creative
                max_tokens=LLM_MAX_TOKENS,
            )
            response_text = chat_completion.choices[0].message.content
            return response_text.strip()

        except Exception as e:
            # Graceful degradation: log the error and return a safe message
            print(f"[LLM] Error calling Groq API: {e}")
            return (
                "I'm experiencing a technical issue right now. "
                "Please try again in a moment or contact us at support@pixelflow.in."
            )
