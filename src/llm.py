from __future__ import annotations

import os

# Small non-reasoning model: no hidden reasoning tokens, supports temperature=0.
OPENAI_LLM_MODEL = "gpt-4.1-nano"


class OpenAIChatLLM:
    """OpenAI chat-completions backed llm_fn for KnowledgeBaseAgent.

    Kept deliberately cheap: small model, temperature 0 and a hard cap on output tokens.
    """

    def __init__(self, model_name: str | None = None, max_tokens: int = 256) -> None:
        from openai import OpenAI

        self.model_name = model_name or os.getenv("OPENAI_LLM_MODEL", OPENAI_LLM_MODEL)
        self.max_tokens = max_tokens
        self._backend_name = self.model_name
        self.usage_tokens = 0
        self.client = OpenAI()

    def __call__(self, prompt: str) -> str:
        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=[{"role": "user", "content": prompt}],
            temperature=0,
            max_tokens=self.max_tokens,
        )
        if response.usage:
            self.usage_tokens += response.usage.total_tokens
        return (response.choices[0].message.content or "").strip()
