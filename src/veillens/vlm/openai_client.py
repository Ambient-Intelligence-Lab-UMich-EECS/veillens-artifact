"""
OpenAI Responses-API client.

This is the measurement instrument: the final evaluation pass runs through it, and its numbers are
compared against runs made before the local backend existed. The request it builds is the one
:mod:`veillens.reason.oracle` built inline before the split -- same model, same
``reasoning={"effort": "low"}``, same strict ``json_schema``, same token ceiling. Keep it that way.
"""

from __future__ import annotations

import os
from typing import Any, Dict, Optional

import openai

from veillens.vlm.base import encode_image_data_url

__all__ = ["OpenAIClient"]


class OpenAIClient:
    """Talks to OpenAI's Responses API with strict structured output."""

    def __init__(
        self,
        model: str = "gpt-5",
        api_key: Optional[str] = None,
        max_output_tokens: int = 8000,
        timeout: int = 300,
        reasoning_effort: str = "low",
        max_image_edge: Optional[int] = None,
        **_ignored: Any,
    ):
        # client_from_config forwards every config key as a kwarg, so an option meant for the
        # local backend (reasoning.max_tokens, reasoning.presence_penalty) aimed at this one
        # would otherwise vanish without trace.
        if _ignored:
            print(f"[openai] Ignoring unknown config key(s): {sorted(_ignored)}")
        self.model = model
        self.api_key = api_key
        self.max_output_tokens = max_output_tokens
        self.timeout = timeout
        self.reasoning_effort = reasoning_effort
        self.max_image_edge = max_image_edge
        self._client: Optional[openai.AsyncOpenAI] = None

    def _api(self) -> openai.AsyncOpenAI:
        """
        Build the SDK client once and reuse it.

        Lazily, for two reasons: the constructor raises without a key, and a config that never
        reaches OpenAI (``--dry-run`` on a laptop) should not need one.
        """
        if self._client is None:
            self._client = openai.AsyncOpenAI(api_key=self.api_key or os.getenv("OPENAI_API_KEY"))
        return self._client

    def encode_image(self, image_path: str) -> str:
        return encode_image_data_url(image_path, self.max_image_edge)

    async def infer(
        self,
        image_data_url: str,
        prompt: str,
        schema: Dict[str, Any],
        schema_name: str,
        trace_label: Optional[str] = None,
    ) -> str:
        # trace_label is ignored: the Responses API returns no reasoning text to record, only an
        # opaque token count. The local backend is the one with a <think> stream worth keeping.
        response = await self._api().responses.create(
            model=self.model,
            reasoning={"effort": self.reasoning_effort},
            text={
                "format": {
                    "type": "json_schema",
                    "name": schema_name,
                    "schema": schema,
                    "strict": True,
                }
            },
            input=[
                {
                    "role": "user",
                    "content": [
                        {"type": "input_text", "text": prompt},
                        {"type": "input_image", "image_url": image_data_url},
                    ],
                }
            ],
            max_output_tokens=self.max_output_tokens,
            timeout=self.timeout,
        )
        return response.output_text
