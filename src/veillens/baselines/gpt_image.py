"""
`gpt-image` — hand the whole job to an image-editing model, in one call.

The prompt is `prompts/<class>/defense_image_edit_prompt.txt`: the same reasoning prompt as the
loop's round 1, with only its ending swapped from "emit JSON" to "emit the edited image". Keeping
those two files in step is what makes the comparison a comparison.

This is the one baseline that makes paid network calls — one image edit per image. Requires
`OPENAI_API_KEY`.
"""

from __future__ import annotations

import base64
import io
import os
import time
from typing import Any, Dict, List, Optional, Tuple

from PIL import Image

from veillens.baselines.base import Baseline
from veillens.paths import resolve

__all__ = ["GPTImageDefense", "GPTImageRefusal"]


class GPTImageRefusal(Exception):
    """
    The model declined to edit the image.

    Terminal, not transient: the same request will be declined again, so a retry only burns time
    and money. Counted separately from network failures in the run summary, because a baseline
    that silently skipped part of the dataset is not comparable to one that processed all of it.
    """


class GPTImageDefense(Baseline):
    name = "gpt-image"
    summary = "One gpt-image-2 privacy edit per frame, from the round-1 reasoning prompt"

    def __init__(self, model: str = "gpt-image-2", prompt: str = "", size: str = "auto",
                 quality: str = "high", max_retries: int = 5, timeout: int = 300,
                 input_fidelity: Optional[str] = None, concurrency: int = 4,
                 **params: Any) -> None:
        super().__init__(model=model, prompt=prompt, size=size, quality=quality,
                         max_retries=max_retries, timeout=timeout,
                         input_fidelity=input_fidelity, concurrency=concurrency, **params)
        self.model = model
        self.prompt_path = prompt
        self.size = size
        self.quality = quality
        self.max_retries = max_retries
        self.timeout = timeout
        self.input_fidelity = input_fidelity
        self.workers = max(1, int(concurrency))
        self.prompt = ""
        self.client = None
        self._openai = None

    @classmethod
    def check_prerequisites(cls, params: Dict[str, Any]) -> List[str]:
        problems = []
        prompt = params.get("prompt")
        if not prompt or not resolve(prompt).is_file():
            problems.append(f"missing image-edit prompt: {prompt or '<unset>'}")
        if not os.getenv("OPENAI_API_KEY"):
            problems.append("OPENAI_API_KEY is not set; gpt-image makes paid API calls "
                            "(see .env.example)")
        return problems

    def prepare(self, items: List[Dict[str, Any]]) -> None:
        import openai

        self.prompt = resolve(self.prompt_path).read_text()
        self._openai = openai
        self.client = openai.OpenAI(api_key=os.environ["OPENAI_API_KEY"])

    def _edit_once(self, image_path: str) -> Tuple[bytes, Dict[str, Any]]:
        """One API call. Returns the PNG bytes and whatever usage the response reported."""
        kwargs: Dict[str, Any] = {
            "model": self.model,
            "prompt": self.prompt,
            "n": 1,
            "size": self.size,
            "quality": self.quality,
            "timeout": self.timeout,
        }
        # Passed only when set, so an older SDK does not choke on an unknown keyword.
        if self.input_fidelity is not None:
            kwargs["input_fidelity"] = self.input_fidelity

        with open(image_path, "rb") as handle:
            response = self.client.images.edit(image=handle, **kwargs)

        if not getattr(response, "data", None):
            raise RuntimeError("API returned no image data")
        payload = getattr(response.data[0], "b64_json", None)
        if not payload:
            raise RuntimeError("API returned a result with no b64_json payload")

        usage = getattr(response, "usage", None)
        reported = {}
        if usage is not None:
            for field in ("input_tokens", "output_tokens", "total_tokens"):
                value = getattr(usage, field, None)
                if value is not None:
                    reported[field] = value
        return base64.b64decode(payload), reported

    def apply(self, image_path: str, image_id: str) -> Tuple[Image.Image, Dict[str, Any]]:
        with Image.open(image_path) as source:
            original_size = source.size

        started = time.time()
        png: Optional[bytes] = None
        usage: Dict[str, Any] = {}
        last_error = "unknown error"
        attempts = 0

        for attempt in range(self.max_retries):
            attempts = attempt + 1
            try:
                png, usage = self._edit_once(image_path)
                break
            except self._openai.BadRequestError as exc:
                # Content policy, moderation, or a malformed request. Retrying will not help.
                raise GPTImageRefusal(f"{type(exc).__name__}: {exc}") from exc
            except Exception as exc:  # noqa: BLE001 - SDK and network exceptions are varied
                last_error = f"{type(exc).__name__}: {exc}"
                if attempt < self.max_retries - 1:
                    time.sleep(2.0 * (2 ** attempt))

        if png is None:
            raise RuntimeError(f"gave up after {self.max_retries} attempts: {last_error}")

        image = Image.open(io.BytesIO(png))
        image.load()
        image = image.convert("RGB")
        returned_size = image.size

        # The model picks its own output resolution. Everything downstream -- the perceptual
        # metrics especially -- compares against the source pixel for pixel, so a mismatch is not
        # a smaller image, it is no measurement at all.
        resized = returned_size != original_size
        if resized:
            image = image.resize(original_size, Image.LANCZOS)

        stats = {
            "model": self.model,
            "prompt_chars": len(self.prompt),
            "size_requested": self.size,
            "quality": self.quality,
            "input_fidelity": self.input_fidelity,
            "original_size": list(original_size),
            "returned_size": list(returned_size),
            "resized": resized,
            "latency_s": round(time.time() - started, 2),
            "attempts": attempts,
            "usage": usage,
        }
        return image, stats
