"""
The VLM client interface.

A client is the only thing that knows how to talk to one provider. It takes an image, a prompt
and a JSON schema, and returns the raw JSON text the model produced. Everything above it --
prompt selection, schema selection, parsing, retries, the privacy spec -- is provider-agnostic
and lives in :mod:`veillens.reason.oracle`.

This is where a new provider goes: implement :class:`VLMClient` and add it to
:data:`veillens.vlm.registry.CLIENT_CLASSES`. Nothing above this layer changes.

Provider quirks belong *here*, never in the caller. Two live examples: OpenAI tolerates a wrong
``data:`` MIME type by sniffing the bytes and vLLM does not, so :func:`encode_image_data_url`
declares the real one; and grammar-constrained decoders reject the stray ``additionalProperties``
that the shipped schemas put on string properties, so the local client strips it.
"""

from __future__ import annotations

import base64
import io
from pathlib import Path
from typing import Any, Dict, Optional, Protocol, runtime_checkable

from PIL import Image

__all__ = ["VLMClient", "VLMError", "encode_image_data_url"]


class VLMError(RuntimeError):
    """
    A backend failed in a way the caller should hear about.

    Distinct from returning ``None``: the oracle turns transport failures into ``None`` because a
    dropped request is survivable, but a truncated reply or an unreachable server is a
    configuration problem that would otherwise show up as a run's worth of silently missing
    predictions.
    """


#: Pillow format name -> MIME type, for the formats a dataset might realistically hold.
_MIME_BY_FORMAT = {
    "JPEG": "image/jpeg",
    "PNG": "image/png",
    "WEBP": "image/webp",
    "GIF": "image/gif",
    "BMP": "image/bmp",
    "TIFF": "image/tiff",
}


def encode_image_data_url(image_path: str | Path, max_edge: Optional[int] = None) -> str:
    """
    Read an image and return it as an RFC 2397 ``data:`` URL.

    Args:
        image_path: image to encode.
        max_edge: if set and the longest side exceeds it, downscale to it first. Vision tokens
            scale with pixel count, so this is the main lever on prompt size for a local server
            with a finite KV cache.

    The MIME type is derived from the file's actual format rather than assumed. Cityscapes ships
    2048x1024 PNGs; the pre-refactor code declared every one of them ``image/jpeg``.
    """
    with Image.open(image_path) as img:
        fmt = (img.format or "").upper()
        mime = _MIME_BY_FORMAT.get(fmt)
        longest = max(img.size)

        if max_edge is not None and longest > max_edge:
            scale = max_edge / longest
            size = (max(1, round(img.width * scale)), max(1, round(img.height * scale)))
            buffer = io.BytesIO()
            img.convert("RGB").resize(size, Image.LANCZOS).save(buffer, format="PNG", optimize=True)
            payload, mime = buffer.getvalue(), "image/png"
        elif mime is None:
            # Unrecognised container (or a format the server may not accept). Normalise to PNG
            # rather than declaring a MIME type we are guessing at.
            buffer = io.BytesIO()
            img.convert("RGB").save(buffer, format="PNG", optimize=True)
            payload, mime = buffer.getvalue(), "image/png"
        else:
            payload = Path(image_path).read_bytes()

    return f"data:{mime};base64,{base64.b64encode(payload).decode('utf-8')}"


@runtime_checkable
class VLMClient(Protocol):
    """
    One provider's half of a structured-output vision request.

    Split in two so the oracle can encode outside its concurrency semaphore -- file I/O and a
    possible resize should not hold a slot that exists to rate-limit the network.
    """

    def encode_image(self, image_path: str) -> str:
        """Return a ``data:`` URL for *image_path*, at this backend's preferred resolution."""
        ...

    async def infer(
        self,
        image_data_url: str,
        prompt: str,
        schema: Dict[str, Any],
        schema_name: str,
        trace_label: Optional[str] = None,
    ) -> str:
        """
        Run one schema-constrained request and return the raw JSON **text**.

        Parsing is the caller's job: the oracle owns the truncated-response debug dump and wants
        the unparsed string when ``json.loads`` fails.

        ``trace_label`` names this call for any per-call record the backend keeps -- the local
        client uses it to file the model's ``<think>`` text under the image it was thinking about.
        Backends that produce nothing worth tracing ignore it.

        Raises:
            VLMError: the backend produced no usable content (e.g. hit the token ceiling).
        """
        ...
