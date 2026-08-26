"""
Factory for creating VLM clients by backend name.

A name-to-class table plus one builder, so a config string is all it takes to swap the model
doing the reasoning.

    from veillens.vlm.registry import build_client
    client = build_client("openai", model="gpt-5")
"""

from __future__ import annotations

from typing import Any, Dict, Mapping, Type

from veillens.vlm.base import VLMClient
from veillens.vlm.openai_client import OpenAIClient

__all__ = ["CLIENT_CLASSES", "build_client", "client_from_config", "get_available_backends"]

#: Registry of client classes by backend name.
CLIENT_CLASSES: Dict[str, Type[Any]] = {
    "openai": OpenAIClient,
}

#: Config keys that name a backend rather than configure one.
_NON_KWARGS = ("backend", "concurrent_requests")


def get_available_backends() -> list:
    """Get list of available backend names."""
    return list(CLIENT_CLASSES)


def build_client(backend: str, **kwargs: Any) -> VLMClient:
    """
    Construct the client for *backend*, passing *kwargs* to its constructor.

    Raises:
        ValueError: no such backend.
    """
    try:
        cls = CLIENT_CLASSES[backend]
    except KeyError:
        raise ValueError(
            f"Unknown VLM backend {backend!r}. Available: {', '.join(sorted(CLIENT_CLASSES))}"
        ) from None
    return cls(**kwargs)


def client_from_config(
    section: Mapping[str, Any],
    default_backend: str = "openai",
    **overrides: Any,
) -> VLMClient:
    """
    Build a client from a config section such as ``reasoning:`` or ``evaluation:``.

    Keys that steer the pipeline rather than the client (``backend``, ``concurrent_requests``)
    are filtered out; everything else is forwarded, so a backend can grow an option without this
    function changing.

    ``overrides`` are for values the caller knows and the config file cannot: ``trace_dir`` is the
    live run's output directory, which is only resolved once the driver has built its paths. A
    ``None`` override is dropped rather than shadowing the section.
    """
    kwargs = {k: v for k, v in section.items() if k not in _NON_KWARGS}
    kwargs.update({k: v for k, v in overrides.items() if v is not None})
    return build_client(section.get("backend", default_backend), **kwargs)
