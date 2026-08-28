"""
The privacy-class registry.

Maps a name to a :class:`~veillens.privacy.spec.PrivacyClassSpec`. Adding a class means adding
one entry here.

Note the legacy aliases. The paper's vocabulary is *compositional privacy*, but the code and
several thousand cached result JSONs were written against the older name ``personal_info``.
Renaming the directory without keeping the alias would invalidate those caches, so both spellings
resolve to the same spec and old configs keep working untouched.
"""

from __future__ import annotations

from typing import Dict, Iterable

from veillens.privacy.spec import (
    COMPOSITIONAL_SPEC,
    GEOLOCATION_SPEC,
    SOCIAL_RELATION_SPEC,
    PrivacyClassSpec,
)

__all__ = ["register_spec", "get_spec_by_name", "available_specs", "canonical_name"]


#: Canonical name -> spec.
_SPECS: Dict[str, PrivacyClassSpec] = {
    "geolocation": GEOLOCATION_SPEC,
    "compositional": COMPOSITIONAL_SPEC,
    "social_relation": SOCIAL_RELATION_SPEC,
}

#: Older spellings that must keep resolving. Do not remove these; cached results depend on them.
LEGACY_ALIASES: Dict[str, str] = {
    "personal_info": "compositional",
    "pi": "compositional",
    "geo": "geolocation",
    "relationship": "social_relation",
    "social_relationship": "social_relation",
}


def canonical_name(name: str) -> str:
    """Fold a legacy alias onto its canonical name. Unknown names pass through unchanged."""
    return LEGACY_ALIASES.get(name, name)


def register_spec(spec: PrivacyClassSpec, *, aliases: Iterable[str] = ()) -> None:
    """
    Register a new privacy class.

    Args:
        spec: the spec to register; ``spec.name`` becomes its canonical name.
        aliases: any additional names that should resolve to it.

    Raises:
        ValueError: if the name is already taken, which would silently shadow an existing class.
    """
    if spec.name in _SPECS:
        raise ValueError(f"Privacy class {spec.name!r} is already registered.")
    _SPECS[spec.name] = spec
    for alias in aliases:
        LEGACY_ALIASES[alias] = spec.name


def get_spec_by_name(name: str) -> PrivacyClassSpec:
    """Look up a spec by canonical name or alias."""
    resolved = canonical_name(name)
    if resolved not in _SPECS:
        raise ValueError(
            f"Unknown privacy class {name!r}. Available: {sorted(_SPECS)}"
        )
    return _SPECS[resolved]


def available_specs() -> Dict[str, PrivacyClassSpec]:
    """All registered specs, keyed by canonical name."""
    return dict(_SPECS)
