"""Privacy-class specs: the extension point for new privacy settings."""

from veillens.privacy.registry import (
    available_specs,
    get_spec_by_name,
    register_spec,
)
from veillens.privacy.spec import PrivacyClassSpec, get_spec

__all__ = [
    "PrivacyClassSpec",
    "get_spec",
    "get_spec_by_name",
    "register_spec",
    "available_specs",
]
