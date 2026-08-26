"""
Every registered privacy class must be internally complete.

Parametrised over the registry, so a newly added privacy class is covered automatically — a typo
in a prompt path fails here in a second rather than three hours into a GPU run.
"""

from __future__ import annotations

import pytest

from veillens.paths import resolve
from veillens.privacy.registry import LEGACY_ALIASES, available_specs, get_spec_by_name

SPEC_NAMES = sorted(available_specs())

REQUIRED_CALLABLES = (
    "feature_passes_mask_level",
    "format_round1_context",
    "build_round2_prompt",
    "compute_error",
    "extract_prediction_summary",
    "normalize_ground_truth",
    "serialize_ground_truth",
    "format_mask_level_label",
)

PATH_FIELDS = (
    "round1_schema_path",
    "round1_prompt_path",
    "round2_schema_path",
    "round2_prompt_path",
    "eval_schema_path",
    "eval_prompt_path",
)


def test_registry_is_not_empty():
    assert SPEC_NAMES, "no privacy classes registered"


@pytest.mark.parametrize("name", SPEC_NAMES)
def test_spec_name_matches_registry_key(name):
    """A spec registered under one name but calling itself another confuses every log line."""
    assert get_spec_by_name(name).name == name


@pytest.mark.parametrize("name", SPEC_NAMES)
@pytest.mark.parametrize("field", PATH_FIELDS)
def test_referenced_files_exist(name, field):
    path = getattr(get_spec_by_name(name), field)
    assert resolve(path).is_file(), f"{name}.{field} points at a missing file: {path}"


@pytest.mark.parametrize("name", SPEC_NAMES)
@pytest.mark.parametrize("attr", REQUIRED_CALLABLES)
def test_required_callables_present(name, attr):
    fn = getattr(get_spec_by_name(name), attr)
    assert callable(fn), f"{name}.{attr} is not callable"


@pytest.mark.parametrize("name", SPEC_NAMES)
def test_prompts_and_schemas_load(name):
    spec = get_spec_by_name(name)
    for mode in ("r1", "r2", "eval"):
        schema = spec.get_schema(mode)
        assert isinstance(schema, dict) and schema, f"{name}: {mode} schema is empty"
    for getter in ("get_round1_prompt", "get_round2_prompt", "get_eval_prompt"):
        text = getattr(spec, getter)()
        assert text.strip(), f"{name}: {getter}() returned nothing"


@pytest.mark.parametrize("alias,canonical", sorted(LEGACY_ALIASES.items()))
def test_legacy_aliases_resolve(alias, canonical):
    """Cached results are keyed on the older names; these must never stop resolving."""
    assert get_spec_by_name(alias).name == canonical


def test_unknown_class_raises_with_a_useful_message():
    with pytest.raises(ValueError) as excinfo:
        get_spec_by_name("does-not-exist")
    assert "Available" in str(excinfo.value)
