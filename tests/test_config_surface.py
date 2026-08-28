"""The shipped configs and the code that reads them must agree, in both directions.

This artifact was trimmed down from a working repository that had accumulated config keys the
pipeline no longer read -- `round1_max_mask_ratio` and `round2_max_mask_ratio` were documented as
working while nothing in the package ever looked at them. A dead key is worse than no key: it
tells a reader that a lever exists, and reviewers reproduce a paper by turning levers.

Both directions are checked, statically, without importing torch:

  1. Every key in a shipped config is named somewhere in `src/`.
  2. Every hard `config['section']['key']` read is satisfied by every shipped config, so no code
     path can raise KeyError halfway through a run.
"""

import ast

import pytest
import yaml

from veillens import paths

SRC = paths.REPO_ROOT / "src"
CONFIGS = sorted((paths.REPO_ROOT / "configs").glob("*.yaml"))
SOURCE_TEXT = "".join(p.read_text() for p in SRC.rglob("*.py"))


def _sections(config_path):
    """Yield (section, key) for every leaf key under a top-level mapping."""
    doc = yaml.safe_load(config_path.read_text())
    for section, value in doc.items():
        if isinstance(value, dict):
            for key in value:
                yield section, key


def _config_reads():
    """Yield (file, lineno, section, key) for every hard two-level config subscript in src/."""
    def base_is_config(node):
        if isinstance(node, ast.Name):
            return node.id in ("config", "cfg")
        if isinstance(node, ast.Attribute):
            return node.attr in ("config", "cfg")
        return False

    for path in sorted(SRC.rglob("*.py")):
        for node in ast.walk(ast.parse(path.read_text())):
            if not isinstance(node, ast.Subscript):
                continue
            inner = node.value
            if not (isinstance(inner, ast.Subscript) and base_is_config(inner.value)):
                continue
            section = getattr(inner.slice, "value", None)
            key = getattr(node.slice, "value", None)
            if isinstance(section, str) and isinstance(key, str):
                yield path.relative_to(SRC), node.lineno, section, key


def test_there_are_configs_to_check():
    assert CONFIGS, "no configs found"


@pytest.mark.parametrize("config_path", CONFIGS, ids=lambda p: p.name)
def test_no_config_key_is_dead(config_path):
    """A key nobody reads is a lever that does not move."""
    dead = [
        f"{section}.{key}"
        for section, key in _sections(config_path)
        if f"'{key}'" not in SOURCE_TEXT and f'"{key}"' not in SOURCE_TEXT
    ]
    assert not dead, (
        f"{config_path.name} declares {dead}, which no source file names. "
        "Either wire the key up or remove it."
    )


@pytest.mark.parametrize("config_path", CONFIGS, ids=lambda p: p.name)
def test_every_hard_config_read_is_satisfied(config_path):
    doc = yaml.safe_load(config_path.read_text())
    missing = []
    for where, lineno, section, key in _config_reads():
        if section not in doc:
            missing.append(f"{where}:{lineno} reads [{section!r}][{key!r}] but there is no {section}: section")
        elif isinstance(doc[section], dict) and key not in doc[section]:
            missing.append(f"{where}:{lineno} reads [{section!r}][{key!r}], absent from {section}:")
    assert not missing, f"{config_path.name} would raise KeyError:\n  " + "\n  ".join(missing)
