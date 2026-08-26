"""
Config loading, path resolution, and manifest portability.

These are the parts that used to be hardcoded to one machine, so they are the parts most worth
pinning down.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from veillens import paths
from veillens.config import load_config, substitute_paths, validate_config
from veillens.datasets.schema import load_manifest

CONFIG_FILES = sorted((paths.REPO_ROOT / "configs").rglob("*.yaml"))


def test_repo_root_looks_right():
    assert (paths.REPO_ROOT / "pyproject.toml").is_file()
    assert paths.PROMPTS.is_dir()


def test_resolve_leaves_absolute_paths_alone(tmp_path):
    assert paths.resolve(tmp_path) == tmp_path


def test_resolve_treats_relative_as_repo_relative():
    assert paths.resolve("prompts") == paths.REPO_ROOT / "prompts"


def test_substitute_paths_walks_nested_structures():
    out = substitute_paths(
        {"a": ["${paths.repo_root}/x", {"b": "${paths.data}/y"}], "n": 1},
        {"repo_root": "/R", "data": "/D"},
    )
    assert out == {"a": ["/R/x", {"b": "/D/y"}], "n": 1}


def test_unknown_placeholder_is_an_error():
    with pytest.raises(KeyError):
        substitute_paths("${paths.nope}/x", {"data": "/D"})


@pytest.mark.parametrize("config_path", CONFIG_FILES, ids=lambda p: str(p.name))
def test_every_shipped_config_loads(config_path):
    config = load_config(config_path, check_dataset=False)
    validate_config(config, check_dataset=False)


@pytest.mark.parametrize("config_path", CONFIG_FILES, ids=lambda p: str(p.name))
def test_no_config_contains_a_machine_specific_path(config_path):
    text = config_path.read_text()
    # A config must name roots through ${paths.*}, never an absolute path on one machine.
    for bad in ("/home/", "/Users/", "/mnt/", "/scratch/", "/data/"):
        assert bad not in text, f"{config_path.name} still hardcodes {bad}"


@pytest.mark.parametrize("config_path", CONFIG_FILES, ids=lambda p: str(p.name))
def test_configs_follow_the_data_root(config_path, monkeypatch):
    """Every path a config produces must move when VEILLENS_DATA moves."""
    monkeypatch.setenv("VEILLENS_DATA", "/tmp/somewhere-else")
    monkeypatch.setenv("VEILLENS_RUNS", "/tmp/runs-elsewhere")
    config = load_config(config_path, check_dataset=False)
    for value in (config["dataset"]["dataset_json_file"], config["output"]["base_dir"]):
        assert "${" not in value, f"unsubstituted placeholder: {value}"
        assert Path(value).is_absolute()


MANIFESTS = sorted((paths.REPO_ROOT / "manifests").glob("*.json"))


@pytest.mark.parametrize("manifest", MANIFESTS, ids=lambda p: p.name)
def test_manifests_are_schema_v2(manifest):
    raw = json.loads(manifest.read_text())
    assert isinstance(raw, dict) and raw.get("schema_version") == 2, (
        f"{manifest.name} is not a schema-v2 manifest (needs schema_version: 2 and items[])"
    )


@pytest.mark.parametrize("manifest", MANIFESTS, ids=lambda p: p.name)
def test_manifest_paths_follow_the_data_root(manifest, monkeypatch, tmp_path):
    monkeypatch.setenv("VEILLENS_DATA", str(tmp_path))
    items = load_manifest(manifest)
    assert items, f"{manifest.name} is empty"
    for item in items[:20]:
        assert item["image_path"].startswith(str(tmp_path)), (
            f"{manifest.name} did not relocate: {item['image_path']}"
        )
        assert "image_id" in item, "image_id must survive migration; caches are keyed on it"
