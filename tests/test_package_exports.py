"""Package-level re-exports must name things that exist.

`veillens/edit/__init__.py` once re-exported a class that had been renamed, so `import
veillens.edit` raised ImportError while every test still passed -- the suite imports submodules
directly and never went through the package. The check is done on the AST rather than by
importing, because importing `veillens.edit` pulls in torch and diffusers and this suite must run
without a GPU or model weights.
"""

import ast
from pathlib import Path

import pytest

from veillens import paths

SRC = paths.REPO_ROOT / "src"
INITS = sorted(SRC.rglob("__init__.py"))


def _module_path(dotted: str) -> Path:
    return SRC / (dotted.replace(".", "/") + ".py")


def _toplevel_names(tree: ast.Module) -> set:
    names = set()
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            names.add(node.name)
        elif isinstance(node, ast.Assign):
            names.update(t.id for t in node.targets if isinstance(t, ast.Name))
        elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
            names.add(node.target.id)
        elif isinstance(node, (ast.Import, ast.ImportFrom)):
            names.update(a.asname or a.name.split(".")[0] for a in node.names)
    return names


def test_there_is_something_to_check():
    assert INITS, "no package __init__.py files found"


@pytest.mark.parametrize("init", INITS, ids=lambda p: str(p.relative_to(SRC)))
def test_reexported_names_are_defined_in_their_source_module(init):
    tree = ast.parse(init.read_text())
    for node in tree.body:
        # `from __future__ import annotations` and anything else outside the package has no
        # file under src/ to resolve against; only our own re-exports are the point here.
        if not isinstance(node, ast.ImportFrom) or not (node.module or "").startswith("veillens"):
            continue
        source = _module_path(node.module)
        if not source.exists():
            pytest.fail(f"{init.relative_to(SRC)} imports from missing module {node.module}")
        defined = _toplevel_names(ast.parse(source.read_text()))
        for alias in node.names:
            assert alias.name in defined, (
                f"{init.relative_to(SRC)} re-exports '{alias.name}', "
                f"which {node.module} does not define"
            )


@pytest.mark.parametrize("init", INITS, ids=lambda p: str(p.relative_to(SRC)))
def test_dunder_all_matches_what_the_package_imports(init):
    tree = ast.parse(init.read_text())
    declared = None
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(
            isinstance(t, ast.Name) and t.id == "__all__" for t in node.targets
        ):
            declared = {e.value for e in node.value.elts}
    if declared is None:
        return
    available = _toplevel_names(tree)
    missing = declared - available
    assert not missing, f"{init.relative_to(SRC)} lists {sorted(missing)} in __all__ but never binds them"
