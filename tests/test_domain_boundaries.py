"""
Structural guards for the package boundaries described in
openspec/changes/restructure-package-by-domain.

These assert on the import graph and on module size rather than on behaviour, so
they fail the moment a shortcut re-couples the packages.
"""

from __future__ import annotations

import ast
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
APP_ROOT = REPO_ROOT / "app"


def app_modules() -> list[Path]:
    return sorted(p for p in APP_ROOT.rglob("*.py") if "__pycache__" not in p.parts)


def imported_modules(path: Path) -> set[str]:
    """Every dotted module name imported by `path`, including inside functions."""
    tree = ast.parse(path.read_text())
    names: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            names.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
            names.add(node.module)
    return names


def test_serving_package_never_imports_the_training_pipeline():
    offenders = {
        path.relative_to(REPO_ROOT).as_posix(): sorted(
            name for name in imported_modules(path) if name.split(".")[0] == "ml_pipeline"
        )
        for path in app_modules()
    }
    offenders = {path: names for path, names in offenders.items() if names}

    assert offenders == {}, (
        "app/ must stay deployable on its own — both Lambda images copy app/ only. "
        f"Offending imports: {offenders}"
    )


DOMAIN_PACKAGES = {"wellness", "triage", "feeding"}
SHARED_PACKAGES = {"inference", "llm", "domain"}


def package_of(path: Path) -> str | None:
    parts = path.relative_to(APP_ROOT).parts
    return parts[0] if len(parts) > 1 else None


def app_imports(path: Path) -> set[str]:
    """First-level `app.<x>` packages that `path` imports from."""
    return {
        name.split(".")[1]
        for name in imported_modules(path)
        if name.startswith("app.") and len(name.split(".")) > 1
    }


def test_domain_packages_do_not_import_each_other():
    """Wellness, triage, and feeding are independent — shared concepts belong in
    app/domain or a shared package, not in a sibling's namespace."""
    offenders = {
        path.relative_to(REPO_ROOT).as_posix(): sorted(
            app_imports(path) & (DOMAIN_PACKAGES - {package_of(path)})
        )
        for path in app_modules()
        if package_of(path) in DOMAIN_PACKAGES
    }
    offenders = {path: names for path, names in offenders.items() if names}

    assert offenders == {}, f"sibling-domain imports: {offenders}"


def test_shared_packages_do_not_import_domain_packages():
    """Dependencies point inward: api -> domain packages -> shared packages."""
    offenders = {
        path.relative_to(REPO_ROOT).as_posix(): sorted(app_imports(path) & DOMAIN_PACKAGES)
        for path in app_modules()
        if package_of(path) in SHARED_PACKAGES
    }
    offenders = {path: names for path, names in offenders.items() if names}

    assert offenders == {}, f"shared packages reaching into a domain: {offenders}"


def test_schemas_are_owned_by_their_domain():
    """The aggregate app/schemas.py shim is gone; each model has one home."""
    assert not (APP_ROOT / "schemas.py").exists(), "the app/schemas.py shim must stay deleted"

    offenders = [
        path.relative_to(REPO_ROOT).as_posix()
        for path in app_modules()
        if "app.schemas" in imported_modules(path)
    ]
    assert offenders == [], f"still importing the removed shim: {offenders}"


def test_condition_metadata_imports_without_an_inference_backend():
    """The condition map is domain data, so it must not drag in torch/onnxruntime."""
    script = (
        "import sys;"
        "sys.modules['torch'] = None;"
        "sys.modules['onnxruntime'] = None;"
        "import app.domain.conditions as c;"
        "print(len(c.CONDITION_METADATA))"
    )
    result = subprocess.run(
        [sys.executable, "-c", script],
        capture_output=True,
        text=True,
        cwd=REPO_ROOT,
    )

    assert result.returncode == 0, result.stderr
    assert int(result.stdout.strip()) == 16


def test_condition_metadata_covers_every_trained_class():
    from app.domain.conditions import CONDITION_METADATA, get_condition_metadata

    assert len(CONDITION_METADATA) == 16
    for condition in CONDITION_METADATA:
        meta = get_condition_metadata(condition)
        assert meta.urgency is not None
        assert meta.specialist is not None
        assert meta.home_advice, f"{condition} has no home advice"


MODULE_LINE_LIMIT = 250

# Exempt: a lookup table, not logic. Splitting data across files to satisfy a
# limit aimed at branching code would hurt readability.
SIZE_EXEMPT = {"domain/conditions.py"}

# Modules still awaiting their split. This list may only shrink — a module that
# drops below the limit is removed here, and nothing new is ever added.
PENDING_SPLIT: set[str] = set()


@pytest.mark.parametrize("path", app_modules(), ids=lambda p: p.relative_to(APP_ROOT).as_posix())
def test_module_stays_under_the_size_limit(path: Path):
    """
    Cap on module size — this refactor exists because a 1332-line service had
    seven reasons to change.
    """
    relative = path.relative_to(APP_ROOT).as_posix()
    if relative in SIZE_EXEMPT or relative in PENDING_SPLIT:
        pytest.skip(f"{relative}: known-oversized, tracked in PENDING_SPLIT")

    line_count = len(path.read_text().splitlines())
    assert line_count <= MODULE_LINE_LIMIT, (
        f"{path.relative_to(REPO_ROOT)} is {line_count} lines; split it into "
        f"single-responsibility modules (limit {MODULE_LINE_LIMIT})"
    )


def test_pending_split_list_only_shrinks():
    """A module that has been split must be dropped from PENDING_SPLIT, so the
    allowlist can never quietly turn into a permanent exemption."""
    already_small = {
        relative
        for relative in PENDING_SPLIT
        if (APP_ROOT / relative).exists()
        and len((APP_ROOT / relative).read_text().splitlines()) <= MODULE_LINE_LIMIT
    }
    missing = {relative for relative in PENDING_SPLIT if not (APP_ROOT / relative).exists()}

    assert (
        already_small == set()
    ), f"these are now under the limit — remove them from PENDING_SPLIT: {sorted(already_small)}"
    assert (
        missing == set()
    ), f"these no longer exist — remove them from PENDING_SPLIT: {sorted(missing)}"
