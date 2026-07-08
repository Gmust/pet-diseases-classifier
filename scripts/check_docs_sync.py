#!/usr/bin/env python3
"""Detect API endpoint and SAM-default drift in maintained documentation."""

from __future__ import annotations

import re
import sys
from pathlib import Path

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.check_openapi import main as check_openapi  # noqa: E402
from scripts.check_yaml import _TaggedSafeLoader  # noqa: E402

API_DOC = Path("docs/api-reference.md")
DEPLOY_DOC = Path("docs/deploy-onnx.md")
TEMPLATE = Path("template.yaml")


def documented_endpoints(text: str) -> set[tuple[str, str]]:
    return set(re.findall(r"^## `((?:GET|POST)) ([^`]+)`$", text, flags=re.MULTILINE))


def active_endpoints() -> set[tuple[str, str]]:
    from app.main import app

    excluded = {"/openapi.json", "/docs", "/docs/oauth2-redirect", "/redoc"}
    return {
        (method, route.path)
        for route in app.routes
        if route.path not in excluded
        for method in route.methods
        if method in {"GET", "POST"}
    }


def deployment_expectations(template: dict, documentation: str) -> list[str]:
    parameters = template["Parameters"]
    timeout = template["Globals"]["Function"]["Timeout"]
    retention = template["Resources"]["PetCareAiFunctionLogGroup"]["Properties"]["RetentionInDays"]
    expected = (
        f"defaults to `{parameters['FunctionMemory']['Default']}` MB",
        f"defaults to `{parameters['ReservedConcurrency']['Default']}`",
        f"`Timeout={timeout}`",
        f"for {retention} days",
    )
    return [token for token in expected if token not in documentation]


def main() -> int:
    api_text = API_DOC.read_text(encoding="utf-8")
    actual = active_endpoints()
    documented = documented_endpoints(api_text)
    failures: list[str] = []
    if actual != documented:
        failures.append(
            f"API documentation mismatch: actual={sorted(actual)}, docs={sorted(documented)}"
        )

    template = yaml.load(TEMPLATE.read_text(encoding="utf-8"), Loader=_TaggedSafeLoader)
    missing = deployment_expectations(template, DEPLOY_DOC.read_text(encoding="utf-8"))
    if missing:
        failures.append(f"Deployment documentation is missing current defaults: {missing}")
    if failures:
        print("\n".join(failures))
        return 1
    return check_openapi()


if __name__ == "__main__":
    raise SystemExit(main())
