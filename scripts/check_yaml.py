"""Validate YAML files passed by pre-commit without constructing unsafe objects."""

from __future__ import annotations

import sys
from pathlib import Path

import yaml


class _TaggedSafeLoader(yaml.SafeLoader):
    """Safe loader that preserves CloudFormation-style tagged values as plain data."""


def _construct_tagged_value(loader, _tag_suffix, node):
    if isinstance(node, yaml.ScalarNode):
        return loader.construct_scalar(node)
    if isinstance(node, yaml.SequenceNode):
        return loader.construct_sequence(node)
    return loader.construct_mapping(node)


_TaggedSafeLoader.add_multi_constructor("!", _construct_tagged_value)


def main(paths: list[str]) -> int:
    failed = False
    for raw_path in paths:
        path = Path(raw_path)
        try:
            list(yaml.load_all(path.read_text(encoding="utf-8"), Loader=_TaggedSafeLoader))
        except (OSError, UnicodeError, yaml.YAMLError) as exc:
            print(f"{path}: {exc}", file=sys.stderr)
            failed = True
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
