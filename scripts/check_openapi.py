"""Check or update the canonical OpenAPI contract fingerprint."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

DEFAULT_FINGERPRINT_PATH = Path("docs/openapi.sha256")


def openapi_sha256() -> str:
    from app.main import app

    payload = json.dumps(
        app.openapi(),
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--write", action="store_true", help="Update the tracked fingerprint.")
    parser.add_argument("--fingerprint", type=Path, default=DEFAULT_FINGERPRINT_PATH)
    args = parser.parse_args()

    actual = openapi_sha256()
    if args.write:
        args.fingerprint.parent.mkdir(parents=True, exist_ok=True)
        args.fingerprint.write_text(f"{actual}\n", encoding="utf-8")
        print(f"Updated {args.fingerprint}: {actual}")
        return 0

    try:
        expected = args.fingerprint.read_text(encoding="utf-8").strip()
    except FileNotFoundError:
        print(
            f"OpenAPI fingerprint is missing: {args.fingerprint}. "
            "Run `python scripts/check_openapi.py --write` after reviewing the contract.",
        )
        return 1

    if actual != expected:
        print(f"OpenAPI contract changed: expected {expected}, generated {actual}.")
        print("Review the generated contract, then run `python scripts/check_openapi.py --write`.")
        return 1

    print(f"OpenAPI contract matches {actual}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
