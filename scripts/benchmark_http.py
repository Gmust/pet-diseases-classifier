#!/usr/bin/env python3
"""Measure HTTP latency and throughput of a running service instance.

Usage:
    uvicorn app.main:app --port 8000   # in another shell
    python scripts/benchmark_http.py --requests 200 --concurrency 1,4,16 \
        --output docs/benchmarks/http-<date>.json

Each scenario gets warm-up requests (excluded from stats), then `--requests`
requests per concurrency level. Latency is client-side wall time per request.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import platform
import statistics
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import httpx

SCENARIOS: dict[str, tuple[str, str, dict[str, Any] | None]] = {
    "health": ("GET", "/health", None),
    "predict": ("POST", "/predict", {"text": "my dog has been vomiting and wont eat for 2 days"}),
    "predict_red_flag": ("POST", "/predict", {"text": "my dog collapsed and is not breathing"}),
    "chat_health": (
        "POST",
        "/chat",
        {
            "symptomSummary": "The cat is sneezing.",
            "petType": "cat",
            "messages": [
                {"role": "user", "content": "my cat is sneezing"},
                {"role": "assistant", "content": "Does your cat have a runny nose?"},
                {"role": "user", "content": "yes, runny nose and watery eyes, and coughing"},
            ],
        },
    ),
    "chat_general": (
        "POST",
        "/chat",
        {
            "messages": [{"role": "user", "content": "how often should I brush a Persian cat?"}],
            "petType": "cat",
        },
    ),
    "wellness": (
        "POST",
        "/wellness",
        {
            "pet": {"species": "dog", "breed": "Labrador", "ageMonths": 36, "weightKg": 28.5},
            "activity": {
                "avgStepsPerDay": 8000,
                "avgActiveMinutesPerDay": 60,
                "avgSleepHoursPerDay": 12,
                "daysTracked": 7,
            },
            "feeding": {"avgMealsPerDay": 2, "consistencyDays": 7},
            "preventiveCare": {"recentVetVisit": True, "vaccinationsUpToDate": True},
            "previousScore": 82,
        },
    ),
    "feeding_summary_100": (
        "POST",
        "/feeding-summary",
        {
            "pets": [
                {
                    "petId": f"pet-{i}",
                    "species": "dog",
                    "weightKg": 5 + i % 40,
                    "ageMonths": 6 + i % 120,
                    "products": [{"name": "Kibble", "calories": 300 + i}],
                }
                for i in range(100)
            ]
        },
    ),
}


def summarize(latencies_ms: list[float], errors: int, wall_s: float) -> dict[str, Any]:
    q = statistics.quantiles(latencies_ms, n=100, method="inclusive")
    return {
        "requests": len(latencies_ms),
        "errors": errors,
        "rps": round(len(latencies_ms) / wall_s, 1),
        "mean_ms": round(statistics.fmean(latencies_ms), 1),
        "min_ms": round(min(latencies_ms), 1),
        "p50_ms": round(q[49], 1),
        "p95_ms": round(q[94], 1),
        "p99_ms": round(q[98], 1),
        "max_ms": round(max(latencies_ms), 1),
    }


async def run_level(
    client: httpx.AsyncClient, scenario: str, total: int, concurrency: int
) -> dict[str, Any]:
    method, path, body = SCENARIOS[scenario]
    latencies: list[float] = []
    errors = 0
    remaining = iter(range(total))

    async def worker() -> None:
        nonlocal errors
        for _ in remaining:
            start = time.perf_counter()
            response = await client.request(method, path, json=body)
            latencies.append((time.perf_counter() - start) * 1000)
            errors += not response.is_success

    wall_start = time.perf_counter()
    await asyncio.gather(*(worker() for _ in range(concurrency)))
    return summarize(latencies, errors, time.perf_counter() - wall_start)


async def main_async(args: argparse.Namespace) -> list[dict[str, Any]]:
    headers = {"X-API-Key": args.api_key} if args.api_key else {}
    limits = httpx.Limits(max_connections=max(args.concurrency))
    results: list[dict[str, Any]] = []
    async with httpx.AsyncClient(
        base_url=args.base_url, headers=headers, timeout=60, limits=limits
    ) as client:
        for scenario in args.scenarios:
            method, path, body = SCENARIOS[scenario]
            for _ in range(args.warmup):
                await client.request(method, path, json=body)
            for concurrency in args.concurrency:
                row = {"scenario": scenario, "concurrency": concurrency}
                row |= await run_level(client, scenario, args.requests, concurrency)
                results.append(row)
                print(
                    f"{scenario:<20} c={concurrency:<3} p50={row['p50_ms']:>8} ms  "
                    f"p95={row['p95_ms']:>8} ms  rps={row['rps']:>7}  errors={row['errors']}",
                    flush=True,
                )
    return results


def to_markdown(results: list[dict[str, Any]]) -> str:
    cols = ["scenario", "concurrency", "requests", "errors", "rps"]
    cols += ["mean_ms", "p50_ms", "p95_ms", "p99_ms", "max_ms"]
    lines = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    lines += ["| " + " | ".join(str(r[c]) for c in cols) + " |" for r in results]
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default="http://127.0.0.1:8000")
    parser.add_argument("--api-key")
    parser.add_argument("--requests", type=int, default=200)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument(
        "--concurrency", type=lambda s: [int(x) for x in s.split(",")], default=[1, 4, 16]
    )
    parser.add_argument(
        "--scenarios", type=lambda s: s.split(","), default=list(SCENARIOS), help="comma list"
    )
    parser.add_argument("--label", default="", help="free-text run description for the report")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.requests < 2 or args.warmup < 0:
        raise SystemExit("--requests must be >= 2 and --warmup >= 0")
    unknown = set(args.scenarios) - set(SCENARIOS)
    if unknown:
        raise SystemExit(f"unknown scenarios: {sorted(unknown)}")

    results = asyncio.run(main_async(args))
    print("\n" + to_markdown(results))
    if args.output:
        report = {
            "schema_version": 1,
            "measured_at": datetime.now(UTC).isoformat(),
            "label": args.label,
            "base_url": args.base_url,
            "client_host": f"{platform.system()} {platform.machine()} {platform.processor()}",
            "python": platform.python_version(),
            "method": "client-side wall time per request; warm-up excluded",
            "results": results,
        }
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
