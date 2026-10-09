from types import SimpleNamespace

from ml_pipeline.claude_labeling import (
    LABELS,
    build_generate_requests,
    build_label_requests,
    parse_rows,
)


def _message(text: str, stop_reason: str = "end_turn") -> SimpleNamespace:
    return SimpleNamespace(
        stop_reason=stop_reason, content=[SimpleNamespace(type="text", text=text)]
    )


def test_label_requests_cover_every_row_once_and_variants_differ() -> None:
    ids = [f"r{i}" for i in range(95)]
    texts = [f"my dog coughs {i}" for i in range(95)]
    requests_a, members_a = build_label_requests(ids, texts, "a", 40, "medium")
    _, members_b = build_label_requests(ids, texts, "b", 40, "medium")

    assert len(requests_a) == 3
    assert sorted(i for chunk in members_a.values() for i in chunk) == sorted(ids)
    assert list(members_a.values())[0] != list(members_b.values())[0]
    enum = requests_a[0]["params"]["output_config"]["format"]["schema"]["properties"]["rows"]
    assert enum["items"]["properties"]["condition"]["enum"] == [*LABELS, "none"]


def test_generate_requests_split_counts_and_vary_briefs() -> None:
    requests, specs = build_generate_requests({"Respiratory Conditions": 70}, 30, "medium")

    assert len(requests) == 3
    assert "Write 10 distinct" in requests[-1]["params"]["messages"][0]["content"]
    assert len({(s["species"], s["style"]) for s in specs.values()}) == 3
    assert all(s["condition"] == "Respiratory Conditions" for s in specs.values())


def test_parse_rows_rejects_refusals_and_broken_json() -> None:
    assert parse_rows(_message('{"rows": [{"id": "1"}]}')) == [{"id": "1"}]
    assert parse_rows(_message('{"rows": []}', stop_reason="refusal")) is None
    assert parse_rows(_message('{"rows": [', stop_reason="max_tokens")) is None
    assert parse_rows(_message("not json")) is None
