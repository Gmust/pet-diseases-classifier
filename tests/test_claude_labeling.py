import json
from types import SimpleNamespace

import pandas as pd
import pytest

from ml_pipeline.claude_labeling import (
    LABELS,
    build_generate_requests,
    build_label_requests,
    cmd_label,
    generated_rows,
    parse_args,
    parse_rows,
    request_fingerprint,
    resumable_batch_id,
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


def test_generated_rows_get_stable_content_row_ids() -> None:
    messages = {
        "gen-00000": _message('{"rows": [{"text": "cat coughs"}, {"text": "cat coughs"}]}'),
        "gen-00001": _message('{"rows": [{"text": "dog limps"}]}'),
    }
    specs = {
        "gen-00000": {"condition": "Respiratory Conditions"},
        "gen-00001": {"condition": "Musculoskeletal Conditions"},
    }

    first, again = generated_rows(messages, specs), generated_rows(messages, specs)

    assert first["text"].tolist() == ["cat coughs", "dog limps"]
    assert first["row_id"].is_unique and first["row_id"].tolist() == again["row_id"].tolist()


def test_resume_refuses_state_saved_for_different_requests(tmp_path) -> None:
    state = tmp_path / "out.parquet.batch.json"
    assert resumable_batch_id(state, "abc") is None

    state.write_text(json.dumps({"batch_id": "msgbatch_1", "fingerprint": "abc"}))
    assert resumable_batch_id(state, "abc") == "msgbatch_1"
    with pytest.raises(SystemExit):
        resumable_batch_id(state, "other")
    assert request_fingerprint([{"a": 1}]) != request_fingerprint([{"a": 2}])


def test_label_keep_usable_from_limits_rows_to_earlier_usable_ones(tmp_path, capsys) -> None:
    rows = pd.DataFrame({"row_id": [f"r{i}" for i in range(5)], "text": ["dog coughs"] * 5})
    prior = pd.DataFrame({"id": ["r1", "r3", "r4"], "usable": [True, False, True]})
    rows.to_parquet(tmp_path / "in.parquet")
    prior.to_parquet(tmp_path / "a.parquet")

    cmd_label(
        parse_args(
            [
                "label",
                "--input",
                str(tmp_path / "in.parquet"),
                "--id-col",
                "row_id",
                "--keep-usable-from",
                str(tmp_path / "a.parquet"),
                "--rows",
                "1",
                "--out",
                str(tmp_path / "out.parquet"),
            ]
        )
    )

    assert capsys.readouterr().out.startswith("2 requests")
