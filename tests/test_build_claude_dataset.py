import pandas as pd

from ml_pipeline.build_claude_dataset import agreed, drop_overlap, relabel, split_eval


def _labels(rows: list[tuple[str, bool | None, str | None]], variant: str) -> pd.DataFrame:
    return pd.DataFrame(rows, columns=["id", "usable", "claude_condition"]).assign(variant=variant)


def test_relabel_keeps_usable_rows_with_claude_category() -> None:
    rows = pd.DataFrame({"row_id": ["1", "2", "3"], "text": ["a", "b", "c"], "condition": "X"})
    labels = _labels([("1", True, "Skin Conditions"), ("2", False, "none"), ("3", None, None)], "a")

    out = relabel(rows, labels)

    assert out["row_id"].tolist() == ["1"]
    assert out["condition"].tolist() == ["Skin Conditions"]
    assert out["label_source"].tolist() == ["claude-a"]


def test_agreed_requires_both_passes_usable_and_equal() -> None:
    a = _labels(
        [
            ("1", True, "Ear Conditions"),
            ("2", True, "Ear Conditions"),
            ("3", True, "Eye Conditions"),
        ],
        "a",
    )
    b = _labels(
        [("1", True, "Ear Conditions"), ("2", True, "Skin Conditions"), ("3", False, "none")], "b"
    )

    assert agreed(a, b)["id"].tolist() == ["1"]


def test_split_eval_caps_each_class_and_keeps_the_rest() -> None:
    pool = pd.DataFrame({"text": [f"t{i}" for i in range(7)], "condition": ["A"] * 5 + ["B"] * 2})

    eval_df, rest = split_eval(pool, per_class=3, seed=0)

    assert eval_df["condition"].value_counts().to_dict() == {"A": 3, "B": 2}
    assert len(rest) == 2 and set(rest.index).isdisjoint(eval_df.index)


def test_drop_overlap_removes_exact_and_near_duplicates_of_eval_rows() -> None:
    eval_df = pd.DataFrame({"text": ["My dog keeps scratching his left ear all night long"]})
    train = pd.DataFrame(
        {
            "text": [
                "my dog keeps scratching his left ear, all night long!",
                "My dog keeps scratching his left ear all night long now",
                "Cat is sneezing and has a runny nose",
            ]
        }
    )

    assert drop_overlap(train, eval_df, jaccard=0.8)["text"].tolist() == [
        "Cat is sneezing and has a runny nose"
    ]
