import argparse
import ast
import json
from pathlib import Path

import pandas as pd


REQUIRED_COLUMNS = {"text", "condition", "record_type"}
EMPTY_LABEL_VALUES = {"", "none", "null", "nan", "na", "n/a", "unknown", "[]"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Merge multiple pet-health datasets into one training file.")
    parser.add_argument(
        "--inputs",
        nargs="+",
        required=True,
        help="Input dataset files (.csv or .parquet).",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="data/merged_pet_dataset.parquet",
        help="Output dataset path (.csv or .parquet).",
    )
    parser.add_argument(
        "--label-map-path",
        type=str,
        default=None,
        help="Optional JSON file mapping source condition labels to normalized labels.",
    )
    parser.add_argument(
        "--dedupe-mode",
        type=str,
        choices=["none", "text", "text_condition"],
        default="text_condition",
        help="How to deduplicate rows after merge.",
    )
    parser.add_argument(
        "--min-text-length",
        type=int,
        default=5,
        help="Drop rows where normalized text length is below this value.",
    )
    parser.add_argument(
        "--add-source-column",
        action="store_true",
        help="Add a 'source' column with the input file name for traceability.",
    )
    return parser.parse_args()


def _read_dataset(path: Path) -> pd.DataFrame:
    suffix = path.suffix.lower()
    if suffix == ".csv":
        return pd.read_csv(path)
    if suffix in {".parquet", ".pq"}:
        return pd.read_parquet(path)
    raise ValueError(f"Unsupported input file format: {path}")


def _extract_condition(value: object) -> str | None:
    if pd.isna(value):
        return None

    if isinstance(value, list):
        for item in value:
            if isinstance(item, str) and item.strip():
                candidate = item.strip()
                if candidate.lower() not in EMPTY_LABEL_VALUES:
                    return candidate
        return None

    if isinstance(value, str):
        stripped = value.strip()
        if stripped.lower() in EMPTY_LABEL_VALUES:
            return None

        if stripped.startswith("[") and stripped.endswith("]"):
            try:
                parsed = ast.literal_eval(stripped)
            except (ValueError, SyntaxError):
                return stripped
            if isinstance(parsed, list):
                for item in parsed:
                    if isinstance(item, str) and item.strip():
                        candidate = item.strip()
                        if candidate.lower() not in EMPTY_LABEL_VALUES:
                            return candidate
                return None
            parsed_text = str(parsed).strip()
            if parsed_text.lower() in EMPTY_LABEL_VALUES:
                return None
            return parsed_text if parsed_text else None

        return stripped

    as_text = str(value).strip()
    if as_text.lower() in EMPTY_LABEL_VALUES:
        return None
    return as_text if as_text else None


def _coerce_to_required_schema(df: pd.DataFrame) -> pd.DataFrame:
    if REQUIRED_COLUMNS.issubset(df.columns):
        return df[["text", "condition", "record_type"]].copy()

    text_col = None
    if "text" in df.columns:
        text_col = "text"
    elif "sentence" in df.columns:
        text_col = "sentence"

    condition_col = None
    for candidate in ["condition", "icd_label", "disease", "label"]:
        if candidate in df.columns:
            condition_col = candidate
            break

    if text_col is None or condition_col is None:
        available = ", ".join(df.columns.astype(str).tolist())
        raise ValueError(
            "Could not infer required columns for dataset. "
            f"Need text/condition equivalents. Available columns: {available}"
        )

    normalized = pd.DataFrame(
        {
            "text": df[text_col],
            "condition": df[condition_col].map(_extract_condition),
        }
    )
    if "record_type" in df.columns:
        normalized["record_type"] = df["record_type"]
    else:
        normalized["record_type"] = "External Dataset"

    return normalized


def _write_dataset(df: pd.DataFrame, output_path: Path) -> None:
    suffix = output_path.suffix.lower()
    if suffix == ".csv":
        df.to_csv(output_path, index=False)
        return
    if suffix in {".parquet", ".pq"}:
        df.to_parquet(output_path, index=False)
        return
    raise ValueError("Output format must be .csv or .parquet")


def _load_label_map(path: str | None) -> dict[str, str]:
    if not path:
        return {}
    path_obj = Path(path)
    with path_obj.open("r", encoding="utf-8") as f:
        mapping = json.load(f)
    if not isinstance(mapping, dict):
        raise ValueError("Label map JSON must be an object: {\"source_label\": \"normalized_label\"}")
    return {str(k): str(v) for k, v in mapping.items()}


def _normalize_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    normalized = df.copy()
    normalized["text"] = (
        normalized["text"].astype(str).str.replace(r"\s+", " ", regex=True).str.strip()
    )
    normalized["condition"] = normalized["condition"].map(_extract_condition)
    normalized["record_type"] = normalized["record_type"].map(
        lambda value: None if pd.isna(value) else str(value).strip()
    )
    normalized["record_type"] = normalized["record_type"].fillna("External Dataset")

    normalized = normalized.replace({"": pd.NA})
    normalized = normalized.dropna(subset=["text", "condition", "record_type"])
    return normalized


def merge_datasets(
    inputs: list[str],
    output: str,
    label_map_path: str | None = None,
    dedupe_mode: str = "text_condition",
    min_text_length: int = 5,
    add_source_column: bool = False,
) -> None:
    label_map = _load_label_map(label_map_path)
    dataframes: list[pd.DataFrame] = []

    for input_path in inputs:
        path_obj = Path(input_path)
        if not path_obj.exists():
            raise FileNotFoundError(f"Input dataset not found: {path_obj}")

        raw_df = _read_dataset(path_obj)
        df = _coerce_to_required_schema(raw_df)

        trimmed = df.copy()
        if add_source_column:
            trimmed["source"] = path_obj.name
        dataframes.append(trimmed)
        print(f"Loaded {len(trimmed)} rows from {path_obj}")

    merged = pd.concat(dataframes, ignore_index=True)
    merged = _normalize_dataframe(merged)

    if label_map:
        merged["condition"] = merged["condition"].replace(label_map)

    if min_text_length > 0:
        merged = merged[merged["text"].str.len() >= min_text_length]

    before_dedup = len(merged)
    if dedupe_mode == "text":
        merged = merged.drop_duplicates(subset=["text"])
    elif dedupe_mode == "text_condition":
        merged = merged.drop_duplicates(subset=["text", "condition"])
    dedup_removed = before_dedup - len(merged)

    output_path = Path(output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    _write_dataset(merged, output_path)

    print(f"Rows after cleaning: {len(merged)}")
    print(f"Rows removed by deduplication: {dedup_removed}")
    print(f"Saved merged dataset to: {output_path}")
    print("Condition distribution:")
    print(merged["condition"].value_counts().to_string())


if __name__ == "__main__":
    args = parse_args()
    merge_datasets(
        inputs=args.inputs,
        output=args.output,
        label_map_path=args.label_map_path,
        dedupe_mode=args.dedupe_mode,
        min_text_length=args.min_text_length,
        add_source_column=args.add_source_column,
    )
