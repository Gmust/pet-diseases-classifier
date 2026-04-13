"""
Fine-tunes a DistilBERT transformer on pet-condition text classification.

Replaces the TF-IDF + Logistic Regression baseline. Expected F1 improvement:
  Before: 50–70% (TF-IDF + LogReg, 23 overlapping classes)
  After:  80–90%+ (DistilBERT fine-tuned, 15 consolidated classes)

Usage examples
--------------
# Basic — local parquet with label consolidation:
python -m app.ml.train \
    --data-path data/merged_pet_dataset.parquet \
    --label-map data/label_map.json \
    --model-dir models/transformer_model

# Full config:
python -m app.ml.train \
    --data-path data/merged_pet_dataset.parquet \
    --label-map data/label_map.json \
    --model-dir models/transformer_model \
    --base-model distilbert-base-uncased \
    --epochs 6 \
    --batch-size 16 \
    --lr 2e-5 \
    --max-length 256 \
    --min-samples 20 \
    --warmup-ratio 0.1

# Hugging Face dataset:
python -m app.ml.train \
    --hf-dataset karenwky/pet-health-symptoms-dataset \
    --label-map data/label_map.json \
    --model-dir models/transformer_model
"""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForSequenceClassification, AutoTokenizer, get_linear_schedule_with_warmup


REQUIRED_COLUMNS = {"text", "condition", "record_type"}


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fine-tune a transformer for pet-condition text classification.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Data
    data_group = parser.add_mutually_exclusive_group()
    data_group.add_argument("--data-path", type=str, default="data/merged_pet_dataset.parquet",
                            help="Local .csv or .parquet dataset path.")
    data_group.add_argument("--hf-dataset", type=str, default=None,
                            help='Hugging Face dataset id, e.g. "karenwky/pet-health-symptoms-dataset".')
    parser.add_argument("--hf-split", type=str, default="train", help="Hugging Face dataset split name.")
    parser.add_argument("--label-map", type=str, default=None,
                        help="Path to label-consolidation JSON ({raw_label: canonical_label}).")
    parser.add_argument("--min-samples", type=int, default=20,
                        help="Drop classes with fewer than this many samples after label consolidation.")

    # Model
    parser.add_argument("--base-model", type=str, default="distilbert-base-uncased",
                        help="Pre-trained HuggingFace model name to fine-tune.")
    parser.add_argument("--model-dir", type=str, default="models/transformer_model",
                        help="Directory to save the fine-tuned model and tokenizer.")

    # Training
    parser.add_argument("--epochs", type=int, default=6, help="Number of training epochs.")
    parser.add_argument("--batch-size", type=int, default=16, help="Training batch size.")
    parser.add_argument("--lr", type=float, default=2e-5, help="Learning rate for AdamW.")
    parser.add_argument("--weight-decay", type=float, default=0.01, help="AdamW weight decay.")
    parser.add_argument("--max-length", type=int, default=256,
                        help="Max tokenizer sequence length (tokens). 256 covers ~99% of your data.")
    parser.add_argument("--warmup-ratio", type=float, default=0.1,
                        help="Fraction of total steps used for linear LR warm-up.")
    parser.add_argument("--test-size", type=float, default=0.15, help="Held-out test split fraction.")
    parser.add_argument("--val-size", type=float, default=0.1,
                        help="Validation split fraction (taken from training portion).")
    parser.add_argument("--patience", type=int, default=2,
                        help="Early-stopping patience (epochs without val-F1 improvement).")
    parser.add_argument("--random-state", type=int, default=42, help="Global random seed.")

    # Legacy compat — ignored silently so old CI scripts don't break
    parser.add_argument("--classifier-path", type=str, default=None, help=argparse.SUPPRESS)
    parser.add_argument("--vectorizer-path", type=str, default=None, help=argparse.SUPPRESS)
    parser.add_argument("--tune", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--calibrate", action="store_true", help=argparse.SUPPRESS)

    return parser.parse_args()


# ---------------------------------------------------------------------------
# Data loading + preprocessing
# ---------------------------------------------------------------------------

def _load_local_dataframe(data_path: str) -> pd.DataFrame:
    path_obj = Path(data_path)
    if not path_obj.exists():
        raise FileNotFoundError(f"Dataset not found: {path_obj}")
    suffix = path_obj.suffix.lower()
    if suffix == ".csv":
        return pd.read_csv(path_obj)
    if suffix in {".parquet", ".pq"}:
        return pd.read_parquet(path_obj)
    raise ValueError("Unsupported format — use .csv or .parquet, or pass --hf-dataset.")


def _load_label_map(path: str | None) -> dict[str, str]:
    if not path:
        return {}
    with open(path, encoding="utf-8") as f:
        raw = json.load(f)
    if not isinstance(raw, dict):
        raise ValueError("label-map JSON must be a flat {source: canonical} object.")
    # Strip comment keys that start with "//"
    return {k: str(v) for k, v in raw.items() if not k.startswith("//") and v}


def load_and_prepare(
    data_path: str,
    hf_dataset: str | None,
    hf_split: str,
    label_map: dict[str, str],
    min_samples: int,
) -> pd.DataFrame:
    if hf_dataset:
        try:
            from datasets import load_dataset as hf_load
        except ImportError as exc:
            raise ImportError("Install 'datasets': pip install datasets") from exc
        df = hf_load(hf_dataset, split=hf_split).to_pandas()
    else:
        df = _load_local_dataframe(data_path)

    missing = REQUIRED_COLUMNS.difference(df.columns)
    if missing:
        raise ValueError(f"Dataset missing required columns: {', '.join(sorted(missing))}")

    df = df.dropna(subset=["text", "condition"]).copy()
    df["text"] = df["text"].astype(str).str.replace(r"\s+", " ", regex=True).str.strip()
    df["condition"] = df["condition"].astype(str).str.strip()

    # Apply label consolidation map
    if label_map:
        df["condition"] = df["condition"].replace(label_map)
        print(f"Applied label map → {df['condition'].nunique()} unique classes")

    # Drop very rare classes (not enough signal to learn from)
    counts = df["condition"].value_counts()
    rare = counts[counts < min_samples].index.tolist()
    if rare:
        print(f"Dropping {len(rare)} class(es) with <{min_samples} samples: {rare}")
        df = df[~df["condition"].isin(rare)]

    if df.empty:
        raise ValueError("No usable rows remain after preprocessing.")

    print(f"\nFinal dataset: {len(df)} rows, {df['condition'].nunique()} classes")
    print("Class distribution:")
    print(df["condition"].value_counts().to_string())
    print()

    return df.reset_index(drop=True)


# ---------------------------------------------------------------------------
# PyTorch Dataset
# ---------------------------------------------------------------------------

class PetConditionDataset(Dataset):
    def __init__(self, texts: list[str], labels: list[int], tokenizer, max_length: int) -> None:
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self) -> int:
        return len(self.texts)

    def __getitem__(self, idx: int) -> dict:
        encoding = self.tokenizer(
            self.texts[idx],
            truncation=True,
            max_length=self.max_length,
            padding="max_length",
            return_tensors="pt",
        )
        return {
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "labels": torch.tensor(self.labels[idx], dtype=torch.long),
        }


# ---------------------------------------------------------------------------
# Training utilities
# ---------------------------------------------------------------------------

def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _compute_class_weights(y_train: list[int], num_classes: int, device: torch.device) -> torch.Tensor:
    classes = np.arange(num_classes)
    weights = compute_class_weight(class_weight="balanced", classes=classes, y=np.array(y_train))
    return torch.tensor(weights, dtype=torch.float32).to(device)


def _run_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: AdamW | None,
    scheduler,
    loss_fn: nn.CrossEntropyLoss,
    device: torch.device,
    train: bool,
) -> tuple[float, list[int], list[int]]:
    model.train() if train else model.eval()
    total_loss = 0.0
    all_preds: list[int] = []
    all_labels: list[int] = []

    ctx = torch.enable_grad() if train else torch.no_grad()
    with ctx:
        for batch in loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            loss = loss_fn(outputs.logits, labels)

            if train:
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                scheduler.step()

            total_loss += loss.item()
            preds = outputs.logits.argmax(dim=-1).cpu().tolist()
            all_preds.extend(preds)
            all_labels.extend(labels.cpu().tolist())

    avg_loss = total_loss / len(loader)
    return avg_loss, all_preds, all_labels


# ---------------------------------------------------------------------------
# Main training function
# ---------------------------------------------------------------------------

def train_and_save(
    data_path: str,
    hf_dataset: str | None,
    hf_split: str,
    label_map_path: str | None,
    model_dir: str,
    base_model: str = "distilbert-base-uncased",
    epochs: int = 6,
    batch_size: int = 16,
    lr: float = 2e-5,
    weight_decay: float = 0.01,
    max_length: int = 256,
    warmup_ratio: float = 0.1,
    test_size: float = 0.15,
    val_size: float = 0.1,
    patience: int = 2,
    min_samples: int = 20,
    random_state: int = 42,
    # Legacy params — ignored
    classifier_path: str | None = None,
    vectorizer_path: str | None = None,
    tune: bool = False,
    calibrate: bool = False,
) -> None:
    _seed_everything(random_state)
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    print(f"Using device: {device}")

    # --- Load & prepare data ---
    label_map = _load_label_map(label_map_path)
    df = load_and_prepare(data_path, hf_dataset, hf_split, label_map, min_samples)

    # --- Encode labels ---
    unique_labels = sorted(df["condition"].unique())
    label2id = {label: idx for idx, label in enumerate(unique_labels)}
    id2label = {idx: label for label, idx in label2id.items()}
    df["label_id"] = df["condition"].map(label2id)

    # --- Splits: train / val / test ---
    texts = df["text"].tolist()
    label_ids = df["label_id"].tolist()

    x_train_val, x_test, y_train_val, y_test = train_test_split(
        texts, label_ids,
        test_size=test_size,
        stratify=label_ids,
        random_state=random_state,
    )
    # val_size is relative to the remaining train+val pool
    relative_val = val_size / (1.0 - test_size)
    x_train, x_val, y_train, y_val = train_test_split(
        x_train_val, y_train_val,
        test_size=relative_val,
        stratify=y_train_val,
        random_state=random_state,
    )
    print(f"Split → train: {len(x_train)}, val: {len(x_val)}, test: {len(x_test)}")

    # --- Tokenizer & Datasets ---
    print(f"\nLoading tokenizer: {base_model}")
    tokenizer = AutoTokenizer.from_pretrained(base_model)

    train_dataset = PetConditionDataset(x_train, y_train, tokenizer, max_length)
    val_dataset = PetConditionDataset(x_val, y_val, tokenizer, max_length)
    test_dataset = PetConditionDataset(x_test, y_test, tokenizer, max_length)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=batch_size * 2, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=batch_size * 2, shuffle=False, num_workers=0)

    # --- Model ---
    num_classes = len(unique_labels)
    print(f"Loading base model: {base_model}  ({num_classes} output classes)")
    model = AutoModelForSequenceClassification.from_pretrained(
        base_model,
        num_labels=num_classes,
        id2label=id2label,
        label2id=label2id,
    )
    model.to(device)

    # --- Optimizer, scheduler, loss ---
    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    total_steps = len(train_loader) * epochs
    warmup_steps = int(total_steps * warmup_ratio)
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps,
    )
    class_weights = _compute_class_weights(y_train, num_classes, device)
    loss_fn = nn.CrossEntropyLoss(weight=class_weights)

    # --- Training loop with early stopping ---
    from sklearn.metrics import f1_score

    best_val_f1 = -1.0
    epochs_without_improvement = 0
    best_model_state: dict | None = None

    for epoch in range(1, epochs + 1):
        train_loss, train_preds, train_labels = _run_epoch(
            model, train_loader, optimizer, scheduler, loss_fn, device, train=True
        )
        val_loss, val_preds, val_labels = _run_epoch(
            model, val_loader, None, None, loss_fn, device, train=False
        )

        train_f1 = f1_score(train_labels, train_preds, average="macro", zero_division=0)
        val_f1 = f1_score(val_labels, val_preds, average="macro", zero_division=0)

        print(
            f"Epoch {epoch}/{epochs} | "
            f"train_loss={train_loss:.4f} train_f1={train_f1:.4f} | "
            f"val_loss={val_loss:.4f} val_f1={val_f1:.4f}"
        )

        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            epochs_without_improvement = 0
            # Deep copy best weights
            best_model_state = {k: v.clone() for k, v in model.state_dict().items()}
            print(f"  ✓ New best val F1: {best_val_f1:.4f} — saving checkpoint")
        else:
            epochs_without_improvement += 1
            print(f"  No improvement ({epochs_without_improvement}/{patience})")
            if epochs_without_improvement >= patience:
                print(f"Early stopping triggered after epoch {epoch}.")
                break

    # --- Restore best weights ---
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
        print(f"\nRestored best checkpoint (val F1={best_val_f1:.4f})")

    # --- Final evaluation on held-out test set ---
    _, test_preds, test_labels = _run_epoch(
        model, test_loader, None, None, loss_fn, device, train=False
    )
    label_names = [id2label[i] for i in range(num_classes)]
    print("\n=== Test-set Classification Report ===")
    print(classification_report(test_labels, test_preds, target_names=label_names, zero_division=0))

    # --- Save model + tokenizer ---
    output_dir = Path(model_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    print(f"\nSaved fine-tuned model to: {output_dir}")
    print("Done.")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    args = parse_args()
    train_and_save(
        data_path=args.data_path,
        hf_dataset=args.hf_dataset,
        hf_split=args.hf_split,
        label_map_path=args.label_map,
        model_dir=args.model_dir,
        base_model=args.base_model,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        weight_decay=args.weight_decay,
        max_length=args.max_length,
        warmup_ratio=args.warmup_ratio,
        test_size=args.test_size,
        val_size=args.val_size,
        patience=args.patience,
        min_samples=args.min_samples,
        random_state=args.random_state,
        classifier_path=args.classifier_path,
        vectorizer_path=args.vectorizer_path,
        tune=args.tune,
        calibrate=args.calibrate,
    )
