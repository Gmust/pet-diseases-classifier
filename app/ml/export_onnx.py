"""
Export the fine-tuned transformer to ONNX and apply int8 dynamic quantization.

Why
---
On Lambda, PyTorch is the dominant cost driver (forces ~3 GB RAM, big image,
15-30 s cold start). Serving the model with ONNX Runtime lets us drop torch
entirely from the *runtime* image → image <500 MB, cold start ~2-4 s, and ~1 GB
memory. int8 dynamic quantization shrinks the model ~4x with typically <1% F1 loss.

This is a BUILD-TIME script — run it where the trained weights live. It uses
optimum's exporter (torch, here, is fine) to write the ONNX graph, then quantizes
with onnxruntime's low-level API (no ORTModel session, so it avoids the
torch.int4 version clash between new onnxruntime and torch<2.6).

Install (build-time only — NOT a runtime dep):
    pip install "optimum[exporters]" "onnxruntime>=1.17"

Usage
-----
python -m app.ml.export_onnx \
    --model-dir models/transformer_model \
    --output-dir models/transformer_model_onnx
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description="Export model to quantized ONNX.")
    parser.add_argument(
        "--model-dir",
        default="models/transformer_model",
        help="HuggingFace-format directory produced by train.py.",
    )
    parser.add_argument(
        "--output-dir",
        default="models/transformer_model_onnx",
        help="Destination directory for the ONNX model + tokenizer.",
    )
    parser.add_argument(
        "--no-quantize",
        action="store_true",
        help="Export fp32 ONNX only (skip int8 dynamic quantization).",
    )
    parser.add_argument(
        "--keep-fp32",
        action="store_true",
        help="Keep the fp32 model.onnx after quantizing. By default it is removed "
        "so the deploy image only ships the int8 model (~260 MB smaller).",
    )
    parser.add_argument(
        "--opset",
        type=int,
        default=18,
        help="ONNX opset version (>=18 recommended for distilbert).",
    )
    args = parser.parse_args()

    src = Path(args.model_dir)
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    # --- Export to ONNX (writes model.onnx + tokenizer + config into `out`) ---
    try:
        from optimum.exporters.onnx import main_export
    except ImportError as exc:  # pragma: no cover
        raise SystemExit(
            'Missing build dep. Install: pip install "optimum[exporters]" onnxruntime'
        ) from exc

    print(f"Exporting {src} → ONNX (opset {args.opset}) …")
    main_export(
        model_name_or_path=str(src),
        output=str(out),
        task="text-classification",
        opset=args.opset,
    )

    # Ensure the label map travels with the model (main_export copies config.json,
    # but copy defensively in case of older optimum versions).
    if (src / "config.json").exists() and not (out / "config.json").exists():
        shutil.copy2(src / "config.json", out / "config.json")

    model_onnx = out / "model.onnx"
    if not model_onnx.exists():
        # Some optimum versions name the file differently — pick the first .onnx.
        onnx_files = list(out.glob("*.onnx"))
        if not onnx_files:
            raise SystemExit(f"Export produced no .onnx file in {out}")
        model_onnx = onnx_files[0]

    if args.no_quantize:
        print(f"Done (fp32 ONNX). Saved to: {out}  ({model_onnx.name})")
        return

    # --- int8 dynamic quantization (pure onnxruntime — no torch, no ORTModel) ---
    from onnxruntime.quantization import QuantType, quantize_dynamic

    quant_onnx = out / "model_quantized.onnx"
    print("Applying int8 dynamic quantization …")
    quantize_dynamic(str(model_onnx), str(quant_onnx), weight_type=QuantType.QInt8)

    # Drop the fp32 intermediate so the deploy image only ships the int8 model.
    if not args.keep_fp32 and model_onnx.name == "model.onnx" and model_onnx.exists():
        size_mb = model_onnx.stat().st_size / 1e6
        model_onnx.unlink()
        # ONNX external-data files (large models) sit next to the .onnx — clean those too.
        for extra in out.glob("model.onnx_data*"):
            extra.unlink()
        print(
            f"Removed fp32 intermediate model.onnx (~{size_mb:.0f} MB) — use --keep-fp32 to retain it."
        )

    print(f"Done (int8 ONNX). Saved to: {out}  ({quant_onnx.name})")
    print("Deploy with: MODEL_BACKEND=onnx MODEL_PATH=" + str(out))


if __name__ == "__main__":
    main()
