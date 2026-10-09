# ADR 0004: Text-based labels and per-channel int8 parity

- Status: Accepted
- Date: 2026-10-09

## Decision

Training and evaluation labels follow `docs/labeling-guide.md`: a row's category
is what its owner-language text supports, not a cause the text never mentions.
The primary holdout is `data/owner_eval_real.parquet`, real owner questions from
Big Red Bark Chat on which two independent Claude labeling passes agree.

The ONNX artifact is exported with per-channel int8 weights, and the Torch/ONNX
parity gate requires label agreement of at least 0.95 (was 0.99). The accuracy
delta (at most 0.01) and mean confidence delta (at most 0.03) bounds are unchanged.

## Context

Old labels often encoded a hidden cause (scaly skin labeled Immune, fast breathing
labeled Cardiovascular); the shipped model scored top-1 0.485 on real owner
questions. `bge-base-en-v1.5` trained on relabeled data scores 0.876 (Torch).

On 534 real questions every int8 variant of that model disagreed with fp32 on
3–5% of labels, mostly short texts near a class boundary, while accuracy stayed
within the delta bound (per-channel: 0.884 ONNX vs 0.876 Torch). fp32 ONNX
(419 MB) agrees fully but risks the 600 MB Lambda image budget.

## Consequences

What ships is judged by its own accuracy on the real-owner holdout plus the
unchanged safety gate, not by bit-for-bit agreement with Torch. Labels are only as
good as the guide and Claude's reading of it; the holdout shares that labeler, so
a veterinarian review of a holdout sample is the next trust step. The gate change
applies to all future releases; revisit it if a smaller or distilled model can
meet 0.99 again.
