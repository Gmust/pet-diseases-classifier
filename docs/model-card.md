# Release Model Card

Generated from `configs/release-documentation.json`; do not edit by hand.

## Intended and prohibited use

Cautious condition-category pre-assessment used only with deterministic triage, abstention, and advice-safety policy.

Prohibited: diagnosis, prescribing, autonomous treatment, emergency de-escalation, or replacing veterinary care.

## Training and provenance

- Active backend: quantized ONNX on Lambda; Torch is the reference backend
- Training configuration: `configs/train.json`
- Dataset/split evidence: `dataset build manifest plus immutable split manifest`
- Rollback version: Declared in each release-manifest.json; never inferred from latest

## Evaluation and safety evidence

Release evaluation reports explicit labels, per-class/segment metrics, confusion matrix, coverage, and baseline deltas. Current local baseline is documented in docs/baseline-2026-07-07.md.

Calibration/abstention: Versioned thresholds and selective-accuracy evidence are required in the promoted bundle.

Safety: data/safety_eval.json must pass; Torch/ONNX parity must meet configs/release-gates.json.

## Limitations and governance

Confidence is not clinical probability. Performance may degrade on unseen phrasing, species, conditions, languages, or distribution shift. The deterministic safety layer remains authoritative.

Clinical status: Not clinically validated; veterinary approval metadata is required for advice content.
