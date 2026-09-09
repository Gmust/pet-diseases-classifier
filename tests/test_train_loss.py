"""Numeric tests for the focal-loss / class-weight behavior in train.py.
Skipped without torch (needs the training env, requirements-train.txt)."""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from ml_pipeline.train import FocalLoss  # noqa: E402


def test_focal_loss_matches_weighted_ce_when_gamma_is_zero():
    # gamma=0 => focal_term is always 1, so focal loss must reduce to plain
    # class-weighted cross-entropy exactly.
    logits = torch.tensor([[2.0, 0.5, -1.0], [0.1, 1.5, 0.2]])
    targets = torch.tensor([0, 1])
    alpha = torch.tensor([1.0, 2.0, 0.5])

    focal = FocalLoss(alpha=alpha, gamma=0.0)(logits, targets)
    # Match FocalLoss's own reduction (plain mean of per-sample weighted CE),
    # not F.cross_entropy's default reduction="mean" (which normalizes by the
    # sum of the sample weights, not by batch size) — a different convention.
    weighted_ce = torch.nn.functional.cross_entropy(
        logits, targets, weight=alpha, reduction="none"
    ).mean()
    assert torch.isclose(focal, weighted_ce, atol=1e-6)


def test_focal_loss_pt_uses_unweighted_probability_not_weighted_ce():
    # Regression test for the bug: pt = exp(-weighted_ce) computes p_true**alpha
    # instead of p_true. With a single sample we can compute the correct
    # value by hand and confirm the loss no longer takes that distorted path.
    logits = torch.tensor([[3.0, 0.0]])
    targets = torch.tensor([0])
    alpha = torch.tensor([5.0, 1.0])  # large weight on the true class
    gamma = 2.0

    true_pt = torch.softmax(logits, dim=-1)[0, 0]
    ce_unweighted = torch.nn.functional.cross_entropy(logits, targets, reduction="none")[0]
    expected = alpha[0] * ((1 - true_pt) ** gamma) * ce_unweighted

    actual = FocalLoss(alpha=alpha, gamma=gamma)(logits, targets)
    assert torch.isclose(actual, expected, atol=1e-6)

    # The buggy formula (pt from weighted CE) would give a different value —
    # assert we do NOT match it, so a regression to the old code is caught.
    ce_weighted = torch.nn.functional.cross_entropy(
        logits, targets, weight=alpha, reduction="none"
    )[0]
    buggy_pt = torch.exp(-ce_weighted)
    buggy = ((1 - buggy_pt) ** gamma) * ce_weighted
    assert not torch.isclose(actual, buggy, atol=1e-6)


def test_focal_loss_without_alpha_reduces_high_confidence_loss_more_than_low():
    # Core focal-loss property: an easy (high-confidence-correct) example is
    # down-weighted more than a hard one, for the same target class.
    easy_logits = torch.tensor([[5.0, 0.0]])  # confidently correct
    hard_logits = torch.tensor([[0.5, 0.0]])  # barely correct
    targets = torch.tensor([0])

    loss_fn = FocalLoss(alpha=None, gamma=2.0)
    easy_loss = loss_fn(easy_logits, targets)
    hard_loss = loss_fn(hard_logits, targets)
    assert easy_loss < hard_loss
