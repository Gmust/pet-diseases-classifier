import pytest

from scripts.check_image_size import check_size


def test_image_size_within_budget() -> None:
    check_size(99, 100, "test")


def test_image_size_exceeding_budget_fails() -> None:
    with pytest.raises(SystemExit, match="budget"):
        check_size(101, 100, "test")
