from pathlib import Path

import pytest


@pytest.mark.parametrize("requirements", ["requirements.txt", "requirements-onnx.txt"])
def test_runtime_images_install_settings_dependency(requirements: str) -> None:
    lines = Path(requirements).read_text(encoding="utf-8").splitlines()
    assert any(line.startswith("pydantic-settings==") for line in lines)
