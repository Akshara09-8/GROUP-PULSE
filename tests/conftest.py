from pathlib import Path

import pytest

FIXTURES = Path(__file__).parent / "fixtures"


@pytest.fixture
def ann_root() -> Path:
    return FIXTURES / "ami_mini"
