import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent))


def pytest_configure(config: pytest.Config) -> None:
    config.addinivalue_line(
        "markers",
        "slow: Integrations-Tests mit echtem Training (per Default aktiv; "
        "mit '-m not slow' deaktivierbar).",
    )
