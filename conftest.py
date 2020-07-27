"""General test configuration."""

# Core Library modules
import logging
from typing import Any, Dict


def pytest_configure(config: Dict[Any, Any]) -> None:
    """Flake8 is to verbose. Mute it."""
    logging.getLogger("flake8").setLevel(logging.WARN)
