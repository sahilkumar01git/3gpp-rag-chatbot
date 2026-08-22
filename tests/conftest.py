import sys
from pathlib import Path

# Make `app` importable when running `pytest` from the project root.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pytest

from app.config import get_settings


@pytest.fixture(autouse=True)
def _reset_settings_cache():
    """Settings are cached via lru_cache; clear between tests so
    monkeypatched env vars in one test don't leak into the next."""
    get_settings.cache_clear()
    yield
    get_settings.cache_clear()
