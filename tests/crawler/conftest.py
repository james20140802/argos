from __future__ import annotations

from unittest.mock import AsyncMock

import pytest


@pytest.fixture(autouse=True)
def allow_robots_by_default(monkeypatch):
    """Patch is_robots_allowed to return True for all tests by default.

    Tests that want to verify robots-disallow behaviour should override this
    via their own monkeypatch call after this fixture has run.
    """
    monkeypatch.setattr(
        "argos.crawler.static_fetcher.is_robots_allowed",
        AsyncMock(return_value=True),
    )
