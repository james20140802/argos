from __future__ import annotations

import random
import uuid
from datetime import datetime, timezone
from unittest.mock import AsyncMock

import numpy as np
import pytest


@pytest.fixture
def tech_id():
    return uuid.UUID("aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa")


@pytest.fixture
def tech_id2():
    return uuid.UUID("bbbbbbbb-bbbb-bbbb-bbbb-bbbbbbbbbbbb")


@pytest.fixture
def mock_ack():
    return AsyncMock()


@pytest.fixture
def mock_respond():
    return AsyncMock()


@pytest.fixture
def now_utc():
    return datetime(2026, 5, 4, 10, 0, 0, tzinfo=timezone.utc)


@pytest.fixture(autouse=True)
def _seed_rng():
    random.seed(0)
    np.random.seed(0)
    yield
