"""테스트 공통 fixture 정의."""

import uuid
from datetime import datetime, timezone

import pytest


@pytest.fixture
def sample_uuid():
    """테스트용 고정 UUID."""
    return uuid.UUID("12345678-1234-5678-1234-567812345678")


@pytest.fixture
def sample_datetime():
    """테스트용 고정 datetime."""
    return datetime(2026, 4, 3, 12, 0, 0, tzinfo=timezone.utc)
