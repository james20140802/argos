"""Shared fixtures for web-layer tests.

ARG-134 introduces a shared TestClient that hits the real FastAPI app
factory from ARG-133. No DB connection is required because the factory
is lazy and only the static + template paths are exercised here.
"""
from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from argos.web.app import build_web_app


@pytest.fixture(scope="session")
def web_client() -> TestClient:
    return TestClient(build_web_app())
