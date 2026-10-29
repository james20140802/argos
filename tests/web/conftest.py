"""Shared fixtures for web-layer tests.

ARG-134 introduces a shared TestClient that hits the real FastAPI app
factory from ARG-133. No DB connection is required because the factory
is lazy and only the static + template paths are exercised here.

ARG-195: the app enforces login + CSRF by default. Route tests here are about
route behaviour, so an autouse fixture swaps the *default* policy to disabled
(the policy is resolved per request, so the session-scoped ``web_client`` is
covered too). Tests marked ``@pytest.mark.real_security`` get the real,
enforced policy pointed at a tmp token file instead.
"""
from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from argos.web import auth, security
from argos.web.app import build_web_app


@pytest.fixture(autouse=True)
def _web_security_policy(request, tmp_path, monkeypatch):
    if request.node.get_closest_marker("real_security"):
        token_path = tmp_path / "argos" / "web_token"
        token, _, _ = auth.ensure_token(token_path)
        policy = security.SecurityPolicy(
            enabled=True, token_path=token_path, failed_login_delay=0.0
        )
    else:
        token_path, token = None, None
        policy = security.SecurityPolicy(enabled=False)
    monkeypatch.setattr(security, "_default_policy_override", policy)
    # Never fall back to the operator's real ~/.config/argos/web_token.
    monkeypatch.setenv(auth.TOKEN_FILE_ENV, str(tmp_path / "unused-web-token"))
    # Same for the web-push VAPID key (ARG-321): never the operator's real one.
    monkeypatch.setenv("ARGOS_WEB_PUSH_KEY_FILE", str(tmp_path / "unused-vapid.pem"))
    return {"policy": policy, "token": token, "token_path": token_path}


@pytest.fixture(scope="session")
def web_client() -> TestClient:
    return TestClient(build_web_app())
