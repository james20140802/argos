"""Tests for the `argos web` CLI subcommand (ARG-133)."""
from __future__ import annotations

from unittest.mock import patch

import pytest

from argos.cli import main


def test_web_invokes_uvicorn_with_config_defaults():
    """`argos web` runs uvicorn against build_web_app() with config host/port."""
    with patch("argos.cli.uvicorn") as mock_uvicorn:
        rc = main(["web"])
    assert rc == 0
    mock_uvicorn.run.assert_called_once()
    args, kwargs = mock_uvicorn.run.call_args
    # First positional is the FastAPI app instance.
    from fastapi import FastAPI

    assert isinstance(args[0], FastAPI)
    assert kwargs["host"] == "127.0.0.1"
    assert kwargs["port"] == 8765


def test_web_cli_flags_override_config():
    """--host / --port flags override values from settings.user.web."""
    with patch("argos.cli.uvicorn") as mock_uvicorn:
        rc = main(["web", "--host", "0.0.0.0", "--port", "9000"])
    assert rc == 0
    _, kwargs = mock_uvicorn.run.call_args
    assert kwargs["host"] == "0.0.0.0"
    assert kwargs["port"] == 9000


def test_web_cli_rejects_invalid_port():
    """argparse rejects non-positive ports."""
    with patch("argos.cli.uvicorn"):
        with pytest.raises(SystemExit):
            main(["web", "--port", "0"])
