"""Tests for scripts/benchmark_genealogist_quantized.py

ARG-125: Benchmark qwen3:32b-q4_K_M genealogy quality and VRAM headroom

These tests verify:
- The harness wires CLI arguments correctly
- The report shape is correct (required keys present)
- The empty-DB case is handled gracefully (no crash, report still written)
"""
from __future__ import annotations

import json
import sys
import uuid
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest
import respx
import httpx

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_fake_row(*, item_id: str | None = None, title: str = "Test Tech",
                   raw_content: str = "some content about the technology") -> MagicMock:
    row = MagicMock()
    row.id = item_id or str(uuid.uuid4())
    row.title = title
    row.raw_content = raw_content
    return row


def _make_fake_session(rows: list[MagicMock]) -> AsyncMock:
    """Build a session mock that returns the given rows from execute()."""
    session = AsyncMock()
    result_mock = MagicMock()
    result_mock.fetchall.return_value = rows
    session.execute = AsyncMock(return_value=result_mock)
    return session


# ---------------------------------------------------------------------------
# Test: argument wiring / config
# ---------------------------------------------------------------------------

def test_benchmark_args_defaults():
    """BenchmarkArgs defaults match documented values (model, num_ctx, items, out)."""
    # Import from the script (not installed as a package — add scripts/ to sys.path)
    scripts_dir = Path(__file__).parents[2] / "scripts"
    if str(scripts_dir) not in sys.path:
        sys.path.insert(0, str(scripts_dir))

    from benchmark_genealogist_quantized import BenchmarkArgs  # type: ignore[import]

    args = BenchmarkArgs()
    assert args.model == "qwen3:32b"
    assert args.num_ctx >= 3072
    assert args.items >= 10
    assert args.out is not None


def test_benchmark_args_overrides():
    """CLI-level overrides propagate into BenchmarkArgs."""
    scripts_dir = Path(__file__).parents[2] / "scripts"
    if str(scripts_dir) not in sys.path:
        sys.path.insert(0, str(scripts_dir))

    from benchmark_genealogist_quantized import BenchmarkArgs  # type: ignore[import]

    args = BenchmarkArgs(
        model="qwen3:32b-q4_K_M",
        num_ctx=6144,
        items=5,
        out=Path("/tmp/out.json"),
    )
    assert args.model == "qwen3:32b-q4_K_M"
    assert args.num_ctx == 6144
    assert args.items == 5
    assert args.out == Path("/tmp/out.json")


# ---------------------------------------------------------------------------
# Test: prompt reuse — must import from genealogist, not copy
# ---------------------------------------------------------------------------

def test_benchmark_uses_genealogist_prompt():
    """The script must import _GENEALOGIST_PROMPT from genealogist, not duplicate it."""
    scripts_dir = Path(__file__).parents[2] / "scripts"
    if str(scripts_dir) not in sys.path:
        sys.path.insert(0, str(scripts_dir))

    # The module-level attribute must point to the same object as the source
    import benchmark_genealogist_quantized as harness  # type: ignore[import]
    from argos.brain.nodes.genealogist import _GENEALOGIST_PROMPT

    assert harness.GENEALOGIST_PROMPT is _GENEALOGIST_PROMPT


# ---------------------------------------------------------------------------
# Test: report shape
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_run_benchmark_returns_correct_report_shape(tmp_path, monkeypatch):
    """run_benchmark returns a dict with required top-level keys and per-item records."""
    scripts_dir = Path(__file__).parents[2] / "scripts"
    if str(scripts_dir) not in sys.path:
        sys.path.insert(0, str(scripts_dir))

    import benchmark_genealogist_quantized as harness  # type: ignore[import]

    fake_rows = [
        _make_fake_row(title="Tech A", raw_content="content A"),
        _make_fake_row(title="Tech B", raw_content="content B"),
    ]
    session = _make_fake_session(fake_rows)

    ollama_response = '{"replace_target_id": null, "relation_type": null, "reason": "no relation"}'

    from argos.brain.ollama_client import OLLAMA_BASE_URL

    with respx.mock:
        respx.post(f"{OLLAMA_BASE_URL}/api/generate").mock(
            return_value=httpx.Response(200, json={"response": ollama_response})
        )
        report = await harness.run_benchmark(
            session=session,
            model="qwen3:32b",
            num_ctx=3072,
            items=2,
        )

    # Top-level keys
    assert "model" in report
    assert "num_ctx" in report
    assert "items_evaluated" in report
    assert "results" in report
    assert report["model"] == "qwen3:32b"
    assert report["num_ctx"] == 3072


@pytest.mark.asyncio
async def test_run_benchmark_each_result_has_required_fields(tmp_path, monkeypatch):
    """Each result record must have item_id, title, relation_type, reason, elapsed_s."""
    scripts_dir = Path(__file__).parents[2] / "scripts"
    if str(scripts_dir) not in sys.path:
        sys.path.insert(0, str(scripts_dir))

    import benchmark_genealogist_quantized as harness  # type: ignore[import]

    fake_rows = [_make_fake_row(title="Tech X", raw_content="some content")]
    session = _make_fake_session(fake_rows)

    ollama_response = '{"replace_target_id": null, "relation_type": null, "reason": "no prior art"}'

    from argos.brain.ollama_client import OLLAMA_BASE_URL

    with respx.mock:
        respx.post(f"{OLLAMA_BASE_URL}/api/generate").mock(
            return_value=httpx.Response(200, json={"response": ollama_response})
        )
        report = await harness.run_benchmark(
            session=session,
            model="qwen3:32b",
            num_ctx=3072,
            items=1,
        )

    assert len(report["results"]) == 1
    rec = report["results"][0]
    assert "item_id" in rec
    assert "title" in rec
    assert "relation_type" in rec
    assert "reason" in rec
    assert "elapsed_s" in rec
    assert isinstance(rec["elapsed_s"], float)


# ---------------------------------------------------------------------------
# Test: empty DB handled gracefully
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_run_benchmark_empty_db_returns_zero_results():
    """When the DB returns no rows, the report has items_evaluated=0 and empty results."""
    scripts_dir = Path(__file__).parents[2] / "scripts"
    if str(scripts_dir) not in sys.path:
        sys.path.insert(0, str(scripts_dir))

    import benchmark_genealogist_quantized as harness  # type: ignore[import]

    session = _make_fake_session([])  # No rows

    report = await harness.run_benchmark(
        session=session,
        model="qwen3:32b",
        num_ctx=3072,
        items=10,
    )

    assert report["items_evaluated"] == 0
    assert report["results"] == []
    # No crash, report is valid
    assert "model" in report


# ---------------------------------------------------------------------------
# Test: write_report creates JSON file at the specified path
# ---------------------------------------------------------------------------

def test_write_report_creates_json_file(tmp_path):
    """write_report serialises the report dict to JSON at the given path."""
    scripts_dir = Path(__file__).parents[2] / "scripts"
    if str(scripts_dir) not in sys.path:
        sys.path.insert(0, str(scripts_dir))

    import benchmark_genealogist_quantized as harness  # type: ignore[import]

    out_path = tmp_path / "report.json"
    report: dict[str, Any] = {
        "model": "qwen3:32b",
        "num_ctx": 3072,
        "items_evaluated": 1,
        "results": [
            {
                "item_id": "abc",
                "title": "Test Tech",
                "relation_type": None,
                "reason": "no relation",
                "elapsed_s": 1.23,
            }
        ],
    }
    harness.write_report(report, out_path)

    assert out_path.exists()
    loaded = json.loads(out_path.read_text())
    assert loaded["model"] == "qwen3:32b"
    assert len(loaded["results"]) == 1


# ---------------------------------------------------------------------------
# Test: Ollama error is recorded per-item rather than aborting the whole run
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_run_benchmark_records_ollama_error_per_item():
    """When Ollama returns an HTTP error for one item, the error is recorded in the result,
    not raised, so the rest of the items still run."""
    scripts_dir = Path(__file__).parents[2] / "scripts"
    if str(scripts_dir) not in sys.path:
        sys.path.insert(0, str(scripts_dir))

    import benchmark_genealogist_quantized as harness  # type: ignore[import]

    fake_rows = [
        _make_fake_row(title="Tech A"),
        _make_fake_row(title="Tech B"),
    ]
    session = _make_fake_session(fake_rows)

    from argos.brain.ollama_client import OLLAMA_BASE_URL

    call_count = {"n": 0}

    def _side_effect(request):
        call_count["n"] += 1
        if call_count["n"] == 1:
            return httpx.Response(500, json={"error": "gpu oom"})
        return httpx.Response(200, json={"response": '{"replace_target_id": null, "relation_type": null, "reason": "ok"}'})

    with respx.mock:
        respx.post(f"{OLLAMA_BASE_URL}/api/generate").mock(side_effect=_side_effect)
        report = await harness.run_benchmark(
            session=session,
            model="qwen3:32b",
            num_ctx=3072,
            items=2,
        )

    # Both items are represented in results
    assert len(report["results"]) == 2
    # The error item has an error key
    assert "error" in report["results"][0]
    # The successful item has the normal fields
    assert report["results"][1]["relation_type"] is None
