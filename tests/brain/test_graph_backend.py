"""graph_backend — igraph/leidenalg 가용성 프로브 (ARG-277).

DB도 실제 라이브러리도 요구하지 않는다. 미설치 상황은 sys.modules에
None을 심어 ImportError를 강제하는 방식으로 재현한다.
"""
from __future__ import annotations

import builtins

import pytest

from argos.brain import graph_backend


@pytest.fixture(autouse=True)
def _clear_probe_cache():
    """프로브는 lru_cache라 테스트 간 결과가 새지 않게 매번 비운다."""
    graph_backend.load_graph_libs.cache_clear()
    yield
    graph_backend.load_graph_libs.cache_clear()


def _force_missing(monkeypatch, missing: str) -> None:
    """`missing` 모듈만 import를 실패시킨다."""
    real_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name == missing:
            raise ImportError(f"No module named {missing!r}")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)


def test_install_hint_names_the_extra_and_the_command():
    # 사람이 읽고 바로 실행할 수 있어야 한다 — extra 이름과 명령이 둘 다.
    assert "graph" in graph_backend.INSTALL_HINT
    assert "uv sync --all-extras" in graph_backend.INSTALL_HINT


def test_load_returns_none_when_igraph_is_missing(monkeypatch):
    _force_missing(monkeypatch, "igraph")
    assert graph_backend.load_graph_libs() is None


def test_load_returns_none_when_leidenalg_is_missing(monkeypatch):
    _force_missing(monkeypatch, "leidenalg")
    assert graph_backend.load_graph_libs() is None


def test_available_is_false_when_missing(monkeypatch):
    _force_missing(monkeypatch, "leidenalg")
    assert graph_backend.graph_libs_available() is False


def test_missing_library_logs_the_install_hint_instead_of_raising(monkeypatch, caplog):
    # AC: 스택트레이스가 아니라 "무엇을 설치하면 되는지" 알려 주는 메시지가 남는다.
    _force_missing(monkeypatch, "igraph")
    with caplog.at_level("WARNING", logger="argos.brain.graph_backend"):
        assert graph_backend.load_graph_libs() is None
    assert "uv sync --all-extras" in caplog.text


def test_require_raises_a_readable_error_when_missing(monkeypatch):
    _force_missing(monkeypatch, "igraph")
    with pytest.raises(graph_backend.GraphLibsUnavailable) as excinfo:
        graph_backend.require_graph_libs()
    assert "uv sync --all-extras" in str(excinfo.value)


def test_load_returns_both_modules_when_installed():
    igraph = pytest.importorskip("igraph")
    leidenalg = pytest.importorskip("leidenalg")
    loaded = graph_backend.load_graph_libs()
    assert loaded is not None
    assert loaded == (igraph, leidenalg)
    assert graph_backend.graph_libs_available() is True
