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


def test_install_hint_covers_the_pipx_install_path():
    # README가 `pipx install argos-scout`를 정식 설치 경로로 안내한다. 그 경로로
    # 깐 운영자에게 `uv sync`만 주면 실행할 수 없는 한 줄을 주는 셈이다 — pipx의
    # 격리 환경은 건드리지 못하고, 소스 체크아웃 밖에서는 명령 자체가 실패한다.
    assert "pipx inject argos-scout python-igraph leidenalg" in graph_backend.INSTALL_HINT


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


def test_install_commands_is_the_single_source_for_both_paths():
    # 안내 문구가 런타임 예외와 CLI --help 두 군데에 있다. 문자열을 따로 들고
    # 있으면 한쪽만 고치고 끝나서, 실제로 6daa274가 예외 경로만 고치고 --help는
    # `uv sync` 하나만 남긴 채로 지나갔다. 두 자리가 같은 상수를 보게 한다.
    assert "uv sync --all-extras" in graph_backend.INSTALL_COMMANDS
    assert "pipx inject argos-scout python-igraph leidenalg" in graph_backend.INSTALL_COMMANDS
    assert graph_backend.INSTALL_COMMANDS in graph_backend.INSTALL_HINT
