"""그래프 재군집용 optional extra 프로브 — ARG-277.

야간 재군집(2단계)은 Leiden 커뮤니티 탐지를 쓰고, 그건 C 확장인
`python-igraph` + `leidenalg`를 요구한다. 이 둘은 **선택 설치**다:
크롤·배정·브리핑·피드 같은 평소 경로는 이 라이브러리 없이 그대로 돌아야
한다 — 재군집은 품질 개선이지 필수 경로가 아니기 때문이다.

`entity_spacy.py` 선례를 그대로 따른다: ImportError를 삼키고, 경고를 한 번
남기고, None을 돌려준다. 예외를 올리면 미설치 환경에서 평소 동작이 죽는다.

**두 진입점을 나눠 둔 이유:** `load_graph_libs()`는 "없으면 없는 대로"를
표현하고(피드·파이프라인처럼 재군집이 곁다리인 자리), `require_graph_libs()`는
"이건 재군집 전용 코드라 없으면 할 말이 없다"를 표현한다(순수 코어). 후자도
스택트레이스 대신 설치 방법을 담은 메시지를 던진다 — 운영자가 읽는 것은
traceback이 아니라 마지막 한 줄이다.
"""

from __future__ import annotations

import logging
from functools import lru_cache
from types import ModuleType

logger = logging.getLogger(__name__)

INSTALL_COMMANDS = (
    "uv sync --all-extras (source checkout) / "
    "pipx inject argos-scout python-igraph leidenalg (pipx install)"
)
"""설치 명령 두 갈래. **런타임 예외와 CLI `--help`가 같은 이 상수를 본다.**

따로 들고 있으면 한쪽만 고치고 끝난다 — 실제로 그랬다: 예외 경로에 pipx를
더한 커밋이 `--help` 설명은 `uv sync` 하나만 남긴 채로 지나갔다."""

INSTALL_HINT = (
    "그래프 재군집에는 python-igraph + leidenalg가 필요하다 (optional extra "
    f"'graph'). 설치: {INSTALL_COMMANDS}"
)
"""미설치일 때 로그와 예외가 함께 쓰는 한 줄.

**설치 경로를 둘 다 적는 이유:** README가 `pipx install argos-scout`를 정식
설치 경로로 안내한다. 그렇게 깐 운영자에게 `uv sync`만 주면 실행할 수 없는
한 줄을 주는 셈이다 — 소스 체크아웃 밖에서는 명령 자체가 실패하고, 체크아웃
안에서 돌려도 고치는 건 그 프로젝트의 `.venv`지 pipx의 격리 환경이 아니다.
이 문자열은 `argos recluster-events`가 `ERROR:` 뒤에 그대로 찍는 마지막 한
줄이라, 여기서 못 고치면 운영자는 고칠 방법을 끝내 못 본다.

`--extra graph`가 아니라 `--all-extras`인 이유는 entity_spacy와 같다 —
앞엣것은 고른 extra만 남기고 나머지(dev)를 지워서 pytest·ruff가 사라진다.
pipx 쪽이 `inject`인 것도 같은 결이다: `pipx install --force
"argos-scout[graph]"`는 멀쩡히 돌던 설치를 통째로 갈아엎는다."""


class GraphLibsUnavailable(RuntimeError):
    """재군집 전용 코드가 라이브러리 없이 불렸을 때."""

    def __init__(self, message: str = INSTALL_HINT) -> None:
        super().__init__(message)


@lru_cache(maxsize=1)
def load_graph_libs() -> tuple[ModuleType, ModuleType] | None:
    """`(igraph, leidenalg)` 또는 None. 예외를 올리지 않는다.

    캐시가 1칸인 건 인자가 없어서다 — 프로세스당 한 번만 실제 import를 시도한다.
    """
    try:
        import igraph
        import leidenalg
    except ImportError:
        logger.warning("%s", INSTALL_HINT)
        return None
    return igraph, leidenalg


def graph_libs_available() -> bool:
    """라이브러리가 둘 다 있으면 True."""
    return load_graph_libs() is not None


def require_graph_libs() -> tuple[ModuleType, ModuleType]:
    """`(igraph, leidenalg)`. 없으면 `GraphLibsUnavailable`."""
    loaded = load_graph_libs()
    if loaded is None:
        raise GraphLibsUnavailable()
    return loaded
