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

INSTALL_HINT = (
    "그래프 재군집에는 python-igraph + leidenalg가 필요하다 (optional extra "
    "'graph'). 설치: uv sync --all-extras"
)
"""미설치일 때 로그와 예외가 함께 쓰는 한 줄. `--extra graph`가 아니라
`--all-extras`인 이유는 entity_spacy와 같다 — 앞엣것은 고른 extra만 남기고
나머지(dev)를 지워서 pytest·ruff가 사라진다."""


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
