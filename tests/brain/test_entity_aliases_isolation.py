"""별칭 층은 opt-in이다 — 사건 묶기가 읽는 이름 경로가 별칭을 알지 못한다 (ARG-312).

DB 없이 도는 정적 회귀 테스트. 이름 저장·정규화·추출 모듈이 별칭 모듈을 가져오면
사건 묶기 입력이 별칭 상태에 따라 달라질 수 있으니, 그 연결이 생기는 순간 여기서 걸린다.
(별칭을 사건 묶기에 반영하려면 별도 이슈에서 이 테스트를 의도적으로 고쳐야 한다.)
행동 쪽 회귀(별칭 행 유무와 무관하게 이름 신호가 같다)는 ``test_entity_aliases_db.py``.
"""

from __future__ import annotations

import importlib
import inspect

import pytest

_NAME_PATH_MODULES = [
    "argos.brain.entity_store",
    "argos.brain.entity_names",
    "argos.brain.entity_extraction",
    "argos.brain.entity_frequency",
    "argos.brain.event_scoring",
    "argos.brain.event_candidates",
    "argos.brain.event_assignment",
    "argos.brain.recluster",
    "argos.brain.recluster_core",
    "argos.brain.recluster_input",
]


@pytest.mark.parametrize("module_name", _NAME_PATH_MODULES)
def test_event_grouping_name_path_does_not_reference_aliases(module_name):
    source = inspect.getsource(importlib.import_module(module_name))
    assert "entity_alias" not in source
    assert "EntityAlias" not in source
    assert "alias_store" not in source
