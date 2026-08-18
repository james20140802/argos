"""마이그레이션 테스트가 진짜 DB를 드롭하지 않는지 지키는 가드 — Postgres 없이 돈다.

``tests/test_migration_event_layer.py``는 자기만의 throwaway DB
(``argos_migration_test``)에 대고 ``DROP DATABASE ... WITH (FORCE)``를 무조건
실행한다. 그 이름이 개발자의 진짜 dev DB나 pytest 스크래치 DB와 겹치면 해당
DB가 통째로 날아간다.

가드 본체는 그 모듈에 있지만(파괴적 작업 바로 앞에 두는 게 맞다), 검증은
여기서 한다. 그 모듈은 Postgres가 없으면 통째로 skip되므로 — Postgres가 없는
release CI에서도 이 가드만은 계속 돌아야 한다.
"""
from __future__ import annotations

import pytest

from tests.conftest import DEV_DB_NAME, TEST_DB_NAME
from tests.test_migration_event_layer import (
    _MIGRATION_DB_NAME,
    _assert_migration_db_is_disposable,
)


def test_rejects_the_developers_dev_database():
    """POSTGRES_DB가 하필 마이그레이션 DB 이름인 개발자 — conftest 검사는 통과한다."""
    with pytest.raises(RuntimeError, match="collides with the dev"):
        _assert_migration_db_is_disposable(
            "argos_migration_test", "argos_migration_test", "argos_test"
        )


def test_rejects_the_pytest_scratch_database():
    with pytest.raises(RuntimeError, match="collides with the pytest"):
        _assert_migration_db_is_disposable("argos_test", "argos", "argos_test")


def test_allows_a_name_nobody_else_uses():
    _assert_migration_db_is_disposable(
        "argos_migration_test", "argos", "argos_test"
    )


def test_the_name_this_module_actually_uses_is_disposable_here():
    """이 개발 환경에서 실제로 쓰이는 세 이름이 서로 겹치지 않는다."""
    _assert_migration_db_is_disposable(
        _MIGRATION_DB_NAME, DEV_DB_NAME, TEST_DB_NAME
    )
