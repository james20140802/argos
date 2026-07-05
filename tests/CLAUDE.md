# tests/ — 테스트 작성·실행 규칙

- **릴리즈 CI에는 DB가 없다.** `.github/workflows/release.yml`이 테스트를 돌리는 유일한 CI이고 Postgres 서비스가 없다. DB 연결이 필요한 테스트는 반드시 self-skip 처리한다(연결 불가 시 `pytest.skip`). 안 그러면 다음 릴리즈 태그의 PyPI 배포가 그 테스트 때문에 막힌다.
- **ANSI 강제 환경 주의.** 일부 자동화 셸이 `FORCE_COLOR=3`을 강제해 CLI 출력 비교 테스트가 거짓 실패한다. 자동화 환경에서는 `env -u FORCE_COLOR uv run pytest ...`로 실행한다. 릴리즈 CI 동등성 확인이 필요하면 `POSTGRES_*`를 도달 불가 값으로 두고 돌린다.
- **출력은 quiet가 기본.** `uv run pytest tests/ -q --tb=short`. `-v`는 특정 실패를 파고들 때만 — 전체 `-v` 출력은 테스트 이름 수백 줄을 쏟아낸다.
- **새 테스트는 DB 없이 돌 수 있게** 설계한다(모델 단위 테스트, 목 세션). 통합 테스트가 꼭 필요하면 위 self-skip 규칙을 따른다.
