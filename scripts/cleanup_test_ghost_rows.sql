-- ARG-191: dev DB(`argos`) 유령 행 정리 스크립트.
--
-- 배경: ARG-155 (`tests/web/services/test_feed_service.py`)를 비롯한 과거
-- DB-backed 테스트가 격리된 테스트 DB가 아니라 개발자의 실제 dev DB에
-- 대고 시딩했고, assert 실패 시 delete 정리가 실행되지 않아(assert 뒤에만
-- 있고 finally가 아니었음) 행이 누적되었다. ARG-191에서 conftest.py가
-- 전용 `argos_test` DB로 테스트를 격리하도록 고쳤고(더 이상 새로 쌓이지
-- 않음), 이 스크립트는 그 이전에 이미 dev DB에 쌓인 유령 행만 정리한다.
--
-- 실행 금지: 이 스크립트는 커밋용 산출물이며, Claude가 실행하지 않았다.
-- 사용자가 아래 SELECT로 먼저 재확인한 뒤 본인 판단으로 직접 실행할 것.
--
-- 사용법:
--   docker exec -i argos-db psql -U argos -d argos < scripts/cleanup_test_ghost_rows.sql
-- 또는 psql 접속 후 각 문을 검토하며 단계별로 실행.
--
-- 접속 정보는 ~/.config/argos/.env 참조 (POSTGRES_USER/DB/HOST/PORT).

-- ---------------------------------------------------------------------
-- 1) 집계만 (읽기 전용) — 실행해도 안전. 삭제 전 반드시 먼저 확인할 것.
-- ---------------------------------------------------------------------
SELECT
    count(*) FILTER (WHERE source_url LIKE 'https://example.com/arg155/%') AS arg155_ghost_rows,
    count(*)                                                              AS total_tech_items
FROM tech_items;

-- 2026-07-04 기준 확인된 수치 (ARG-191 조사 시점, 참고용 — 실행 시 값은 달라질 수 있음):
--   arg155_ghost_rows = 385  (source_url LIKE 'https://example.com/arg155/%')
--   total_tech_items  = 1528
--
-- 이 385건은 모두 ARG-155 테스트(`test_fetch_feed_orders_newest_first_and_paginates_with_cursor`,
-- `test_fetch_feed_filters_by_category_and_joins_status`)가 assert 실패 시 정리(delete)를
-- 건너뛰어 dev DB에 남긴 `feed-test-*` / `arg155-mainstream` / `arg155-alpha` 타이틀의
-- 테스트 전용 tech_items다. 이 중 5건은 ARG-191 작업 중 "수정 전 버그를 재현"하려고
-- 고쳐지지 않은 코드로 전체 테스트 스위트를 1회 실행한 데서 추가된 것이며(원래 관측치는
-- 380건), 이후로는 conftest.py가 전용 argos_test DB로 격리하므로 더 이상 dev DB에 쌓이지 않는다.

-- ---------------------------------------------------------------------
-- 2) 참조 확인 (읽기 전용) — 삭제 전 연결된 user_assets / track_history 행도 함께
--    지워질지 확인. tech_item.py FK는 전부 ON DELETE CASCADE이므로 아래 3)의
--    tech_items DELETE 한 번으로 연쇄 삭제된다.
-- ---------------------------------------------------------------------
SELECT count(*) AS orphaned_user_assets
FROM user_assets ua
JOIN tech_items ti ON ti.id = ua.tech_id
WHERE ti.source_url LIKE 'https://example.com/arg155/%';

SELECT count(*) AS orphaned_track_history
FROM track_history th
JOIN user_assets ua ON ua.id = th.user_asset_id
JOIN tech_items ti ON ti.id = ua.tech_id
WHERE ti.source_url LIKE 'https://example.com/arg155/%';

-- ---------------------------------------------------------------------
-- 3) 실제 삭제 (쓰기) — 검토 후 사용자가 직접 실행. CASCADE로 user_assets /
--    track_history / tech_succession의 관련 행도 함께 제거된다.
-- ---------------------------------------------------------------------
-- DELETE FROM tech_items
-- WHERE source_url LIKE 'https://example.com/arg155/%';

-- 삭제 후 재확인 (기대값: 0):
-- SELECT count(*) FROM tech_items WHERE source_url LIKE 'https://example.com/arg155/%';
