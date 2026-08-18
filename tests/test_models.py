"""ORM 모델 단위 테스트.

DB 연결 없이 모델의 구조, 컬럼, 관계, Enum, 제약조건을 검증한다.
"""

import uuid

from sqlalchemy import inspect

from argos.models import Base, CrawlQueue, TechItem, TechSuccession, TrackHistory, UserAsset
from argos.models.tech_item import CategoryType
from argos.models.tech_succession import RelationType
from argos.models.user_asset import AssetStatus


# ──────────────────────────────────────────
# Base 메타데이터 테스트
# ──────────────────────────────────────────

class TestBaseMetadata:
    """Base.metadata에 테이블이 정상 등록되었는지 검증."""

    def test_all_tables_registered(self):
        table_names = set(Base.metadata.tables.keys())
        expected = {
            "crawl_queue",
            "tech_items",
            "tech_succession",
            "user_assets",
            "track_history",
            "feed_events",
            "tech_events",
            "event_documents",
            "entities",
            "event_entities",
        }
        assert expected == table_names

    def test_metadata_is_not_empty(self):
        assert len(Base.metadata.tables) == 10


# ──────────────────────────────────────────
# CrawlQueue 모델 테스트
# ──────────────────────────────────────────

class TestCrawlQueueModel:
    """crawl_queue 스테이징 테이블 ORM 모델 검증."""

    def test_tablename(self):
        assert CrawlQueue.__tablename__ == "crawl_queue"

    def test_required_columns_exist(self):
        mapper = inspect(CrawlQueue)
        column_names = {col.key for col in mapper.columns}
        required = {"id", "source_url", "raw_content", "source", "source_category",
                    "published_at", "queued_at"}
        assert required.issubset(column_names)

    def test_id_is_uuid_primary_key(self):
        mapper = inspect(CrawlQueue)
        pk_cols = [col.name for col in mapper.columns if col.primary_key]
        assert "id" in pk_cols

    def test_source_url_is_unique(self):
        mapper = inspect(CrawlQueue)
        col = mapper.columns["source_url"]
        assert col.unique is True

    def test_published_at_is_nullable(self):
        mapper = inspect(CrawlQueue)
        col = mapper.columns["published_at"]
        assert col.nullable is True

    def test_raw_content_is_nullable(self):
        mapper = inspect(CrawlQueue)
        col = mapper.columns["raw_content"]
        assert col.nullable is True

    def test_image_url_is_nullable_varchar_2048(self):
        mapper = inspect(CrawlQueue)
        col = mapper.columns["image_url"]
        assert col.nullable is True
        assert col.type.length == 2048

    def test_queued_at_is_not_nullable(self):
        mapper = inspect(CrawlQueue)
        col = mapper.columns["queued_at"]
        assert col.nullable is False


# ──────────────────────────────────────────
# TechItem 모델 테스트
# ──────────────────────────────────────────

class TestTechItemModel:
    """tech_items 테이블 ORM 모델 검증."""

    def test_tablename(self):
        assert TechItem.__tablename__ == "tech_items"

    def test_required_columns_exist(self):
        mapper = inspect(TechItem)
        column_names = {col.key for col in mapper.columns}
        required = {"id", "title", "source_url", "raw_content", "embedding",
                     "category", "trust_score", "created_at", "updated_at"}
        assert required.issubset(column_names)

    def test_id_is_uuid_primary_key(self):
        mapper = inspect(TechItem)
        pk_cols = [col.name for col in mapper.columns if col.primary_key]
        assert "id" in pk_cols

    def test_source_url_is_unique(self):
        mapper = inspect(TechItem)
        source_url_col = mapper.columns["source_url"]
        assert source_url_col.unique is True

    def test_category_enum_values(self):
        assert CategoryType.MAINSTREAM.value == "Mainstream"
        assert CategoryType.ALPHA.value == "Alpha"
        assert len(CategoryType) == 2

    def test_trust_score_default(self):
        mapper = inspect(TechItem)
        trust_score_col = mapper.columns["trust_score"]
        assert trust_score_col.nullable is True

    def test_embedding_column_nullable(self):
        """임베딩은 초기 수집 시 없을 수 있으므로 nullable이어야 한다."""
        mapper = inspect(TechItem)
        embedding_col = mapper.columns["embedding"]
        assert embedding_col.nullable is True

    def test_published_at_column(self):
        """published_at은 원문 발행일이며 nullable + timezone-aware + indexed여야 한다."""
        mapper = inspect(TechItem)
        published_at_col = mapper.columns["published_at"]
        assert published_at_col.nullable is True
        assert published_at_col.type.timezone is True
        assert published_at_col.index is True

    def test_image_url_column(self):
        """image_url은 nullable VARCHAR(2048)이어야 한다 (og:image 저장용)."""
        mapper = inspect(TechItem)
        image_url_col = mapper.columns["image_url"]
        assert image_url_col.nullable is True
        assert image_url_col.type.length == 2048

    def test_relationships_defined(self):
        mapper = inspect(TechItem)
        relationship_names = {rel.key for rel in mapper.relationships}
        assert "predecessors" in relationship_names
        assert "successors" in relationship_names
        assert "user_assets" in relationship_names

    def test_repr(self):
        item = TechItem(
            title="A" * 50,
            source_url="https://example.com",
            raw_content="test",
        )
        result = repr(item)
        assert "TechItem" in result
        assert "..." in result  # 30자 초과 시 truncate


# ──────────────────────────────────────────
# TechSuccession 모델 테스트
# ──────────────────────────────────────────

class TestTechSuccessionModel:
    """tech_succession 테이블 ORM 모델 검증."""

    def test_tablename(self):
        assert TechSuccession.__tablename__ == "tech_succession"

    def test_required_columns_exist(self):
        mapper = inspect(TechSuccession)
        column_names = {col.key for col in mapper.columns}
        required = {"id", "predecessor_id", "successor_id", "relation_type", "reasoning"}
        assert required.issubset(column_names)

    def test_foreign_keys_to_tech_items(self):
        mapper = inspect(TechSuccession)
        predecessor_col = mapper.columns["predecessor_id"]
        successor_col = mapper.columns["successor_id"]

        pred_fk_targets = {fk.target_fullname for fk in predecessor_col.foreign_keys}
        succ_fk_targets = {fk.target_fullname for fk in successor_col.foreign_keys}

        assert "tech_items.id" in pred_fk_targets
        assert "tech_items.id" in succ_fk_targets

    def test_relation_type_enum_values(self):
        assert RelationType.REPLACE.value == "Replace"
        assert RelationType.ENHANCE.value == "Enhance"
        assert RelationType.FORK.value == "Fork"
        assert len(RelationType) == 3

    def test_cascade_delete_on_foreign_keys(self):
        """tech_item 삭제 시 연관 succession도 삭제되어야 한다."""
        mapper = inspect(TechSuccession)
        for col_name in ("predecessor_id", "successor_id"):
            col = mapper.columns[col_name]
            for fk in col.foreign_keys:
                assert fk.ondelete == "CASCADE"

    def test_relationships_defined(self):
        mapper = inspect(TechSuccession)
        relationship_names = {rel.key for rel in mapper.relationships}
        assert "predecessor" in relationship_names
        assert "successor" in relationship_names

    def test_repr(self):
        obj = TechSuccession(
            predecessor_id=uuid.uuid4(),
            successor_id=uuid.uuid4(),
            relation_type=RelationType.REPLACE,
        )
        result = repr(obj)
        assert "Replace" in result
        assert "TechSuccession" in result


# ──────────────────────────────────────────
# UserAsset 모델 테스트
# ──────────────────────────────────────────

class TestUserAssetModel:
    """user_assets 테이블 ORM 모델 검증."""

    def test_tablename(self):
        assert UserAsset.__tablename__ == "user_assets"

    def test_required_columns_exist(self):
        mapper = inspect(UserAsset)
        column_names = {col.key for col in mapper.columns}
        required = {"id", "tech_id", "status", "last_monitored_at", "created_at", "updated_at"}
        assert required.issubset(column_names)

    def test_tech_id_foreign_key(self):
        mapper = inspect(UserAsset)
        tech_id_col = mapper.columns["tech_id"]
        fk_targets = {fk.target_fullname for fk in tech_id_col.foreign_keys}
        assert "tech_items.id" in fk_targets

    def test_status_enum_values(self):
        """ERD 기준 3개 상태: Keep, Tracking, Archived."""
        assert AssetStatus.KEEP.value == "Keep"
        assert AssetStatus.TRACKING.value == "Tracking"
        assert AssetStatus.ARCHIVED.value == "Archived"
        assert len(AssetStatus) == 3

    def test_last_monitored_at_nullable(self):
        """최초 Keep 시에는 모니터링 기록이 없으므로 nullable."""
        mapper = inspect(UserAsset)
        col = mapper.columns["last_monitored_at"]
        assert col.nullable is True

    def test_relationships_defined(self):
        mapper = inspect(UserAsset)
        relationship_names = {rel.key for rel in mapper.relationships}
        assert "tech_item" in relationship_names
        assert "history" in relationship_names


# ──────────────────────────────────────────
# TrackHistory 모델 테스트
# ──────────────────────────────────────────

class TestTrackHistoryModel:
    """track_history 테이블 ORM 모델 검증."""

    def test_tablename(self):
        assert TrackHistory.__tablename__ == "track_history"

    def test_required_columns_exist(self):
        mapper = inspect(TrackHistory)
        column_names = {col.key for col in mapper.columns}
        required = {"id", "user_asset_id", "changed_from", "changed_to", "changed_at"}
        assert required.issubset(column_names)

    def test_user_asset_id_foreign_key(self):
        mapper = inspect(TrackHistory)
        col = mapper.columns["user_asset_id"]
        fk_targets = {fk.target_fullname for fk in col.foreign_keys}
        assert "user_assets.id" in fk_targets

    def test_cascade_delete_on_user_asset_fk(self):
        mapper = inspect(TrackHistory)
        col = mapper.columns["user_asset_id"]
        for fk in col.foreign_keys:
            assert fk.ondelete == "CASCADE"

    def test_changed_from_to_not_nullable(self):
        mapper = inspect(TrackHistory)
        assert mapper.columns["changed_from"].nullable is False
        assert mapper.columns["changed_to"].nullable is False

    def test_relationship_to_user_asset(self):
        mapper = inspect(TrackHistory)
        relationship_names = {rel.key for rel in mapper.relationships}
        assert "user_asset" in relationship_names

    def test_repr(self):
        obj = TrackHistory(
            user_asset_id=uuid.uuid4(),
            changed_from="Keep",
            changed_to="Archived",
        )
        result = repr(obj)
        assert "Keep" in result
        assert "Archived" in result


# ──────────────────────────────────────────
# FeedEvent 모델 테스트 (ARG-207)
# ──────────────────────────────────────────

def test_feed_event_model_shape():
    from argos.models.feed_event import FeedEvent, FeedEventType

    assert FeedEventType.IMPRESSION.value == "Impression"
    assert FeedEventType.CLICK.value == "Click"
    assert FeedEventType.DWELL.value == "Dwell"
    ev = FeedEvent(event_type=FeedEventType.DWELL, tech_item_id=None, value=3.5)
    assert ev.event_type == FeedEventType.DWELL
    assert ev.value == 3.5
    assert FeedEvent.__tablename__ == "feed_events"


# ──────────────────────────────────────────
# Docker / Alembic 설정 파일 테스트
# ──────────────────────────────────────────

class TestInfraFiles:
    """인프라 설정 파일의 핵심 내용을 검증한다."""

    def test_docker_compose_exists_and_valid(self):
        from pathlib import Path
        compose_path = Path(__file__).resolve().parents[1] / "docker-compose.yml"
        assert compose_path.exists()
        content = compose_path.read_text(encoding="utf-8")
        assert "pgvector/pgvector:pg16" in content
        assert "pgdata:/var/lib/postgresql/data" in content
        assert "init.sql" in content

    def test_init_sql_has_vector_extension(self):
        from pathlib import Path
        init_path = Path(__file__).resolve().parents[1] / "init.sql"
        assert init_path.exists()
        content = init_path.read_text(encoding="utf-8")
        assert "CREATE EXTENSION IF NOT EXISTS vector" in content
        assert "uuid-ossp" in content

    def test_alembic_env_imports_models(self):
        from pathlib import Path
        env_path = Path(__file__).resolve().parents[1] / "alembic" / "env.py"
        assert env_path.exists()
        content = env_path.read_text(encoding="utf-8")
        assert "from argos.models import Base" in content
        assert "target_metadata = Base.metadata" in content

    def test_gitignore_has_essentials(self):
        from pathlib import Path
        gitignore_path = Path(__file__).resolve().parents[1] / ".gitignore"
        assert gitignore_path.exists()
        content = gitignore_path.read_text(encoding="utf-8")
        assert "pgdata/" in content
        assert ".env" in content
        assert "__pycache__" in content


# ──────────────────────────────────────────
# TechEvent 모델 테스트 (ARG-254)
# ──────────────────────────────────────────

class TestTechEventModel:
    """tech_events — 사건 테이블. 병합은 삭제가 아니라 툼스톤이다."""

    def test_tablename_is_tech_events_not_events(self):
        # feed_events(행동 로그)와 헷갈리면 안 되므로 이름을 못박는다.
        from argos.models import TechEvent

        assert TechEvent.__tablename__ == "tech_events"

    def test_required_columns_exist(self):
        from argos.models import TechEvent

        mapper = inspect(TechEvent)
        column_names = {col.key for col in mapper.columns}
        required = {
            "id",
            "title",
            "summary",
            "occurred_at",
            "naming_stale",
            "merged_into_id",
            "created_at",
            "updated_at",
        }
        assert required.issubset(column_names)

    def test_no_embedding_column(self):
        # A1: 사건 임베딩은 이번 범위 밖이다.
        from argos.models import TechEvent

        mapper = inspect(TechEvent)
        assert "embedding" not in {col.key for col in mapper.columns}

    def test_id_is_uuid_primary_key(self):
        from argos.models import TechEvent

        mapper = inspect(TechEvent)
        pk_cols = [col.name for col in mapper.columns if col.primary_key]
        assert pk_cols == ["id"]

    def test_title_and_summary_are_nullable(self):
        # 이름 짓기는 2단계이므로 사건은 이름 없이 생길 수 있어야 한다.
        from argos.models import TechEvent

        columns = inspect(TechEvent).columns
        assert columns["title"].nullable is True
        assert columns["summary"].nullable is True

    def test_occurred_at_and_naming_stale_are_not_nullable(self):
        from argos.models import TechEvent

        columns = inspect(TechEvent).columns
        assert columns["occurred_at"].nullable is False
        assert columns["naming_stale"].nullable is False

    def test_naming_stale_defaults_to_false(self):
        from argos.models import TechEvent

        event = TechEvent()
        # SQLAlchemy는 flush 전까지 Python-side default를 적용하지 않으므로
        # 컬럼 정의에 default가 걸려 있는지를 직접 본다.
        default = inspect(TechEvent).columns["naming_stale"].default
        assert default is not None
        assert default.arg is False
        assert event.naming_stale is None  # flush 전 상태

    def test_merged_into_id_is_nullable_self_fk(self):
        from argos.models import TechEvent

        column = inspect(TechEvent).columns["merged_into_id"]
        assert column.nullable is True
        fks = list(column.foreign_keys)
        assert len(fks) == 1
        assert fks[0].column.table.name == "tech_events"

    def test_merged_into_fk_restricts_delete_to_keep_tombstones(self):
        # 툼스톤은 절대 연쇄 삭제되면 안 된다 — CASCADE였다면 옛 링크가 죽는다.
        from argos.models import TechEvent

        fk = list(inspect(TechEvent).columns["merged_into_id"].foreign_keys)[0]
        assert fk.ondelete == "RESTRICT"

    def test_merged_into_id_is_indexed_for_chain_walks(self):
        from argos.models import TechEvent

        assert inspect(TechEvent).columns["merged_into_id"].index is True

    def test_self_referential_relationships_exist(self):
        from argos.models import TechEvent

        relationships = {rel.key for rel in inspect(TechEvent).relationships}
        assert {"merged_into", "merged_from"}.issubset(relationships)


# ──────────────────────────────────────────
# EventDocument 링크 테이블 테스트 (ARG-255)
# ──────────────────────────────────────────

class TestEventDocumentModel:
    """event_documents — 사건↔문서 N:N 근거 링크."""

    def test_tablename(self):
        from argos.models import EventDocument

        assert EventDocument.__tablename__ == "event_documents"

    def test_required_columns_exist(self):
        from argos.models import EventDocument

        column_names = {col.key for col in inspect(EventDocument).columns}
        assert {"id", "event_id", "tech_item_id"}.issubset(column_names)

    def test_event_id_points_at_tech_events(self):
        from argos.models import EventDocument

        fk = list(inspect(EventDocument).columns["event_id"].foreign_keys)[0]
        assert fk.column.table.name == "tech_events"
        assert fk.ondelete == "CASCADE"

    def test_tech_item_id_points_at_tech_items(self):
        from argos.models import EventDocument

        fk = list(inspect(EventDocument).columns["tech_item_id"].foreign_keys)[0]
        assert fk.column.table.name == "tech_items"
        assert fk.ondelete == "CASCADE"

    def test_same_document_cannot_be_linked_twice_to_one_event(self):
        from sqlalchemy import UniqueConstraint

        from argos.models import EventDocument

        unique_sets = {
            tuple(sorted(col.name for col in constraint.columns))
            for constraint in EventDocument.__table__.constraints
            if isinstance(constraint, UniqueConstraint)
        }
        assert ("event_id", "tech_item_id") in unique_sets

    def test_tech_items_gains_no_event_column(self):
        # A2: tech_items 스키마는 손대지 않는다.
        from argos.models import TechItem

        column_names = {col.key for col in inspect(TechItem).columns}
        assert "event_id" not in column_names
        assert not any(name.startswith("event") for name in column_names)


# ──────────────────────────────────────────
# Entity / EventEntity 테스트 (ARG-256)
# ──────────────────────────────────────────

class TestEntityModel:
    """entities — 이름 사전. 정규화 키 기준으로 같은 이름이 하나로 모인다."""

    def test_tablename(self):
        from argos.models import Entity

        assert Entity.__tablename__ == "entities"

    def test_keeps_both_display_name_and_normalized_key(self):
        from argos.models import Entity

        column_names = {col.key for col in inspect(Entity).columns}
        assert {"name", "normalized_key", "kind"}.issubset(column_names)

    def test_normalized_key_is_unique(self):
        from argos.models import Entity

        assert inspect(Entity).columns["normalized_key"].unique is True

    def test_kind_is_nullable(self):
        # A4: 분류 로직은 다음 사이클이므로 kind 없이도 저장돼야 한다.
        from argos.models import Entity

        assert inspect(Entity).columns["kind"].nullable is True

    def test_kind_enum_values_are_pascal_case(self):
        from argos.models.entity import EntityKind

        values = {member.value for member in EntityKind}
        assert "Technology" in values
        assert "Organization" in values
        assert all(value[0].isupper() for value in values)


class TestEventEntityModel:
    """event_entities — 사건↔엔티티 N:N 링크."""

    def test_tablename(self):
        from argos.models import EventEntity

        assert EventEntity.__tablename__ == "event_entities"

    def test_event_id_points_at_tech_events(self):
        from argos.models import EventEntity

        fk = list(inspect(EventEntity).columns["event_id"].foreign_keys)[0]
        assert fk.column.table.name == "tech_events"

    def test_entity_id_points_at_entities(self):
        from argos.models import EventEntity

        fk = list(inspect(EventEntity).columns["entity_id"].foreign_keys)[0]
        assert fk.column.table.name == "entities"

    def test_same_entity_cannot_be_linked_twice_to_one_event(self):
        from sqlalchemy import UniqueConstraint

        from argos.models import EventEntity

        unique_sets = {
            tuple(sorted(col.name for col in constraint.columns))
            for constraint in EventEntity.__table__.constraints
            if isinstance(constraint, UniqueConstraint)
        }
        assert ("entity_id", "event_id") in unique_sets
