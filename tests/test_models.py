"""ORM 모델 단위 테스트.

DB 연결 없이 모델의 구조, 컬럼, 관계, Enum, 제약조건을 검증한다.
"""

import uuid

from sqlalchemy import inspect

from argos.models import Base, TechItem, TechSuccession, TrackHistory, UserAsset
from argos.models.tech_item import CategoryType
from argos.models.tech_succession import RelationType
from argos.models.user_asset import AssetStatus


# ──────────────────────────────────────────
# Base 메타데이터 테스트
# ──────────────────────────────────────────

class TestBaseMetadata:
    """Base.metadata에 4개 테이블이 정상 등록되었는지 검증."""

    def test_all_tables_registered(self):
        table_names = set(Base.metadata.tables.keys())
        expected = {"tech_items", "tech_succession", "user_assets", "track_history"}
        assert expected == table_names

    def test_metadata_is_not_empty(self):
        assert len(Base.metadata.tables) == 4


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
