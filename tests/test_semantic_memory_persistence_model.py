from jarvis_platform.db.models.semantic_memory_model import SemanticMemoryModel


def test_semantic_memory_model_imports() -> None:
    assert SemanticMemoryModel.__tablename__ == "semantic_memories"


def test_semantic_memory_model_uses_memory_schema() -> None:
    assert SemanticMemoryModel.__table__.schema == "memory"


def test_semantic_memory_model_has_expected_columns() -> None:
    columns = set(SemanticMemoryModel.__table__.columns.keys())

    assert {
        "id",
        "memory_type",
        "content",
        "summary",
        "tags",
        "importance_score",
        "source",
        "embedding_json",
        "created_at",
        "updated_at",
        "expires_at",
        "metadata",
    }.issubset(columns)
