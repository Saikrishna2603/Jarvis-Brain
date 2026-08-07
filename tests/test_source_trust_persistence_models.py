from jarvis_platform.db.models import SourceTrustModel


def test_source_trust_model_imports() -> None:
    assert SourceTrustModel is not None


def test_source_trust_model_table_name() -> None:
    assert SourceTrustModel.__tablename__ == "source_trust_profiles"


def test_source_trust_model_uses_knowledge_schema() -> None:
    assert SourceTrustModel.__table__.schema == "knowledge"
