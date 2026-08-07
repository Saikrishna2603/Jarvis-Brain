from jarvis_platform.db.base import Base
from jarvis_platform.db.models import (
    EvidenceItemModel,
    GuidancePlanModel,
    GuidanceStepModel,
    KnowledgeGapModel,
    RetrievalRequestModel,
)


def test_knowledge_persistence_models_import_and_register_tables() -> None:
    table_names = set(Base.metadata.tables)

    assert KnowledgeGapModel.__tablename__ == "knowledge_gaps"
    assert RetrievalRequestModel.__tablename__ == "retrieval_requests"
    assert EvidenceItemModel.__tablename__ == "evidence_items"
    assert GuidancePlanModel.__tablename__ == "guidance_plans"
    assert GuidanceStepModel.__tablename__ == "guidance_steps"
    assert "knowledge.knowledge_gaps" in table_names
    assert "knowledge.retrieval_requests" in table_names
    assert "knowledge.evidence_items" in table_names
    assert "knowledge.guidance_plans" in table_names
    assert "knowledge.guidance_steps" in table_names
