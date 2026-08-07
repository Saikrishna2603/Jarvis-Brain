from jarvis_platform.db.base import Base
from jarvis_platform.db.models import AuditLogModel, EventLogModel, TaskMemoryModel


def test_database_models_import_and_register_tables() -> None:
    table_names = set(Base.metadata.tables)

    assert AuditLogModel.__tablename__ == "audit_logs"
    assert TaskMemoryModel.__tablename__ == "task_memories"
    assert EventLogModel.__tablename__ == "event_logs"
    assert "system.audit_logs" in table_names
    assert "memory.task_memories" in table_names
    assert "system.event_logs" in table_names
