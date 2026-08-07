from jarvis_platform.db.base import Base
from jarvis_platform.db.models import (
    WorldEventModel,
    WorldImpactAnalysisModel,
    WorldSuggestionModel,
)


def test_world_persistence_models_import_and_register_tables() -> None:
    table_names = set(Base.metadata.tables)

    assert WorldEventModel.__tablename__ == "world_events"
    assert WorldImpactAnalysisModel.__tablename__ == "world_impact_analyses"
    assert WorldSuggestionModel.__tablename__ == "world_suggestions"
    assert "world.world_events" in table_names
    assert "world.world_impact_analyses" in table_names
    assert "world.world_suggestions" in table_names
