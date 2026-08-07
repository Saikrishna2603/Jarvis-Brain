from sqlalchemy import or_, select
from sqlalchemy.orm import Session

from jarvis_platform.db.models.world_impact_model import WorldImpactAnalysisModel
from jarvis_platform.db.models.world_suggestion_model import WorldSuggestionModel
from jarvis_platform.schemas.world_impact import WorldImpactAnalysis
from jarvis_platform.schemas.world_suggestion import (
    SuggestionPriority,
    SuggestionStatus,
    SuggestionType,
    WorldSuggestion,
)


class WorldIntelligenceRepository:
    """Persist impact analyses and suggestions for world intelligence."""

    def __init__(self, session: Session) -> None:
        """Store the database session used by this repository."""
        self._session = session

    def save_analysis(self, analysis: WorldImpactAnalysis) -> WorldImpactAnalysis:
        """Create or update one impact analysis by analysis_id."""
        model = self._session.get(WorldImpactAnalysisModel, analysis.analysis_id)
        if model is None:
            model = self._analysis_to_model(analysis)
            self._session.add(model)
        else:
            self._update_analysis_model(model, analysis)

        self._session.commit()
        self._session.refresh(model)
        return self._analysis_to_schema(model)

    def save_analyses(self, analyses: list[WorldImpactAnalysis]) -> list[WorldImpactAnalysis]:
        """Create or update several impact analyses."""
        return [self.save_analysis(analysis) for analysis in analyses]

    def get_all_analyses(self) -> list[WorldImpactAnalysis]:
        """Return every persisted impact analysis."""
        statement = select(WorldImpactAnalysisModel).order_by(WorldImpactAnalysisModel.created_at)
        return [self._analysis_to_schema(model) for model in self._session.scalars(statement)]

    def get_analyses_by_event(self, event_id: str) -> list[WorldImpactAnalysis]:
        """Return impact analyses for one world event."""
        statement = (
            select(WorldImpactAnalysisModel)
            .where(WorldImpactAnalysisModel.event_id == event_id)
            .order_by(WorldImpactAnalysisModel.created_at)
        )
        return [self._analysis_to_schema(model) for model in self._session.scalars(statement)]

    def save_suggestion(self, suggestion: WorldSuggestion) -> WorldSuggestion:
        """Create or update one world suggestion by suggestion_id."""
        model = self._session.get(WorldSuggestionModel, suggestion.suggestion_id)
        if model is None:
            model = self._suggestion_to_model(suggestion)
            self._session.add(model)
        else:
            self._update_suggestion_model(model, suggestion)

        self._session.commit()
        self._session.refresh(model)
        return self._suggestion_to_schema(model)

    def save_suggestions(self, suggestions: list[WorldSuggestion]) -> list[WorldSuggestion]:
        """Create or update several world suggestions."""
        return [self.save_suggestion(suggestion) for suggestion in suggestions]

    def get_all_suggestions(self) -> list[WorldSuggestion]:
        """Return every persisted world suggestion."""
        statement = select(WorldSuggestionModel).order_by(WorldSuggestionModel.created_at)
        return [self._suggestion_to_schema(model) for model in self._session.scalars(statement)]

    def get_suggestions_by_event(self, event_id: str) -> list[WorldSuggestion]:
        """Return suggestions for one world event."""
        statement = (
            select(WorldSuggestionModel)
            .where(WorldSuggestionModel.event_id == event_id)
            .order_by(WorldSuggestionModel.created_at)
        )
        return [self._suggestion_to_schema(model) for model in self._session.scalars(statement)]

    def get_alert_suggestions(self) -> list[WorldSuggestion]:
        """Return suggestions that should be shown as alerts."""
        statement = (
            select(WorldSuggestionModel)
            .where(
                or_(
                    WorldSuggestionModel.priority.in_(["high", "urgent"]),
                    WorldSuggestionModel.suggestion_type == "escalate_alert",
                    WorldSuggestionModel.requires_user_approval.is_(True),
                )
            )
            .order_by(WorldSuggestionModel.created_at)
        )
        return [self._suggestion_to_schema(model) for model in self._session.scalars(statement)]

    def _analysis_to_model(self, analysis: WorldImpactAnalysis) -> WorldImpactAnalysisModel:
        """Convert a WorldImpactAnalysis schema object to a model."""
        return WorldImpactAnalysisModel(
            id=analysis.analysis_id,
            event_id=analysis.event_id,
            event_title=analysis.event_title,
            category=analysis.category,
            severity=analysis.severity,
            relevance_score=analysis.relevance_score,
            impact_summary=analysis.impact_summary,
            affected_areas_json=list(analysis.affected_areas),
            possible_outcomes_json=list(analysis.possible_outcomes),
            suggested_actions_json=list(analysis.suggested_actions),
            risk_level=analysis.risk_level,
            confidence_score=analysis.confidence_score,
            requires_user_discussion=analysis.requires_user_discussion,
            should_create_task=analysis.should_create_task,
            metadata_json=dict(analysis.metadata),
            created_at=analysis.created_at,
        )

    def _update_analysis_model(
        self,
        model: WorldImpactAnalysisModel,
        analysis: WorldImpactAnalysis,
    ) -> None:
        """Update an existing impact analysis model."""
        updated = self._analysis_to_model(analysis)
        for field_name in [
            "event_id",
            "event_title",
            "category",
            "severity",
            "relevance_score",
            "impact_summary",
            "affected_areas_json",
            "possible_outcomes_json",
            "suggested_actions_json",
            "risk_level",
            "confidence_score",
            "requires_user_discussion",
            "should_create_task",
            "metadata_json",
            "created_at",
        ]:
            setattr(model, field_name, getattr(updated, field_name))

    def _analysis_to_schema(self, model: WorldImpactAnalysisModel) -> WorldImpactAnalysis:
        """Convert a model into a WorldImpactAnalysis schema object."""
        return WorldImpactAnalysis(
            analysis_id=model.id,
            event_id=model.event_id,
            event_title=model.event_title,
            category=model.category,
            severity=model.severity,
            relevance_score=model.relevance_score,
            impact_summary=model.impact_summary,
            affected_areas=model.affected_areas_json or [],
            possible_outcomes=model.possible_outcomes_json or [],
            suggested_actions=model.suggested_actions_json or [],
            risk_level=model.risk_level,
            confidence_score=model.confidence_score,
            requires_user_discussion=model.requires_user_discussion,
            should_create_task=model.should_create_task,
            metadata=model.metadata_json or {},
            created_at=model.created_at,
        )

    def _suggestion_to_model(self, suggestion: WorldSuggestion) -> WorldSuggestionModel:
        """Convert a WorldSuggestion schema object to a model."""
        return WorldSuggestionModel(
            id=suggestion.suggestion_id,
            event_id=suggestion.event_id,
            analysis_id=suggestion.analysis_id,
            title=suggestion.title,
            message=suggestion.message,
            suggestion_type=suggestion.suggestion_type.value,
            priority=suggestion.priority.value,
            status=suggestion.status.value,
            requires_user_approval=suggestion.requires_user_approval,
            suggested_action=suggestion.suggested_action,
            target=suggestion.target,
            rationale=suggestion.rationale,
            metadata_json=dict(suggestion.metadata),
            created_at=suggestion.created_at,
        )

    def _update_suggestion_model(
        self,
        model: WorldSuggestionModel,
        suggestion: WorldSuggestion,
    ) -> None:
        """Update an existing world suggestion model."""
        updated = self._suggestion_to_model(suggestion)
        for field_name in [
            "event_id",
            "analysis_id",
            "title",
            "message",
            "suggestion_type",
            "priority",
            "status",
            "requires_user_approval",
            "suggested_action",
            "target",
            "rationale",
            "metadata_json",
            "created_at",
        ]:
            setattr(model, field_name, getattr(updated, field_name))

    def _suggestion_to_schema(self, model: WorldSuggestionModel) -> WorldSuggestion:
        """Convert a model into a WorldSuggestion schema object."""
        return WorldSuggestion(
            suggestion_id=model.id,
            event_id=model.event_id,
            analysis_id=model.analysis_id,
            title=model.title,
            message=model.message,
            suggestion_type=SuggestionType(model.suggestion_type),
            priority=SuggestionPriority(model.priority),
            status=SuggestionStatus(model.status),
            requires_user_approval=model.requires_user_approval,
            suggested_action=model.suggested_action,
            target=model.target,
            rationale=model.rationale,
            metadata=model.metadata_json or {},
            created_at=model.created_at,
        )
