from app.knowledge.evidence_manager import EvidenceManager
from app.knowledge.evidence_verifier import EvidenceVerifier
from app.knowledge.retrieval_planner import RetrievalPlanner
from app.retrieval.retrieval_registry import RetrievalRegistry
from jarvis_platform.schemas.evidence import EvidenceItem, EvidenceStatus, EvidenceTrustLevel, EvidenceType


def test_car_thermostat_request_produces_automotive_domain() -> None:
    planner = RetrievalPlanner()

    result = planner.plan("how do I fix my car thermostat")

    assert result["domain"] == "automotive_repair"


def test_car_thermostat_request_needs_user_input() -> None:
    planner = RetrievalPlanner()

    result = planner.plan("how do I fix my car thermostat")

    assert result["needs_user_input"] is True


def test_car_thermostat_request_cannot_answer_now() -> None:
    planner = RetrievalPlanner()

    result = planner.plan("how do I fix my car thermostat")

    assert result["can_answer_now"] is False


def test_car_thermostat_request_creates_retrieval_requests() -> None:
    planner = RetrievalPlanner()

    result = planner.plan("how do I fix my car thermostat")

    assert result["retrieval_requests"]


def test_car_thermostat_request_saves_evidence_into_evidence_manager() -> None:
    evidence_manager = EvidenceManager()
    planner = RetrievalPlanner(evidence_manager=evidence_manager)

    planner.plan("how do I fix my car thermostat")

    assert evidence_manager.get_all_evidence()


def test_coding_request_with_missing_context_needs_user_input() -> None:
    planner = RetrievalPlanner()

    result = planner.plan("I have a Python error in FastAPI")

    assert result["domain"] == "coding"
    assert result["needs_user_input"] is True


def test_general_knowledge_request_can_answer_if_no_gaps() -> None:
    planner = RetrievalPlanner()

    result = planner.plan("What is the capital of France?")

    assert result["domain"] == "general_knowledge"
    assert result["gaps"] == []
    assert result["can_answer_now"] is True


def test_summarize_plan_mentions_needed_details() -> None:
    planner = RetrievalPlanner()

    result = planner.plan("how do I fix my car thermostat")

    assert "need a few details" in result["summary"].lower()


def test_risky_domain_with_gaps_cannot_answer_now() -> None:
    planner = RetrievalPlanner()

    result = planner.plan("We have a suspicious login incident")

    assert result["domain"] == "cybersecurity"
    assert result["gaps"]
    assert result["can_answer_now"] is False


def test_evidence_manager_contains_discovered_evidence_after_plan() -> None:
    evidence_manager = EvidenceManager()
    planner = RetrievalPlanner(evidence_manager=evidence_manager)

    result = planner.plan("I have a Python error in FastAPI")

    assert len(evidence_manager.get_all_evidence()) == len(result["evidence"])


def test_retrieval_planner_verifies_evidence_before_saving() -> None:
    evidence_manager = EvidenceManager()
    planner = RetrievalPlanner(evidence_manager=evidence_manager)

    result = planner.plan("I have a Python error in FastAPI")

    saved = evidence_manager.get_all_evidence()
    assert saved
    assert any(item.trust_level != EvidenceTrustLevel.UNKNOWN for item in saved)
    assert any(item.warnings for item in saved)
    assert result["verified_evidence"] == result["evidence"]


def test_retrieval_planner_includes_source_trust_summary() -> None:
    planner = RetrievalPlanner()

    result = planner.plan("how do I fix my car thermostat")

    assert result["source_trust_summary"]["source_trust_enabled"] is True
    assert "usable_evidence_count" in result["source_trust_summary"]


class BlockingEvidenceVerifier(EvidenceVerifier):
    """Test verifier that marks all evidence blocked."""

    def verify_many(self, items, domain=None):
        return [
            item.model_copy(
                update={
                    "trust_level": EvidenceTrustLevel.BLOCKED,
                    "status": EvidenceStatus.REJECTED,
                    "warnings": ["Blocked test evidence."],
                }
            )
            for item in items
        ]

    def get_usable_verified_evidence(self, items, domain=None):
        return []


def test_risky_domain_with_untrusted_evidence_cannot_answer_detailed_steps() -> None:
    planner = RetrievalPlanner(evidence_verifier=BlockingEvidenceVerifier())

    result = planner.plan("how do I fix my car thermostat")

    assert result["can_answer_now"] is False
    assert result["needs_verification"] is True
    assert "trusted evidence" in result["summary"].lower()


class FakeRegistry(RetrievalRegistry):
    """Simple registry test double."""

    def __init__(self, evidence: list[EvidenceItem]) -> None:
        super().__init__()
        self.evidence = evidence
        self.called = False

    def retrieve_many(self, requests):
        self.called = True
        return self.evidence

    def list_drivers(self):
        return []


def test_retrieval_planner_works_without_registry_as_before() -> None:
    planner = RetrievalPlanner()

    result = planner.plan("I have a Python error in FastAPI")

    assert result["metadata"]["retrieval_registry_enabled"] is False
    assert result["evidence"]


def test_retrieval_planner_uses_registry_when_provided() -> None:
    registry_evidence = EvidenceItem(
        evidence_id="registry-ev",
        title="Registry evidence",
        summary="Official documentation evidence.",
        evidence_type=EvidenceType.DOCUMENTATION,
        source_name="Python Docs",
        source_url="https://docs.python.org/3/",
    )
    registry = FakeRegistry([registry_evidence])
    planner = RetrievalPlanner(retrieval_registry=registry)

    result = planner.plan("I have a Python error in FastAPI")

    assert registry.called is True
    assert result["metadata"]["retrieval_registry_enabled"] is True
    assert any(item.evidence_id == "registry-ev" for item in result["evidence"])


def test_retrieval_planner_registry_evidence_is_verified() -> None:
    registry_evidence = EvidenceItem(
        evidence_id="registry-ev",
        title="Registry evidence",
        summary="Official documentation evidence.",
        evidence_type=EvidenceType.DOCUMENTATION,
        source_name="Python Docs",
        source_url="https://docs.python.org/3/",
    )
    planner = RetrievalPlanner(retrieval_registry=FakeRegistry([registry_evidence]))

    result = planner.plan("I have a Python error in FastAPI")

    found = [item for item in result["evidence"] if item.evidence_id == "registry-ev"][0]
    assert found.trust_level.value == "trusted"


def test_retrieval_planner_falls_back_to_mock_if_registry_returns_no_evidence() -> None:
    planner = RetrievalPlanner(retrieval_registry=FakeRegistry([]))

    result = planner.plan("I have a Python error in FastAPI")

    assert result["evidence"]
    assert result["metadata"]["real_retrieval_attempted"] is True
