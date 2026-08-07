from jarvis_brain.llm.llm_model_router import LLMModelRouter
from jarvis_platform.schemas.llm import LLMTaskType


def test_python_error_routes_to_coding() -> None:
    assert LLMModelRouter().detect_task_type("Python error") == LLMTaskType.CODING


def test_fastapi_traceback_routes_to_coding() -> None:
    assert LLMModelRouter().route("FastAPI traceback")["task_type"] == "coding"


def test_docker_compose_bug_routes_to_coding() -> None:
    assert LLMModelRouter().route("Docker compose bug")["task_type"] == "coding"


def test_general_jarvis_question_routes_to_general() -> None:
    assert LLMModelRouter().route("What is Jarvis?")["task_type"] == "general"


def test_planning_question_routes_to_general_model() -> None:
    route = LLMModelRouter().route("Make an architecture plan")

    assert route["task_type"] == "planning"
    assert route["model"] == "llama3.1:8b"


def test_coding_task_selects_qwen() -> None:
    assert LLMModelRouter().select_model(LLMTaskType.CODING) == "qwen2.5-coder:7b"


def test_classification_task_selects_small_qwen() -> None:
    assert (
        LLMModelRouter().select_model(LLMTaskType.INTENT_RESOLUTION)
        == "qwen2.5-coder:3b"
    )


def test_general_task_selects_llama() -> None:
    assert LLMModelRouter().select_model(LLMTaskType.GENERAL) == "llama3.1:8b"


def test_metadata_can_override_task_type() -> None:
    task_type = LLMModelRouter().detect_task_type(
        "Hello",
        metadata={"task_type": "coding"},
    )

    assert task_type == LLMTaskType.CODING


def test_classify_does_not_match_coding_class_keyword() -> None:
    route = LLMModelRouter().route("Classify this intent")

    assert route == {
        "task_type": "intent_resolution",
        "model": "qwen2.5-coder:3b",
    }
