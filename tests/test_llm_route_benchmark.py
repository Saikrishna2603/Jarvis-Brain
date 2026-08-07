from scripts.benchmark_llm_routes import summarize


def test_benchmark_summary_is_deterministic() -> None:
    result = summarize([10.0, 20.0, 30.0, 40.0])
    assert result == {
        "minimum": 10.0,
        "p50": 25.0,
        "p95": 38.5,
        "maximum": 40.0,
    }


def test_benchmark_summary_handles_no_measurements() -> None:
    assert summarize([]) == {
        "minimum": None,
        "p50": None,
        "p95": None,
        "maximum": None,
    }
