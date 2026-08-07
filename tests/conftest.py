"""Test configuration. Async tests use anyio, matching the source convention."""

import pytest


@pytest.fixture
def anyio_backend() -> str:
    return "asyncio"
