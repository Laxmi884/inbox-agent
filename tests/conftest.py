import pytest


@pytest.fixture(autouse=True)
def isolate_with_offline_backend(monkeypatch):
    """Isolate all tests from live backend detection by defaulting to offline."""
    monkeypatch.setenv("INBOX_LLM_BACKEND", "offline")
