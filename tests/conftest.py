import pytest


@pytest.fixture(autouse=True)
def isolate_with_offline_backend(monkeypatch):
    """Isolate all tests from live backend detection by defaulting to offline."""
    monkeypatch.setenv("INBOX_LLM_BACKEND", "offline")
    # Tests never trace, whatever the developer's .env says: a trace would
    # upload prompt content, and prompts carry real mail.
    monkeypatch.setenv("LANGSMITH_TRACING", "false")


import sys as _sys
from pathlib import Path as _Path

# test_client_contract imports FakeGmailApi from test_live_gmail. The fake is
# the contract fixture for the live client and belongs beside its own tests
# rather than in a third module that neither owns.
_sys.path.insert(0, str(_Path(__file__).parent))
