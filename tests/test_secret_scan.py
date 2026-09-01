# tests/test_secret_scan.py
"""The pre-commit secret scanner.

Both credential leaks in this project were the same mistake: a value copied out
of `.env` - which was handled correctly throughout - into a file that then got
committed. A bot token reached git history that way, and a Cohere key is
plaintext on disk in a notebook that is one `git add .` from the same fate.

A scanner that cries wolf gets bypassed, and a bypassed scanner is worse than
none, so the false-positive cases below are as load-bearing as the true ones.
This repo already contains `sk-or-v1-...` in `.env.example` and `sk-or-v1-test`
in two test files; a scanner that blocks those would have been switched off on
its first day.
"""
from tools.secret_scan import scan_text

ALLOW = "# secret-scan: allow"


def kinds(text, path="x.py"):
    return [f.kind for f in scan_text(text, path)]


# --- the two that actually happened ------------------------------------------

def test_a_telegram_bot_token_is_caught():
    """The exact shape that reached git history in df1f2da."""
    line = 'INBOX_TG_TOKEN = "1234567890:AAF3kQ2mNpXrLvY7wZ1bCdEfGhIjKlMnOpQ"'  # secret-scan: allow
    assert kinds(line) == ["telegram-bot-token"]


def test_a_bare_forty_character_key_in_an_assignment_is_caught():
    """The Cohere key in test.ipynb: no vendor prefix, just entropy in a field
    whose name says what it is."""
    line = 'API_KEY = "drZ5QweOSV7hFYoNqCtBBwbgW4PjVHjuihSoLzRY"'  # secret-scan: allow
    assert kinds(line) == ["assigned-secret"]


def test_a_key_inside_notebook_json_is_caught():
    """The leak lived in a .ipynb, where every line is a JSON string. Scanning
    only .py files would have missed the one that actually happened."""
    line = '    "API_KEY = \\"drZ5QweOSV7hFYoNqCtBBwbgW4PjVHjuihSoLzRY\\"\\n",'
    assert kinds(line, "test.ipynb") == ["assigned-secret"]


# --- other shapes worth refusing ---------------------------------------------

def test_vendor_prefixed_keys_are_caught():
    for line, kind in [
        ('key = "sk-or-v1-9f8e7d6c5b4a39281706f5e4d3c2b1a09f8e7d6c5b4a3928"',  # secret-scan: allow
         "openrouter-key"),
        ('k = "sk-ant-api03-Zx9YwVu8TsRq7PoNmLkJ2iHgFeDcBa1ZyXwVuTsRq7Po"',  # secret-scan: allow
         "anthropic-key"),
        ('k = "lsv2_pt_9f8e7d6c5b4a39281706f5e4d3c2b1a0"', "langsmith-key"),  # secret-scan: allow
        ('k = "AKIAIOSFODNN7EXAMPLE"', "aws-access-key"),  # secret-scan: allow
    ]:
        assert kinds(line) == [kind], line


def test_a_private_key_block_is_caught():
    assert kinds("-----BEGIN RSA PRIVATE KEY-----") == ["private-key"]  # secret-scan: allow


# --- what must NOT trip, or the hook gets disabled ---------------------------

def test_the_placeholder_in_env_example_is_not_a_secret():
    """.env.example is committed on purpose and documents the variable names."""
    assert kinds("# OPENROUTER_API_KEY=sk-or-v1-...", ".env.example") == []


def test_an_obvious_test_fixture_is_not_a_secret():
    """tests/test_config.py has set this for weeks."""
    assert kinds('monkeypatch.setenv("OPENROUTER_API_KEY", "sk-or-v1-test")') == []


def test_reading_a_key_from_the_environment_is_the_thing_we_want():
    assert kinds('API_KEY = os.environ["COHERE_API_KEY"]') == []
    assert kinds('token = os.getenv("INBOX_TG_TOKEN", "")') == []


def test_a_git_sha_is_not_a_secret():
    """Forty hex characters, everywhere in a repo, and low entropy."""
    assert kinds('COMMIT = "b8b3525769e1fd0f4292287012f5790c056a46cc"') == []


def test_a_policy_version_hash_is_not_a_secret():
    assert kinds('policy_version = "local:5afcbb39121f"') == []


def test_a_long_ordinary_sentence_is_not_a_secret():
    assert kinds('reason = "This is a generic marketing offer for credit cards"') == []


def test_an_explicit_allow_marker_is_honoured():
    """The scanner's own tests have to contain realistic fakes. So does any
    fixture. An inline marker is auditable in review; a path exclusion is a
    hole nobody looks at again."""
    line = f'tok = "1234567890:AAF3kQ2mNpXrLvY7wZ1bCdEfGhIjKlMnOpQ"  {ALLOW}'  # secret-scan: allow
    assert kinds(line) == []


# --- the finding itself ------------------------------------------------------

def test_a_finding_names_the_line_and_the_rule():
    """The message has to be actionable at the moment the commit is refused."""
    text = "clean = 1\nAPI_KEY = 'drZ5QweOSV7hFYoNqCtBBwbgW4PjVHjuihSoLzRY'\n"  # secret-scan: allow
    finding = scan_text(text, "notes.py")[0]
    assert finding.line_no == 2
    assert finding.path == "notes.py"
    assert "API_KEY" in finding.excerpt


def test_the_excerpt_does_not_reproduce_the_whole_secret():
    """A refused commit message can end up pasted into a chat or an issue.
    Printing the key there would spread it further than the commit would have."""
    secret = "drZ5QweOSV7hFYoNqCtBBwbgW4PjVHjuihSoLzRY"  # secret-scan: allow
    finding = scan_text(f'API_KEY = "{secret}"', "x.py")[0]
    assert secret not in finding.excerpt
    assert "…" in finding.excerpt or "..." in finding.excerpt
