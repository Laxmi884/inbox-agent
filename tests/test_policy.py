from inbox_agent.config import ALWAYS_FORBIDDEN, Settings
from inbox_agent.policy import Policy, load_policy


def make_settings(tmp_path) -> Settings:
    return Settings(backend="offline", dry_run=True, snapshot_dir=tmp_path,
                    snapshot_size=50, audit_log=tmp_path / "audit.jsonl",
                    forbidden_actions=ALWAYS_FORBIDDEN,
                    context_hub_skill="inbox-triage", context_hub_tag="dev")


def test_local_policy_loads_without_network(tmp_path):
    """The notebook must work with no LangSmith key and no internet."""
    p = load_policy(make_settings(tmp_path), allow_remote=False)
    assert isinstance(p, Policy)
    assert p.source == "local"
    assert len(p.text) > 0


def test_local_policy_version_is_content_addressed(tmp_path):
    """Same policy text must always produce the same version stamp, so an audit
    record from March can be matched to the exact policy that produced it."""
    a = load_policy(make_settings(tmp_path), allow_remote=False)
    b = load_policy(make_settings(tmp_path), allow_remote=False)
    assert a.version == b.version
    assert a.version.startswith("local:")


def test_policy_names_the_forbidden_actions(tmp_path):
    """Belt and braces: the deny-list is enforced in code, and stated in the prompt."""
    text = load_policy(make_settings(tmp_path), allow_remote=False).text.lower()
    assert "send" in text
    assert "permanent" in text and "deletion" in text


def test_policy_lists_all_categories(tmp_path):
    """A future edit must not silently drop a category from the prompt."""
    text = load_policy(make_settings(tmp_path), allow_remote=False).text
    categories = [
        "needs_reply", "important_fyi", "newsletter_valuable", "newsletter_noise",
        "promotion", "receipt", "recruiter", "security_alert", "automated", "other",
    ]
    for category in categories:
        assert f"`{category}`" in text


def test_remote_failure_falls_back_to_local(tmp_path, monkeypatch):
    def boom(*a, **kw):
        raise RuntimeError("no network")
    monkeypatch.setattr("inbox_agent.policy._pull_from_context_hub", boom)
    p = load_policy(make_settings(tmp_path), allow_remote=True)
    assert p.source == "local"
