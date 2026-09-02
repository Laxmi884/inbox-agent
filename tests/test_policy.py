from dataclasses import replace

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


# --- which version gets pulled ----------------------------------------------
# CONTEXT_HUB_TAG shipped as "dev", and "dev" is not a ref Context Hub can
# resolve - only a commit hash, or nothing at all for the latest. So the pull
# 404'd on every run and load_policy silently fell back to the local file. That
# fallback is the right behaviour and it is exactly what hid the bug: the agent
# reported `local:...` for weeks while appearing to be configured for the hub.
#
# Blank now means latest. Reproducibility does not depend on pinning here,
# because the audit record stores the resolved `hub:<commit>` of whatever
# actually ran - pinning is only for forcing an OLD policy deliberately.

class _FakeCtx:
    files = {"POLICY.md": "remote policy text"}
    commit_hash = "13ac11f1deadbeef"


def test_a_blank_tag_pulls_the_latest_rather_than_a_ref_named_empty(tmp_path, monkeypatch):
    seen = {}

    class FakeClient:
        def pull_skill(self, identifier, *, version=None):
            seen["identifier"], seen["version"] = identifier, version
            return _FakeCtx()

    monkeypatch.setattr("langsmith.Client", lambda *a, **kw: FakeClient())
    monkeypatch.setenv("LANGSMITH_API_KEY", "lsv2_pt_test")
    s = make_settings(tmp_path)
    p = load_policy(replace(s, context_hub_tag=""), allow_remote=True)
    assert seen["version"] is None, f"blank tag must mean latest, got {seen['version']!r}"
    assert p.source == "context_hub"
    assert p.version == "hub:13ac11f1deadbeef"


def test_an_explicit_tag_is_still_passed_through_for_pinning(tmp_path, monkeypatch):
    seen = {}

    class FakeClient:
        def pull_skill(self, identifier, *, version=None):
            seen["version"] = version
            return _FakeCtx()

    monkeypatch.setattr("langsmith.Client", lambda *a, **kw: FakeClient())
    monkeypatch.setenv("LANGSMITH_API_KEY", "lsv2_pt_test")
    s = make_settings(tmp_path)
    load_policy(replace(s, context_hub_tag="13ac11f1"), allow_remote=True)
    assert seen["version"] == "13ac11f1"


def test_the_resolved_commit_is_recorded_not_the_tag(tmp_path, monkeypatch):
    """An audit record must name the policy that actually ran, so a run pinned
    to nothing is still reproducible after the fact."""
    class FakeClient:
        def pull_skill(self, identifier, *, version=None):
            return _FakeCtx()

    monkeypatch.setattr("langsmith.Client", lambda *a, **kw: FakeClient())
    monkeypatch.setenv("LANGSMITH_API_KEY", "lsv2_pt_test")
    p = load_policy(replace(make_settings(tmp_path), context_hub_tag=""),
                    allow_remote=True)
    assert p.version == "hub:13ac11f1deadbeef"
    assert "dev" not in p.version
