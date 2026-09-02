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


# --- drift between the two copies --------------------------------------------
# policies/default.md and the hub's POLICY.md are two copies of one text, and
# the local one cannot be deleted: four tests above assert on its content with
# allow_remote=False, and the notebook must run with no key and no network. So
# the duplication is structural and the only question is whether drift can go
# unnoticed - which is precisely how the 404 survived, silently, for weeks.
#
# The pull is the one moment both texts are in hand, so the check costs a file
# read and no network. The hub version still wins: it is what the audit record
# names, and a forgotten push must never take a triage run down.

def _fake_client(text, monkeypatch, commit="abc123"):
    class Ctx:
        files = {"POLICY.md": text}
        commit_hash = commit

    class FakeClient:
        def pull_skill(self, identifier, *, version=None):
            return Ctx()

    monkeypatch.setattr("langsmith.Client", lambda *a, **kw: FakeClient())
    monkeypatch.setenv("LANGSMITH_API_KEY", "lsv2_pt_test")


def test_matching_copies_do_not_report_drift(tmp_path, monkeypatch):
    from inbox_agent.policy import LOCAL_POLICY
    _fake_client(LOCAL_POLICY.read_text(encoding="utf-8"), monkeypatch)
    p = load_policy(replace(make_settings(tmp_path), context_hub_tag=""))
    assert p.source == "context_hub"
    assert p.drifted is False


def test_drift_is_flagged_and_named(tmp_path, monkeypatch, capsys):
    _fake_client("a policy that is not the committed one", monkeypatch)
    p = load_policy(replace(make_settings(tmp_path), context_hub_tag=""))
    assert p.drifted is True, "a hub policy differing from the committed file went unreported"
    out = capsys.readouterr().out
    assert "drift" in out.lower()
    assert "default.md" in out


def test_drift_does_not_stop_the_run_and_the_hub_still_wins(tmp_path, monkeypatch):
    """A forgotten push must not take a triage run down, and the version stamp
    has to keep naming what actually ran."""
    _fake_client("a policy that is not the committed one", monkeypatch, commit="deadbeef")
    p = load_policy(replace(make_settings(tmp_path), context_hub_tag=""))
    assert p.text == "a policy that is not the committed one"
    assert p.version == "hub:deadbeef"


def test_a_local_only_policy_is_never_marked_drifted(tmp_path):
    """Nothing to compare against: with no hub in play there are not two copies."""
    p = load_policy(make_settings(tmp_path), allow_remote=False)
    assert p.source == "local"
    assert p.drifted is False
