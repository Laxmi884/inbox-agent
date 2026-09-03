"""google_auth.py knows about OAuth and nothing about mail.

Every test here runs with no network and no real credentials: the flow and the
Credentials class are both injected, so the only thing under test is our own
load / refresh / consent decision.
"""
import json
import pytest

from inbox_agent.google_auth import (
    GMAIL_MODIFY_SCOPE, ConsentExpiredError, get_credentials,
)


def _stub_secrets(tmp_path):
    p = tmp_path / "credentials.json"
    p.write_text("{}")
    return p


def test_scope_is_modify_and_nothing_wider():
    """gmail.modify grants label, archive, trash and draft-create, and grants
    neither send nor permanent delete - so ALWAYS_FORBIDDEN is enforced at
    Google's edge, not only at audit.py's chokepoint. A wider scope would be
    less work later and is refused for exactly that reason."""
    assert GMAIL_MODIFY_SCOPE == "https://www.googleapis.com/auth/gmail.modify"


def test_missing_client_secret_names_the_console_steps(tmp_path):
    """Same actionable-error style as load_snapshot (gmail.py:61).

    Names the CURRENT console UI. Google moved these settings out of
    "APIs & Services -> OAuth consent screen" into "Google Auth Platform", so
    an error naming the old path sends the reader to a page that no longer
    exists - which is worse than naming no path at all.
    """
    with pytest.raises(FileNotFoundError) as exc:
        get_credentials(client_secrets_path=tmp_path / "absent.json",
                        token_path=tmp_path / "token.json")
    msg = str(exc.value)
    assert "Gmail API" in msg
    assert "Desktop app" in msg
    assert "Google Auth Platform" in msg
    assert "Test users" in msg      # the fallback when publishing is blocked
    assert "seven days" in msg      # why publishing matters


def test_valid_token_is_reused_without_a_consent_flow(tmp_path, monkeypatch):
    """The whole point of persisting a token: one browser consent, ever."""
    token = tmp_path / "token.json"
    token.write_text(json.dumps({"token": "stub"}))

    class FakeCreds:
        valid = True
        expired = False
        refresh_token = None

    def no_flow(*a, **k):
        raise AssertionError("consent flow must not run for a valid token")

    monkeypatch.setattr("inbox_agent.google_auth._creds_from_file",
                        lambda *a, **k: FakeCreds())
    monkeypatch.setattr("inbox_agent.google_auth._run_consent_flow", no_flow)

    assert isinstance(get_credentials(client_secrets_path=_stub_secrets(tmp_path),
                                      token_path=token), FakeCreds)


def test_expired_token_with_a_refresh_token_refreshes_instead_of_reconsenting(
        tmp_path, monkeypatch):
    token = tmp_path / "token.json"
    token.write_text(json.dumps({"token": "stub"}))
    refreshed = []

    class FakeCreds:
        valid = False
        expired = True
        refresh_token = "r"

        def refresh(self, request):
            refreshed.append(request)
            self.valid = True

        def to_json(self):
            return '{"token": "refreshed"}'

    def no_flow(*a, **k):
        raise AssertionError("consent flow must not run when a refresh works")

    monkeypatch.setattr("inbox_agent.google_auth._creds_from_file",
                        lambda *a, **k: FakeCreds())
    monkeypatch.setattr("inbox_agent.google_auth._run_consent_flow", no_flow)

    creds = get_credentials(client_secrets_path=_stub_secrets(tmp_path),
                            token_path=token)
    assert creds.valid is True
    assert len(refreshed) == 1
    assert json.loads(token.read_text())["token"] == "refreshed"


def test_no_token_runs_the_consent_flow_and_writes_the_token(tmp_path, monkeypatch):
    token = tmp_path / "nested" / "token.json"

    class FakeCreds:
        valid = True
        expired = False
        refresh_token = "r"

        def to_json(self):
            return '{"token": "fresh"}'

    monkeypatch.setattr("inbox_agent.google_auth._run_consent_flow",
                        lambda **k: FakeCreds())

    get_credentials(client_secrets_path=_stub_secrets(tmp_path), token_path=token)
    assert json.loads(token.read_text())["token"] == "fresh"


def test_token_is_written_with_owner_only_permissions(tmp_path, monkeypatch):
    """token.json is a live credential for the real mailbox."""
    token = tmp_path / "token.json"

    class FakeCreds:
        valid = True
        expired = False
        refresh_token = "r"

        def to_json(self):
            return '{"token": "fresh"}'

    monkeypatch.setattr("inbox_agent.google_auth._run_consent_flow",
                        lambda **k: FakeCreds())

    get_credentials(client_secrets_path=_stub_secrets(tmp_path), token_path=token)
    assert (token.stat().st_mode & 0o077) == 0


# --- the Testing-mode expiry ------------------------------------------------
# Not in the plan. Added because this project's own consent screen is in
# Testing: publishing to production requires a homepage URL and a privacy
# policy URL, which a personal script does not have, so the refresh token is
# revoked by Google after exactly seven days. That is a certainty here, not a
# hypothetical, and the error Google returns for it - invalid_grant - says
# nothing about the cause.

def test_a_revoked_refresh_token_explains_the_testing_mode_cause(
        tmp_path, monkeypatch):
    """Raise, rather than silently reopening a browser.

    A scheduled digest running at 08:00 with nobody present must fail loudly
    with something actionable. Falling through to consent would leave it
    hanging forever on a browser prompt no one will ever click, which reads as
    a hang rather than an expired credential.
    """
    from google.auth.exceptions import RefreshError

    token = tmp_path / "token.json"
    token.write_text(json.dumps({"token": "stub"}))

    class FakeCreds:
        valid = False
        expired = True
        refresh_token = "r"

        def refresh(self, request):
            raise RefreshError("invalid_grant: Token has been expired or revoked.")

    def no_flow(*a, **k):
        raise AssertionError("must not silently reopen a browser")

    monkeypatch.setattr("inbox_agent.google_auth._creds_from_file",
                        lambda *a, **k: FakeCreds())
    monkeypatch.setattr("inbox_agent.google_auth._run_consent_flow", no_flow)

    with pytest.raises(ConsentExpiredError) as exc:
        get_credentials(client_secrets_path=_stub_secrets(tmp_path),
                        token_path=token)
    msg = str(exc.value)
    assert "seven days" in msg
    assert "Testing" in msg
    assert str(token) in msg          # the file to delete to re-consent


def test_the_dead_token_is_not_left_looking_valid(tmp_path, monkeypatch):
    """The revoked token stays on disk untouched, so re-running reproduces the
    same named error rather than a different one. Deleting it here would turn
    the next scheduled run into a silent browser prompt - exactly what the
    raise above exists to prevent."""
    from google.auth.exceptions import RefreshError

    token = tmp_path / "token.json"
    token.write_text(json.dumps({"token": "stub"}))

    class FakeCreds:
        valid = False
        expired = True
        refresh_token = "r"

        def refresh(self, request):
            raise RefreshError("invalid_grant")

    monkeypatch.setattr("inbox_agent.google_auth._creds_from_file",
                        lambda *a, **k: FakeCreds())
    monkeypatch.setattr("inbox_agent.google_auth._run_consent_flow",
                        lambda **k: (_ for _ in ()).throw(AssertionError()))

    with pytest.raises(ConsentExpiredError):
        get_credentials(client_secrets_path=_stub_secrets(tmp_path),
                        token_path=token)
    assert json.loads(token.read_text())["token"] == "stub"


# --- consent date sidecar ---------------------------------------------------

def test_consent_is_recorded_in_a_sidecar_not_in_the_token(tmp_path):
    """A sidecar because token.json's schema belongs to google-auth: it is
    produced by creds.to_json() and consumed by from_authorized_user_file, so a
    foreign key invites a breakage on upgrade for no benefit."""
    from datetime import datetime, timezone
    from pathlib import Path
    from inbox_agent import google_auth

    token = tmp_path / "token.json"
    token.write_text('{"refresh_token": "x"}')

    google_auth.record_consent(token, now=datetime(2026, 9, 1, tzinfo=timezone.utc))

    assert google_auth.consent_sidecar(token) == tmp_path / "token.consent.json"
    assert '"refresh_token": "x"' in token.read_text(), "token.json must be untouched"
    assert google_auth.consented_at(token) == datetime(2026, 9, 1, tzinfo=timezone.utc)


def test_consent_date_is_unknown_when_no_sidecar_exists(tmp_path):
    """Every token issued before this feature. Returning None makes doctor say
    'unknown' rather than invent a date - a confident wrong prediction about
    when the mailbox stops working is worse than no prediction."""
    from inbox_agent import google_auth

    token = tmp_path / "token.json"
    token.write_text("{}")
    assert google_auth.consented_at(token) is None


def test_a_corrupt_sidecar_reads_as_unknown_rather_than_raising(tmp_path):
    """Doctor must survive a hand-edited or truncated sidecar. This file is
    diagnostics, never authorisation, so it can never be worth crashing over."""
    from inbox_agent import google_auth

    token = tmp_path / "token.json"
    token.write_text("{}")
    google_auth.consent_sidecar(token).write_text("not json{{")
    assert google_auth.consented_at(token) is None


def test_a_naive_timestamp_in_the_sidecar_reads_as_unknown_rather_than_raising(tmp_path):
    """record_consent always writes an aware, UTC timestamp; a naive one can
    only come from a hand-edit - the likeliest one being exactly this field.
    fromisoformat() parses a naive value without error, so without a tzinfo
    check this would return a naive datetime that doctor.oauth_check then
    subtracts from an aware datetime.now(timezone.utc), raising TypeError
    instead of doctor reporting anything at all. Treat it as unknown, the
    same as a corrupt or missing sidecar."""
    from inbox_agent import google_auth

    token = tmp_path / "token.json"
    token.write_text("{}")
    google_auth.consent_sidecar(token).write_text(
        '{"consented_at": "2026-09-01T00:00:00"}')
    assert google_auth.consented_at(token) is None


def test_sidecar_with_no_consented_at_key_reads_as_unknown(tmp_path):
    """A sidecar that is valid JSON but missing the consented_at key (hand-edited
    or half-written file) must not crash doctor; it reads as "unknown" instead."""
    from inbox_agent import google_auth

    token = tmp_path / "token.json"
    token.write_text("{}")
    google_auth.consent_sidecar(token).write_text('{"other_key": "value"}')
    assert google_auth.consented_at(token) is None


def test_refresh_path_does_not_create_a_sidecar(tmp_path, monkeypatch):
    """The refresh path must never stamp the sidecar - Google's 7-day revocation
    runs from consent and is not reset by refreshing, so stamping on refresh would
    promise six more days on the morning the token dies. This test guards against
    accidental placement of record_consent on the refresh branch."""
    from inbox_agent import google_auth

    token = tmp_path / "token.json"
    token.write_text(json.dumps({"token": "stub"}))

    class FakeCreds:
        valid = False
        expired = True
        refresh_token = "r"

        def refresh(self, request):
            self.valid = True

        def to_json(self):
            return '{"token": "refreshed"}'

    def no_flow(*a, **k):
        raise AssertionError("consent flow must not run on refresh")

    monkeypatch.setattr("inbox_agent.google_auth._creds_from_file",
                        lambda *a, **k: FakeCreds())
    monkeypatch.setattr("inbox_agent.google_auth._run_consent_flow", no_flow)

    get_credentials(client_secrets_path=_stub_secrets(tmp_path), token_path=token)
    # The refresh path must NOT create a sidecar
    assert not google_auth.consent_sidecar(token).exists(), \
        "Refresh path must not stamp consent sidecar"


def test_consent_path_creates_the_sidecar(tmp_path, monkeypatch):
    """The consent path must create the sidecar, and only the consent path.
    This test is the counterpart to test_refresh_path_does_not_create_a_sidecar
    and pins both directions: consent creates it, refresh does not."""
    token = tmp_path / "nested" / "token.json"

    class FakeCreds:
        valid = True
        expired = False
        refresh_token = "r"

        def to_json(self):
            return '{"token": "fresh"}'

    monkeypatch.setattr("inbox_agent.google_auth._run_consent_flow",
                        lambda **k: FakeCreds())

    from inbox_agent import google_auth
    get_credentials(client_secrets_path=_stub_secrets(tmp_path), token_path=token)
    # The consent path must create a sidecar
    assert google_auth.consent_sidecar(token).exists(), \
        "Consent path must create sidecar"
    assert google_auth.consented_at(token) is not None, \
        "Sidecar must contain a valid consented_at timestamp"
