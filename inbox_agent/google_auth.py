"""OAuth credentials for the Gmail API. Knows nothing about mail.

Split from gmail.py deliberately: the client should be constructible from any
Credentials object, and this module should be replaceable - a service account,
a different token store - without the client noticing.
"""
from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

# The ONE scope. gmail.modify grants read, label, archive, trash and
# draft-create - every action on the autonomy ladder - and grants neither send
# (gmail.send) nor permanent delete (https://mail.google.com/).
#
# ALWAYS_FORBIDDEN (config.py:28) is exactly {send_message, delete_forever}, so
# the scope boundary and the deny-list coincide. That puts the two forbidden
# actions beyond reach at Google's edge and not only at audit.py's chokepoint,
# which is a strictly stronger guarantee than our own code can offer. A wider
# scope would save work later and is refused for precisely that reason.
GMAIL_MODIFY_SCOPE = "https://www.googleapis.com/auth/gmail.modify"


class ConsentExpiredError(RuntimeError):
    """The stored refresh token no longer works and consent must be redone.

    Its own type because the remedy is specific and the cause is invisible:
    Google returns `invalid_grant` for a revoked token, for a token whose scopes
    changed, and for a password reset, and says nothing about which.
    """


# Written against the console as it exists now. Google moved these settings out
# of "APIs & Services -> OAuth consent screen" into "Google Auth Platform", so
# an error naming the old path sends the reader to a page that is not there any
# more - worse than naming no path at all.
_SETUP_HELP = """\
No OAuth client at {path}.

Create one once, at console.cloud.google.com:

  1. Create a project, and make sure it is the selected one.
  2. APIs & Services -> Library -> enable the **Gmail API**.
  3. Google Auth Platform -> Branding: app name, support email, contact email.
  4. Google Auth Platform -> Audience -> External.

     Then EITHER publish the app (Publishing status -> Publish app), OR add
     yourself under **Test users**.

     Publishing is the one worth doing. An External app left in "Testing" has
     its refresh token revoked by Google after exactly seven days, so a
     scheduled run stops dead once a week and demands a new consent.
     Publishing also requires a homepage URL and a privacy policy URL, which a
     personal script may not have - hence the Test users fallback, and hence
     ConsentExpiredError below, which exists to explain the weekly failure when
     that fallback is what you are on.

  5. Google Auth Platform -> Clients -> Create Client -> **Desktop app**.
  6. Download the JSON and save it to {path}.

Then re-run. A browser opens once for consent and {token} is written; after
that it refreshes silently.\
"""

_EXPIRED_HELP = """\
Google refused to refresh the stored credential at {token}.

The usual cause, and near-certain if this happened about a week after the last
consent: the OAuth consent screen is still in **Testing**. Google revokes the
refresh token of an External app in Testing after exactly seven days. The error
it returns for this - invalid_grant - is the same one it returns for a revoked
grant and for a password reset, so it names no cause on its own.

To fix it permanently: console.cloud.google.com -> Google Auth Platform ->
Audience -> Publish app. That needs an app name, support email, homepage URL
and privacy policy URL.

To carry on for another seven days: delete {token} and re-run, which opens the
browser for a fresh consent.

Underlying error: {cause}\
"""


def _creds_from_file(token_path: Path, scopes: list[str]):
    """Indirection so tests can inject credentials without the google libs."""
    from google.oauth2.credentials import Credentials

    return Credentials.from_authorized_user_file(str(token_path), scopes)


def _run_consent_flow(*, client_secrets_path: Path, scopes: list[str]):
    """Opens the browser once. port=0 lets the OS pick a free loopback port, so
    a stale redirect URI on a fixed port cannot wedge the flow."""
    from google_auth_oauthlib.flow import InstalledAppFlow

    flow = InstalledAppFlow.from_client_secrets_file(
        str(client_secrets_path), scopes)
    return flow.run_local_server(port=0)


def _write_token(token_path: Path, creds) -> None:
    """Owner-only, with the parent directory created if absent.

    Written via os.open with mode 0o600 rather than write_text-then-chmod: the
    latter leaves a window in which a live refresh token for the real mailbox
    is world-readable.
    """
    token_path.parent.mkdir(parents=True, exist_ok=True)
    fd = os.open(token_path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    with os.fdopen(fd, "w") as fh:
        fh.write(creds.to_json())


def consent_sidecar(token_path: Path) -> Path:
    """Where the consent date lives: beside the token, never inside it.

    token.json's schema is google-auth's - creds.to_json() writes it and
    Credentials.from_authorized_user_file reads it - so an extra key there is a
    library upgrade away from breaking authentication for a diagnostic.
    """
    return Path(token_path).with_suffix(".consent.json")


def record_consent(token_path: Path, *, now: Optional[datetime] = None) -> None:
    """Stamp the moment consent was granted.

    Called ONLY from the consent path, never from the refresh path. Google's
    seven-day revocation for an app in Testing runs from consent and is not
    reset by refreshing, so stamping on refresh would promise six more days on
    the morning the token dies - worse than tracking nothing.
    """
    when = now or datetime.now(timezone.utc)
    path = consent_sidecar(token_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"consented_at": when.isoformat()}), encoding="utf-8")


def consented_at(token_path: Path) -> Optional[datetime]:
    """When consent was granted, or None if unrecorded or unreadable.

    None for every token issued before this existed. Diagnostics only - never
    read as authorisation - so a missing or corrupt file degrades to "unknown"
    rather than raising in a caller that is trying to explain what is wrong.
    """
    try:
        raw = json.loads(consent_sidecar(token_path).read_text(encoding="utf-8"))
        return datetime.fromisoformat(raw["consented_at"])
    except Exception:
        return None


def get_credentials(*, client_secrets_path: Path, token_path: Path,
                    scopes: Optional[list[str]] = None):
    """Load, refresh, or obtain credentials - in that order of preference.

    Consent is the last resort, not the first: it needs a human and a browser,
    and a run that silently re-prompts every time is a run that cannot be
    scheduled.
    """
    scopes = scopes or [GMAIL_MODIFY_SCOPE]
    client_secrets_path = Path(client_secrets_path)
    token_path = Path(token_path)

    creds = None
    if token_path.exists():
        creds = _creds_from_file(token_path, scopes)

    if creds is not None and creds.valid:
        return creds

    if creds is not None and creds.expired and creds.refresh_token:
        from google.auth.exceptions import RefreshError
        from google.auth.transport.requests import Request

        try:
            creds.refresh(Request())
        except RefreshError as exc:
            # Deliberately NOT falling through to the consent flow. A digest
            # running at 08:00 with nobody present would hang forever on a
            # browser prompt no one will click, which reads as a hang rather
            # than an expired credential. The dead token is left on disk so a
            # re-run reproduces this same named error instead of a different
            # one.
            raise ConsentExpiredError(
                _EXPIRED_HELP.format(token=token_path, cause=exc)) from exc
        _write_token(token_path, creds)
        return creds

    # Consent is the only path left, and it is the only one needing the client
    # secret. Checked here rather than at the top: a valid token makes the
    # secret file irrelevant, so failing early on its absence would be a false
    # alarm on a machine that is working perfectly well.
    if not client_secrets_path.exists():
        raise FileNotFoundError(
            _SETUP_HELP.format(path=client_secrets_path, token=token_path))

    creds = _run_consent_flow(client_secrets_path=client_secrets_path,
                              scopes=scopes)
    _write_token(token_path, creds)
    # Consent path only. The refresh path above deliberately does not touch
    # this - see record_consent.
    record_consent(token_path)
    return creds


def authorized_http(credentials):
    """A FRESH transport bound to `credentials`.

    Called once per worker thread. httplib2.Http is not thread-safe and
    googleapiclient's service object holds exactly one, so a hydration pool
    sharing it corrupts SSL socket state - see LiveGmailClient._http, and the
    real-mailbox reproduction recorded there.
    """
    import google_auth_httplib2
    import httplib2

    return google_auth_httplib2.AuthorizedHttp(credentials, http=httplib2.Http())
