"""The HTTP transport.

Found the hard way: every Bot-level test passed against a fake transport, and
the first real message to Telegram returned 400. The fake accepted a payload the
API rejects. These tests cover the wire format itself, no network required.
"""
import json
import pytest

from inbox_agent.telegram.bot import HttpTransport, TelegramError

# Obviously fake, but the right SHAPE - Telegram tokens are <digits>:<35 chars>,
# and the redaction tests need something token-like to prove it gets replaced.
# The first version of this file used the real bot token, in the very commit
# that fixed the token leaking into httpx logs. A test fixture is a committed
# file; a live credential must never be one.
TOKEN = "0000000000:" + "A" * 35


def test_none_values_are_omitted_not_sent_as_null():
    """Telegram answers `400 object expected as reply markup` to
    {"reply_markup": null}. The key must be absent, not null."""
    payload = HttpTransport._clean({"chat_id": 1, "text": "hi", "reply_markup": None})
    assert "reply_markup" not in payload
    assert payload == {"chat_id": 1, "text": "hi"}


def test_a_real_keyboard_survives_cleaning():
    kb = {"inline_keyboard": [[{"text": "OK", "callback_data": "a:0"}]]}
    payload = HttpTransport._clean({"chat_id": 1, "text": "hi", "reply_markup": kb})
    assert payload["reply_markup"] == kb


def test_markup_is_none_for_an_empty_keyboard():
    assert HttpTransport._markup([]) is None
    assert HttpTransport._markup(None) is None


def test_markup_shape_matches_the_bot_api():
    kb = HttpTransport._markup([[("OK", "a:0"), ("Skip", "r:0")]])
    assert kb == {"inline_keyboard": [[
        {"text": "OK", "callback_data": "a:0"},
        {"text": "Skip", "callback_data": "r:0"}]]}


# --- the token must never reach a log or an exception ------------------------
# httpx logs the full request URL at INFO, and the bot token is IN that URL.
# HTTPStatusError embeds it too. Either one writes a live credential into logs
# and tracebacks in plaintext.

def test_error_message_never_contains_the_token():
    msg = HttpTransport._redact(
        f"Client error '400 Bad Request' for url "
        f"'https://api.telegram.org/bot{TOKEN}/sendMessage'", TOKEN)
    assert TOKEN not in msg
    assert "AAAAA" not in msg, "the token body survived redaction"
    assert "sendMessage" in msg, "redaction should keep the useful part"


def test_redaction_handles_a_message_with_no_token_in_it():
    assert HttpTransport._redact("plain failure", TOKEN) == "plain failure"


def test_telegram_error_carries_the_api_description_not_the_url():
    err = TelegramError("sendMessage", 400, "object expected as reply markup")
    text = str(err)
    assert "sendMessage" in text
    assert "object expected as reply markup" in text
    assert "api.telegram.org" not in text


def test_constructing_the_transport_silences_httpx_url_logging():
    """httpx at INFO prints the full URL, token included, on every call."""
    import logging
    HttpTransport(TOKEN)
    assert logging.getLogger("httpx").level >= logging.WARNING
