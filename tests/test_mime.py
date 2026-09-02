"""Body extraction from a Gmail message payload.

Pure functions over the dicts the API returns - no network, no service object.
"""
import base64

from inbox_agent.gmail import _extract_body


def b64(text: str) -> str:
    """Gmail uses base64url, and strips padding in practice."""
    return base64.urlsafe_b64encode(text.encode()).decode().rstrip("=")


def test_simple_plain_text_body():
    payload = {"mimeType": "text/plain", "body": {"data": b64("hello world")}}
    assert _extract_body(payload) == "hello world"


def test_multipart_alternative_prefers_plain_text_over_html():
    """Both parts say the same thing; the plain one costs fewer tokens and
    needs no tag stripping."""
    payload = {
        "mimeType": "multipart/alternative",
        "parts": [
            {"mimeType": "text/plain", "body": {"data": b64("plain version")}},
            {"mimeType": "text/html",
             "body": {"data": b64("<p>html version</p>")}},
        ],
    }
    assert _extract_body(payload) == "plain version"


def test_html_only_body_is_stripped_of_tags():
    payload = {
        "mimeType": "multipart/alternative",
        "parts": [{"mimeType": "text/html",
                   "body": {"data": b64("<p>Hello <b>there</b></p>")}}],
    }
    got = _extract_body(payload)
    assert "<p>" not in got and "<b>" not in got
    assert "Hello" in got and "there" in got


def test_html_script_and_style_content_is_dropped_not_just_untagged():
    """Stripping only the tags would leave CSS and JS in the prompt - pure
    token cost, and a confusing thing to hand a classifier."""
    html = "<style>.a{color:red}</style><script>alert(1)</script><p>Real</p>"
    payload = {"mimeType": "text/html", "body": {"data": b64(html)}}
    got = _extract_body(payload)
    assert "color:red" not in got
    assert "alert" not in got
    assert "Real" in got


def test_nested_multipart_is_walked():
    """multipart/mixed wrapping multipart/alternative is the common shape for
    a newsletter with an attachment."""
    payload = {
        "mimeType": "multipart/mixed",
        "parts": [
            {"mimeType": "multipart/alternative",
             "parts": [
                 {"mimeType": "text/plain", "body": {"data": b64("buried")}},
             ]},
            {"mimeType": "application/pdf", "filename": "x.pdf",
             "body": {"attachmentId": "a1"}},
        ],
    }
    assert _extract_body(payload) == "buried"


def test_base64url_padding_is_restored():
    """Gmail strips '=' padding. Decoding without restoring it raises
    binascii.Error on roughly three quarters of all messages."""
    for text in ("a", "ab", "abc", "abcd"):
        payload = {"mimeType": "text/plain", "body": {"data": b64(text)}}
        assert _extract_body(payload) == text


def test_base64url_alphabet_is_handled():
    """'-' and '_' replace '+' and '/'. Standard b64decode rejects them."""
    raw = b"\xfb\xff\xfe"
    data = base64.urlsafe_b64encode(raw).decode().rstrip("=")
    payload = {"mimeType": "text/plain", "body": {"data": data}}
    assert _extract_body(payload) == raw.decode("utf-8", errors="replace")


def test_missing_body_returns_empty_string_rather_than_raising():
    """A calendar invite or a bare attachment has no text part at all. That is
    ordinary mail, not an error - it must classify on subject and sender."""
    assert _extract_body({"mimeType": "text/plain", "body": {}}) == ""
    assert _extract_body({}) == ""
    assert _extract_body({"mimeType": "multipart/mixed", "parts": []}) == ""


def test_undecodable_bytes_do_not_raise():
    """A mislabelled charset must not take down a whole run."""
    data = base64.urlsafe_b64encode(b"\xff\xfe\x00bad").decode().rstrip("=")
    payload = {"mimeType": "text/plain", "body": {"data": data}}
    assert isinstance(_extract_body(payload), str)


def test_attachments_are_never_treated_as_the_body():
    """A part with a filename is an attachment even when its mimeType is
    text/plain - a .txt attachment must not become the body."""
    payload = {
        "mimeType": "multipart/mixed",
        "parts": [
            {"mimeType": "text/plain", "filename": "notes.txt",
             "body": {"data": b64("attachment content")}},
            {"mimeType": "text/plain", "body": {"data": b64("real body")}},
        ],
    }
    assert _extract_body(payload) == "real body"


# --- beyond the plan --------------------------------------------------------

def test_an_empty_plain_part_falls_through_to_the_html_part():
    """The defect this test exists for: deciding on the PRESENCE of a
    text/plain part rather than on whether it yielded any text.

    A multipart/alternative whose plain part is whitespace - or whose data is
    corrupt, which decodes to "" - would return "" and never look at the html
    part that carries the whole message. Every such mail would classify on
    subject and sender alone, silently, and look like mail that genuinely had
    no body.
    """
    payload = {
        "mimeType": "multipart/alternative",
        "parts": [
            {"mimeType": "text/plain", "body": {"data": b64("   \n  ")}},
            {"mimeType": "text/html",
             "body": {"data": b64("<p>the actual message</p>")}},
        ],
    }
    assert "the actual message" in _extract_body(payload)


def test_a_corrupt_plain_part_falls_through_to_the_html_part():
    payload = {
        "mimeType": "multipart/alternative",
        "parts": [
            {"mimeType": "text/plain", "body": {"data": "!!!not base64!!!"}},
            {"mimeType": "text/html", "body": {"data": b64("<p>rescued</p>")}},
        ],
    }
    assert "rescued" in _extract_body(payload)


def test_block_elements_become_line_breaks():
    """Without this the whole mail collapses to one line, so the 4000-char
    truncation in _fence cuts mid-sentence with no structure to orient on."""
    html = "<p>First para</p><p>Second para</p><br>Third"
    payload = {"mimeType": "text/html", "body": {"data": b64(html)}}
    got = _extract_body(payload)
    assert "First para" in got and "Second para" in got and "Third" in got
    assert "\n" in got, got
    assert "First paraSecond" not in got


def test_charset_declared_mimetype_is_still_recognised():
    """Gmail reports 'text/plain; charset="UTF-8"', not a bare 'text/plain'."""
    payload = {"mimeType": 'text/plain; charset="UTF-8"',
               "body": {"data": b64("charset suffixed")}}
    assert _extract_body(payload) == "charset suffixed"


def test_html_entities_are_unescaped():
    payload = {"mimeType": "text/html",
               "body": {"data": b64("<p>Tom &amp; Jerry &lt;3 &nbsp;stuff</p>")}}
    got = _extract_body(payload)
    assert "Tom & Jerry" in got
    assert "&amp;" not in got


def test_zero_width_marketing_filler_is_dropped():
    """Measured on real mail, not hypothesised: a Strava message was 14.3%
    zero-width filler overall and 22.4% across its first 4000 characters -
    the exact slice _fence hands the model. The padding pads the inbox preview
    line and means nothing, but it is charged for."""
    padded = "Real content" + ("‌ " * 200) + " and more"
    payload = {"mimeType": "text/plain", "body": {"data": b64(padded)}}
    got = _extract_body(payload)
    assert "‌" not in got
    assert "Real content" in got and "and more" in got
    assert len(got) < 40, f"filler survived: {len(got)} chars"


def test_nbsp_becomes_a_space_rather_than_vanishing():
    """NBSP is a space and means something, unlike the zero-width class.
    Deleting it would run words together."""
    payload = {"mimeType": "text/plain",
               "body": {"data": b64("Dear Investor,  Greetings")}}
    assert _extract_body(payload) == "Dear Investor, Greetings"


def test_filler_is_stripped_from_html_parts_too():
    html = "<p>Hello" + ("‌" * 100) + "world</p>"
    payload = {"mimeType": "text/html", "body": {"data": b64(html)}}
    got = _extract_body(payload)
    assert "‌" not in got
    assert "Helloworld" in got


def test_combining_grapheme_joiner_filler_is_dropped():
    """Found by counting what survived the first pass on real mail: one Strava
    message carried 448 of these. The set is empirical, not guessed - bulk
    senders use whatever their ESP offers."""
    payload = {"mimeType": "text/plain",
               "body": {"data": b64("Strava" + ("\u034f" * 300) + "news")}}
    got = _extract_body(payload)
    assert "\u034f" not in got
    assert "Strava" in got and "news" in got
    assert len(got) < 30, f"filler survived: {len(got)}"
