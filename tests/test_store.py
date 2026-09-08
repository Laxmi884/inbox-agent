# tests/test_store.py
from datetime import datetime, timezone

from inbox_agent.models import ActionTemplate, Rule, Thread
from inbox_agent.store import (MIN_HITS_BEFORE_DEMOTION, PreferenceStore,
                               choose_scope,
                               build_store, rule_from_correction)


def thread(**kw) -> Thread:
    base = dict(id="t1", subject="Sale 50%", sender="deals@shop.com", to=[],
                date="2026-08-26T10:00:00Z", snippet="s", body="b", label_ids=["INBOX"])
    return Thread(**(base | kw))


def store() -> PreferenceStore:
    return PreferenceStore(build_store())  # no embeddings: exact matching only


def test_added_rule_is_retrievable():
    s = store()
    s.add_rule(rule_from_correction(thread(), [ActionTemplate(kind="archive")], "user archived it"))
    assert len(s.rules()) == 1


def test_rule_matches_thread_by_sender():
    s = store()
    s.add_rule(rule_from_correction(thread(), [ActionTemplate(kind="archive")], "user archived it"))
    assert len(s.matching(thread())) == 1


def test_rule_does_not_match_a_different_sender():
    s = store()
    s.add_rule(rule_from_correction(thread(), [ActionTemplate(kind="archive")], "note"))
    assert s.matching(thread(sender="boss@work.com")) == []


def test_provenance_survives_the_round_trip():
    s = store()
    s.add_rule(rule_from_correction(thread(), [ActionTemplate(kind="archive")], "rejected proposal on t1"))
    assert "rejected proposal on t1" in s.rules()[0].provenance


def test_record_hit_increments_the_counter():
    s = store()
    r = rule_from_correction(thread(), [ActionTemplate(kind="archive")], "note")
    s.add_rule(r)
    s.record_hit(r.id)
    s.record_hit(r.id)
    assert s.rules()[0].hit_count == 2


def test_mark_overridden_flags_the_rule_without_deleting_it():
    """An overridden rule is evidence. It stays visible in the audit trail."""
    s = store()
    r = rule_from_correction(thread(), [ActionTemplate(kind="archive")], "note")
    s.add_rule(r)
    s.mark_overridden(r.id)
    assert s.rules()[0].overridden is True
    assert len(s.rules()) == 1


def test_overridden_rules_stop_matching():
    s = store()
    r = rule_from_correction(thread(), [ActionTemplate(kind="archive")], "note")
    s.add_rule(r)
    s.mark_overridden(r.id)
    assert s.matching(thread()) == []


def test_delete_rule_removes_it():
    s = store()
    r = rule_from_correction(thread(), [ActionTemplate(kind="archive")], "note")
    s.add_rule(r)
    s.delete_rule(r.id)
    assert s.rules() == []


def test_as_table_is_human_readable():
    """The owner must always be able to read what the agent thinks it knows."""
    s = store()
    r = rule_from_correction(thread(), [ActionTemplate(kind="archive")], "note")
    s.add_rule(r)
    s.record_hit(r.id)
    row = s.as_table()[0]
    for col in ("id", "scope", "pattern", "action", "hit_count", "provenance"):
        assert col in row
    assert row["id"] == r.id
    assert row["scope"] == "sender"
    assert row["pattern"] == "deals@shop.com"
    assert row["action"] == "archive"
    assert row["hit_count"] == 1
    assert row["provenance"] == "note"
    assert row["overridden"] is False
    assert row.get("created_at")


def test_rules_returns_all_rules_beyond_the_default_search_limit():
    """BaseStore.search() defaults to limit=10. rules() must paginate past that
    default or the rule set silently truncates once it grows past 10 - exactly
    the silent-wrong-answer failure the audit design exists to prevent."""
    s = store()
    for i in range(25):
        s.add_rule(rule_from_correction(thread(sender=f"sender{i}@shop.com"), [ActionTemplate(kind="archive")], f"note {i}"))
    assert len(s.rules()) == 25


# --- category rules ---------------------------------------------------------
# A category is not a property of a thread, it is the model's conclusion about
# one, so these can never be matched by matching() - which runs before the model.

def category_rule(category="newsletter_valuable", rid="r-cat"):
    return Rule(id=rid, scope="category", pattern=category,
                actions=[ActionTemplate(kind="label", params={"label": category})],
                provenance="owner said keep it in the inbox",
                created_at=datetime.now(timezone.utc))


def test_a_category_rule_is_found_by_its_category():
    s = store()
    s.add_rule(category_rule())
    assert [r.id for r in s.matching_category("newsletter_valuable")] == ["r-cat"]


def test_a_category_rule_does_not_match_another_category():
    s = store()
    s.add_rule(category_rule())
    assert s.matching_category("promotion") == []


def test_a_category_rule_never_matches_a_thread():
    """It cannot be applied before the model has run, so it must not be found
    by the lookup that runs before the model."""
    s = store()
    s.add_rule(category_rule())
    assert s.matching(thread()) == []


def test_a_demoted_category_rule_stops_matching():
    """Same precision discipline as any other rule. A category rule reaches
    every thread of that category, so a bad one is worse here than anywhere."""
    s = store()
    r = s.add_rule(category_rule())
    for _ in range(4):
        s.record_hit(r.id)
        s.record_override(r.id)
    assert s.matching_category("newsletter_valuable") == []


# --- replacing a rule ---------------------------------------------------------
# Correcting the same sender twice produced TWO live rules with identical scope
# and pattern (r-b26fe722 and r-ba2727c6 against 'one8 <updates@one8.com>',
# two minutes apart, 2026-09-02). add_rule only ever stored; nothing reconciled
# a new rule against the one it replaces.
#
# `matching` takes max(created_at), so the newer one wins and the bug is
# invisible - until the newer one is demoted. Then the older takes over, and
# because a rule that never fires has hit_count 0 it can never itself be
# demoted, so the correction the owner replaced becomes permanent.
#
# Same failure as the stale held entry fixed in f2f9357, one layer down: the
# store only ever added, and nothing retired what a later verdict superseded.

ONE8 = "one8 <updates@one8.com>"


def _one8(**kw):
    return thread(sender=ONE8, subject="Order #81612525A confirmed", **kw)


def _receipt_only():
    return [ActionTemplate(kind="label", params={"label": "receipt"}),
            ActionTemplate(kind="unlabel", params={"label": "UNREAD"})]


def _receipt_and_archive():
    return [ActionTemplate(kind="label", params={"label": "receipt"}),
            ActionTemplate(kind="archive"),
            ActionTemplate(kind="unlabel", params={"label": "UNREAD"})]


def test_a_second_correction_of_one_sender_leaves_only_one_live_rule():
    s = store()
    s.add_rule(rule_from_correction(_one8(), _receipt_only(), "owner corrected"))
    s.add_rule(rule_from_correction(_one8(), _receipt_and_archive(), "owner corrected"))
    live = s.matching(_one8())
    assert len(live) == 1, f"two rules decide the same mail: {[r.id for r in live]}"
    assert [a.kind for a in live[0].actions] == ["label", "archive", "unlabel"]


def test_the_replaced_rule_is_retired_not_deleted():
    """mark_overridden's own reasoning: a rule the owner overruled is part of
    the record. It must stop deciding, not stop existing."""
    s = store()
    first = s.add_rule(rule_from_correction(_one8(), _receipt_only(), "owner corrected"))
    s.add_rule(rule_from_correction(_one8(), _receipt_and_archive(), "owner corrected"))
    stored = {r.id: r for r in s.rules()}
    assert first.id in stored, "the replaced rule was deleted"
    assert stored[first.id].overridden is True


def test_the_new_rule_names_what_it_replaced():
    """supersedes had no consumer until now. This is the first."""
    s = store()
    first = s.add_rule(rule_from_correction(_one8(), _receipt_only(), "owner corrected"))
    second = s.add_rule(rule_from_correction(_one8(), _receipt_and_archive(), "owner corrected"))
    assert second.supersedes == first.id


def test_a_retired_rule_does_not_come_back_when_its_replacement_is_demoted():
    """The whole point. A demoted replacement must not resurrect the correction
    the owner threw away - which a zero-hit rule would do permanently, being
    undemotable."""
    s = store()
    s.add_rule(rule_from_correction(_one8(), _receipt_only(), "owner corrected"))
    second = s.add_rule(rule_from_correction(_one8(), _receipt_and_archive(), "owner corrected"))
    for _ in range(MIN_HITS_BEFORE_DEMOTION):
        s.record_hit(second.id)
    for _ in range(MIN_HITS_BEFORE_DEMOTION):
        s.record_override(second.id)
    assert s.matching(_one8()) == [], "the replaced rule came back after a demotion"


def test_replacing_a_rule_does_not_dent_its_precision():
    """Retiring and penalising are different. The replaced rule never fired, so
    counting an override would record a disagreement with a decision that was
    never made - and precision divides by hit_count."""
    s = store()
    first = s.add_rule(rule_from_correction(_one8(), _receipt_only(), "owner corrected"))
    s.add_rule(rule_from_correction(_one8(), _receipt_and_archive(), "owner corrected"))
    stored = {r.id: r for r in s.rules()}[first.id]
    assert stored.override_count == 0
    assert stored.precision is None


def test_a_different_sender_is_untouched():
    s = store()
    other = s.add_rule(rule_from_correction(thread(), [ActionTemplate(kind="archive")], "n"))
    s.add_rule(rule_from_correction(_one8(), _receipt_only(), "owner corrected"))
    s.add_rule(rule_from_correction(_one8(), _receipt_and_archive(), "owner corrected"))
    assert {r.id for r in s.rules() if not r.overridden} >= {other.id}
    assert len(s.matching(thread())) == 1


def test_a_category_rule_never_retires_a_sender_rule():
    """Scope AND pattern. A category rule for `receipt` decides different mail
    from a sender rule that happens to apply the receipt label."""
    s = store()
    sender_rule = s.add_rule(rule_from_correction(_one8(), _receipt_only(), "owner corrected"))
    s.add_rule(Rule(id="r-cat", scope="category", pattern="receipt",
                    actions=[ActionTemplate(kind="archive")],
                    provenance="owner corrected every receipt",
                    created_at=datetime.now(timezone.utc)))
    assert len(s.matching(_one8())) == 1
    assert {r.id for r in s.rules() if not r.overridden} == {sender_rule.id, "r-cat"}


# --- sender patterns across the two Thread.sender formats ---------------------
# The two clients disagree about what Thread.sender holds. A snapshot thread
# carries a bare address, 'no-reply@p.simplywall.st'; a live thread carries the
# full From header, 'Simply Wall St <no-reply@p.simplywall.st>'. matching()
# compared pattern == thread.sender.lower(), so every rule taught against the
# snapshot was dead against the real mailbox - r-4d395834 sat there with 3 hits,
# none of them live, silently deciding nothing.
#
# Normalising BOTH sides to the bare address would fix that and break something
# worse. Four addresses in one 120-thread page send under multiple display
# names: newsletters-noreply@linkedin.com is Forbes AND S&P Global AND a real
# person, messaging-digest-noreply@linkedin.com is two different people. A rule
# taught on Forbes would start deciding a person's mail.
#
# So the pattern's own shape says how wide it reaches. No display name means the
# ADDRESS, whoever it claims to be - which is what every snapshot-taught rule
# meant, having never had one. A display name means that identity, exactly.

SWS_ADDR = "no-reply@p.simplywall.st"
SWS_LIVE = "Simply Wall St <no-reply@p.simplywall.st>"


def _sender_rule(pattern: str) -> Rule:
    return Rule(id=f"r-{pattern[:6]}", scope="sender", pattern=pattern.lower(),
                actions=[ActionTemplate(kind="trash")], provenance="owner corrected",
                created_at=datetime.now(timezone.utc))


def test_a_bare_address_rule_matches_a_live_full_header_sender():
    """The simplywall.st rule, taught against the snapshot, must decide live mail."""
    s = store()
    s.add_rule(_sender_rule(SWS_ADDR))
    assert len(s.matching(thread(sender=SWS_LIVE))) == 1


def test_a_bare_address_rule_still_matches_a_bare_sender():
    s = store()
    s.add_rule(_sender_rule(SWS_ADDR))
    assert len(s.matching(thread(sender=SWS_ADDR))) == 1


def test_a_bare_address_rule_reaches_every_display_name_on_that_address():
    """Deliberate: a pattern that names no identity cannot be asking for one."""
    s = store()
    s.add_rule(_sender_rule(SWS_ADDR))
    assert len(s.matching(thread(sender=f"Simply Wall St Weekly <{SWS_ADDR}>"))) == 1


def test_a_named_sender_rule_does_not_reach_another_name_on_one_address():
    """The guard this whole design exists for. newsletters-noreply@linkedin.com
    carries Forbes, S&P Global and a real person; a rule taught on one of them
    must never decide the others."""
    s = store()
    s.add_rule(_sender_rule("forbes via linkedin <newsletters-noreply@linkedin.com>"))
    assert s.matching(
        thread(sender="Harnoor Saluja via LinkedIn <newsletters-noreply@linkedin.com>")) == []


def test_a_named_sender_rule_still_matches_its_own_sender():
    s = store()
    s.add_rule(_sender_rule("forbes via linkedin <newsletters-noreply@linkedin.com>"))
    assert len(s.matching(
        thread(sender="Forbes via LinkedIn <newsletters-noreply@linkedin.com>"))) == 1


def test_a_bare_address_rule_does_not_match_a_different_address():
    s = store()
    s.add_rule(_sender_rule(SWS_ADDR))
    assert s.matching(thread(sender="Someone <no-reply@other.example>")) == []


def test_rules_written_with_an_index_read_back_without_one(tmp_path):
    """Spec section 3.4. `auto` degrading on a machine whose Ollama died makes
    this path reachable without anyone choosing it, so learned rules must
    survive it. A rule that becomes invisible is worse than one that errors:
    the agent would silently stop honouring a correction the owner made."""
    from datetime import datetime, timezone

    from langchain_core.embeddings import Embeddings

    from inbox_agent.store import PreferenceStore, open_store
    from inbox_agent.models import ActionTemplate, Rule

    class FakeEmbeddings(Embeddings):
        """Deterministic, offline. The suite must not need Ollama.

        MUST subclass Embeddings: langgraph's ensure_embeddings does
        `isinstance(embed, Embeddings)` and otherwise wraps the argument in
        EmbeddingsLambda, which expects a callable - so a duck-typed class with
        embed_documents/embed_query is not accepted and raises TypeError.
        """
        def embed_documents(self, texts):
            return [[0.1] * 768 for _ in texts]

        def embed_query(self, text):
            return [0.1] * 768

    path = tmp_path / "prefs.sqlite"

    indexed = PreferenceStore(open_store(path, FakeEmbeddings()))
    # created_at is REQUIRED on Rule (models.py:94, no default).
    indexed.add_rule(Rule(id="r-keepme", scope="sender", pattern="a@b.com",
                          actions=[ActionTemplate(kind="archive", params={})],
                          provenance="written while indexed",
                          created_at=datetime.now(timezone.utc)))
    assert len(indexed.rules()) == 1

    plain = PreferenceStore(open_store(path))          # embeddings=None
    ids = [r.id for r in plain.rules()]
    assert ids == ["r-keepme"], f"rule lost when the index went away: {ids}"


# --- when a display name is payload rather than an identity -------------------

def _seen(pairs):
    """A sender history: (address, display name) as the agent saw them."""
    s = store()
    for address, name in pairs:
        s.note_sender(f"{name} <{address}>")
    return s


def test_rule_scopes_to_the_address_when_every_mail_brings_a_new_name():
    """invitations@linkedin.com is a different human every single time.

    Taught on one invitation, a rule carrying the inviter's display name can
    never match a second one - it is scoped to a person who will not write
    again. Observed live: r-d33ffdc7, taught 2026-09-08 15:22 EDT, hit_count 0
    while the 19:00 run trashed nothing.
    """
    s = _seen([("invitations@linkedin.com", "charmain guia"),
               ("invitations@linkedin.com", "daphna cibulski-cohen")])
    scope, pattern = choose_scope(
        thread(sender="Daphna Cibulski-Cohen <invitations@linkedin.com>",
               subject="I want to connect"),
        seen=s)
    assert (scope, pattern) == ("sender", "invitations@linkedin.com")


def test_a_rule_scoped_to_the_address_covers_the_next_stranger():
    s = _seen([("invitations@linkedin.com", "charmain guia"),
               ("invitations@linkedin.com", "daphna cibulski-cohen")])
    rule = rule_from_correction(
        thread(sender="Daphna Cibulski-Cohen <invitations@linkedin.com>",
               subject="I want to connect"),
        [ActionTemplate(kind="trash")], "owner corrected", seen=s)
    s.add_rule(rule)
    later = thread(id="t2", sender="Rajesh Kumar <invitations@linkedin.com>",
                   subject="I want to connect")
    assert [r.id for r in s.matching(later)] == [rule.id]


def test_a_recurring_display_name_is_still_an_identity():
    """newsletters-noreply@linkedin.com is Forbes AND S&P Global AND a person.

    Several names, but each one recurs - so the name identifies a publisher and
    scoping to the bare address would let a rule taught on a newsletter decide
    a person's mail. This is the case _sender_matches was built to protect.
    """
    s = _seen([("newsletters-noreply@linkedin.com", "s&p global via linkedin"),
               ("newsletters-noreply@linkedin.com", "s&p global via linkedin"),
               ("newsletters-noreply@linkedin.com", "s&p global via linkedin"),
               ("newsletters-noreply@linkedin.com", "forbes via linkedin"),
               ("newsletters-noreply@linkedin.com", "forbes via linkedin"),
               ("newsletters-noreply@linkedin.com", "forbes via linkedin")])
    scope, pattern = choose_scope(
        thread(sender="S&P Global via LinkedIn <newsletters-noreply@linkedin.com>",
               subject="Daily Update"),
        seen=s)
    assert pattern == "s&p global via linkedin <newsletters-noreply@linkedin.com>"


def test_one_sighting_teaches_nothing_about_the_name():
    """A single mail cannot say whether the name recurs, so nothing widens."""
    s = _seen([("someone@example.com", "a person")])
    scope, pattern = choose_scope(
        thread(sender="A Person <someone@example.com>", subject="hello"), seen=s)
    assert pattern == "a person <someone@example.com>"
