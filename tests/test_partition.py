"""The autonomy ladder as code. Pure, so it needs no graph and no LLM."""
from inbox_agent.models import Action, ReviewItem
from inbox_agent.partition import hold_reason, partition


def item(kind="archive", *, category="promotion", conf=0.9, source="model",
         rule_id=None, tid="t1"):
    return ReviewItem(
        thread_id=tid, category=category, subject="S", sender="a@b.com",
        snippet="s", proposed=[Action(kind=kind, thread_id=tid)],
        reason="because", confidence=conf, source=source, rule_id=rule_id)


def test_reversible_confident_actions_are_not_held():
    for kind in ("label", "archive", "draft", "none"):
        assert hold_reason(item(kind)) is None


def test_model_proposed_trash_is_held():
    assert hold_reason(item("trash")) == "trash"


def test_rule_authorised_trash_is_not_held():
    """The ladder's 'learned rule, else human approval' row. Without this
    clause trash can never graduate, and the row never does anything."""
    assert hold_reason(item("trash", source="rule", rule_id="r-1")) is None


def test_low_confidence_is_held_even_for_a_reversible_action():
    assert hold_reason(item("archive", conf=0.41)) == "low_confidence"


def test_confidence_exactly_at_the_threshold_is_not_low():
    """LOW_CONFIDENCE is 0.5 and the comparison is strict, matching render.py."""
    assert hold_reason(item("archive", conf=0.5)) is None


def test_needs_reply_is_held_for_attention():
    assert hold_reason(item("draft", category="needs_reply")) == "needs_reply"


def test_security_alert_is_held_for_attention():
    assert hold_reason(item("label", category="security_alert")) == "security_alert"


def test_trash_outranks_low_confidence():
    assert hold_reason(item("trash", conf=0.2)) == "trash"


def test_low_confidence_outranks_needs_reply():
    """A low-confidence needs_reply must never be one-tap approvable: the
    one-tap button covers the attention tier only."""
    assert hold_reason(item("draft", category="needs_reply", conf=0.2)) == "low_confidence"


def test_an_item_with_several_actions_is_held_if_any_is_trash():
    multi = ReviewItem(
        thread_id="t9", category="promotion", subject="S", sender="a@b.com",
        snippet="s",
        proposed=[Action(kind="label", thread_id="t9"),
                  Action(kind="trash", thread_id="t9")],
        reason="r", confidence=0.9, source="model")
    assert hold_reason(multi) == "trash"


def test_partition_splits_and_preserves_order():
    items = [item(tid="t0"), item("trash", tid="t1"), item(tid="t2"),
             item("archive", conf=0.1, tid="t3")]
    auto, held = partition(items)
    assert [i.thread_id for i in auto] == ["t0", "t2"]
    assert [(i.thread_id, r) for i, r in held] == [("t1", "trash"),
                                                   ("t3", "low_confidence")]


def test_partition_of_an_empty_list_is_two_empty_lists():
    assert partition([]) == ([], [])
