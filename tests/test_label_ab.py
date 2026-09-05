"""The labelled reference set: the only thing here that knows what is RIGHT.

The A/B compares arms to each other. These tests pin the three properties that
stop the resulting file from quietly ceasing to be evidence - see the module
docstring in tools/label_ab.py.
"""
import json

import pytest

from tools import label_ab


@pytest.fixture
def store(tmp_path, monkeypatch):
    monkeypatch.setattr(label_ab, "STORE", tmp_path)
    monkeypatch.setattr(label_ab, "LABELS", tmp_path / "ab_labels.json")
    monkeypatch.setattr(label_ab, "CACHE", tmp_path / "ab_threads.json")
    monkeypatch.setattr(label_ab, "RESULTS", tmp_path / "ab_results.json")
    (tmp_path / "ab_threads.json").write_text(json.dumps({"threads": [
        {"id": "t1", "subject": "Paid you 10.33", "sender": "a@b.com",
         "date": "2026-01-01", "body": "receipt body"},
        {"id": "t2", "subject": "Agreed thread", "sender": "c@d.com",
         "date": "2026-01-02", "body": "x"}]}))
    (tmp_path / "ab_results.json").write_text(json.dumps({
        "policy_version": "hub:abc123", "strata": {"t1": "primary/2026"},
        "arms": {
            "snippet": [{"thread_id": "t1", "category": "automated",
                         "subject": "Paid you 10.33"},
                        {"thread_id": "t2", "category": "promotion",
                         "subject": "Agreed thread"}],
            "full": [{"thread_id": "t1", "category": "receipt",
                      "subject": "Paid you 10.33"},
                     {"thread_id": "t2", "category": "promotion",
                      "subject": "Agreed thread"}]}}))
    monkeypatch.setattr(label_ab, "categories",
                        lambda: ["receipt", "automated", "promotion", "other"])
    return label_ab


def test_only_threads_the_arms_disagreed_about_need_judging(store):
    assert [d["thread_id"] for d in store.disagreements()] == ["t1"]


def test_a_disagreement_carries_the_body_to_judge_it_by(store):
    d = store.disagreements()[0]
    assert d["body_excerpt"] == "receipt body"
    assert d["arms"] == {"snippet": "automated", "full": "receipt"}


def test_the_label_may_be_neither_arms_answer(store, capsys):
    """Both arms can be wrong. Constraining the truth to their two answers
    would bake their shared blind spots into the ground truth."""
    store.save_label("t1", "other")
    saved = json.loads(store.LABELS.read_text())["items"]["t1"]
    assert saved["truth"] == "other"
    assert "no arm got this right" in capsys.readouterr().out


def test_a_label_records_the_policy_that_defined_it(store):
    """`newsletter_valuable` under a rewritten policy is a different label
    wearing the same name."""
    store.save_label("t1", "receipt")
    assert json.loads(store.LABELS.read_text())["policy_version"] == "hub:abc123"


def test_a_label_keeps_the_disagreement_that_prompted_it(store):
    """Without it the row cannot answer "did the body help", which is the
    question it was collected for."""
    store.save_label("t1", "receipt")
    saved = json.loads(store.LABELS.read_text())["items"]["t1"]
    assert saved["arms"] == {"snippet": "automated", "full": "receipt"}


def test_a_category_the_policy_does_not_define_is_refused(store):
    with pytest.raises(SystemExit) as e:
        store.save_label("t1", "definitely_spam")
    assert "not a policy category" in str(e.value)


def test_a_thread_the_arms_agreed_on_is_refused(store):
    with pytest.raises(SystemExit):
        store.save_label("t2", "promotion")


def test_relabelling_replaces_rather_than_duplicates(store):
    store.save_label("t1", "automated")
    store.save_label("t1", "receipt", note="changed my mind")
    items = json.loads(store.LABELS.read_text())["items"]
    assert len(items) == 1
    assert items["t1"]["truth"] == "receipt"
    assert items["t1"]["note"] == "changed my mind"


def test_status_scores_each_arm_against_the_human(store, capsys):
    store.save_label("t1", "receipt")
    store.status()
    out = capsys.readouterr().out
    assert "snippet   0/1" in out
    assert "full      1/1" in out


def test_review_hides_what_is_already_labelled(store, capsys):
    store.save_label("t1", "receipt")
    store.review(show_all=False)
    assert "1 labelled" in capsys.readouterr().out


def test_categories_come_from_the_policy_file():
    """Not a hardcoded list - a label the policy does not define is a typo
    discovered months later by whatever consumes the file."""
    assert "needs_reply" in label_ab.categories()
