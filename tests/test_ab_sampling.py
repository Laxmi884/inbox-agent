"""The stratified sampler in tools/ab_body.py, and the ids-only client path.

The sampler decides which mail the body-vs-snippet experiment is measured on,
so a bias in it is a wrong answer that looks like a result. These pin the two
properties that matter: coverage of the rare strata, and reaching past the
first page of a mailbox with 20 000 threads in it.
"""
import random

import pytest

from inbox_agent.gmail import LiveGmailClient
from test_live_gmail import FakeGmailApi, message
from tools.ab_body import _strata, allocate, arm_subset

CATEGORIES = 5


def rng():
    return random.Random(0)


# --- allocate ---------------------------------------------------------------

def test_equal_coverage_beats_proportional():
    """A mailbox that is 98% promotions must not return a 98% promotions
    sample - that is the recency bias this sampler exists to remove."""
    pools = {"promotions": [f"p{i}" for i in range(500)],
             "primary": [f"m{i}" for i in range(5)]}
    chosen = allocate(pools, 10, rng())
    assert sum(1 for v in chosen.values() if v == "primary") == 5
    assert sum(1 for v in chosen.values() if v == "promotions") == 5


def test_an_exhausted_stratum_redistributes_its_quota():
    pools = {"a": ["a1"], "b": [f"b{i}" for i in range(20)]}
    chosen = allocate(pools, 10, rng())
    assert len(chosen) == 10
    assert sum(1 for v in chosen.values() if v == "a") == 1


def test_empty_strata_need_no_special_case():
    """Gmail returns nothing for category:forums on an account that never used
    it. That must cost nothing, not skew or crash the draw."""
    pools = {"forums": [], "social": [], "primary": [f"m{i}" for i in range(9)]}
    chosen = allocate(pools, 6, rng())
    assert len(chosen) == 6
    assert set(chosen.values()) == {"primary"}


def test_a_thread_in_two_strata_is_drawn_once():
    """A thread whose messages span a year boundary matches two date strata."""
    pools = {"primary/2026": ["shared", "a"], "primary/2025": ["shared", "b"]}
    chosen = allocate(pools, 4, rng())
    assert len(chosen) == 3
    assert sorted(chosen) == ["a", "b", "shared"]


def test_the_seed_makes_the_draw_reproducible():
    pools = {"a": [f"a{i}" for i in range(50)], "b": [f"b{i}" for i in range(50)]}
    assert allocate(pools, 12, random.Random(7)) == allocate(pools, 12, random.Random(7))
    assert allocate(pools, 12, random.Random(7)) != allocate(pools, 12, random.Random(8))


def test_asking_for_more_than_exists_returns_what_exists():
    assert len(allocate({"a": ["a1", "a2"]}, 100, rng())) == 2


# --- strata -----------------------------------------------------------------

def test_every_category_is_crossed_with_every_year():
    strata = _strata(5, "in:inbox")
    assert len(strata) == CATEGORIES * 5
    assert all(q.startswith("in:inbox category:") for _, q in strata)


def test_the_oldest_stratum_is_open_ended():
    """No mail may be unreachable however old the account is, so the last
    bucket has a before: and no after:."""
    oldest = _strata(5, "in:inbox")[4][1]
    assert "before:" in oldest and "after:" not in oldest


def test_year_strata_do_not_overlap():
    spans = [q.split("category:")[1] for label, q in _strata(3, "in:inbox")
             if label.startswith("primary/")]
    assert len(set(spans)) == 3


# --- arm_subset -------------------------------------------------------------

def test_arm_subset_keeps_cache_order():
    raw = [{"id": f"t{i}"} for i in range(20)]
    strata = {f"t{i}": ("a" if i % 2 else "b") for i in range(20)}
    got = [t["id"] for t in arm_subset(raw, strata, 6, 0)]
    assert got == sorted(got, key=lambda i: int(i[1:]))
    assert len(got) == 6


def test_arm_subset_spans_strata():
    raw = [{"id": f"t{i}"} for i in range(20)]
    strata = {f"t{i}": ("rare" if i == 0 else "common") for i in range(20)}
    got = {t["id"] for t in arm_subset(raw, strata, 4, 0)}
    assert "t0" in got


def test_arm_subset_of_zero_or_all_is_the_whole_cache():
    raw = [{"id": f"t{i}"} for i in range(5)]
    assert arm_subset(raw, {}, 0, 0) == raw
    assert arm_subset(raw, {}, 99, 0) == raw


def test_an_unstratified_cache_still_subsets():
    """A v1 cache from --fetch carries no strata at all."""
    raw = [{"id": f"t{i}"} for i in range(10)]
    assert len(arm_subset(raw, {}, 3, 0)) == 3


# --- the ids-only client path -----------------------------------------------

@pytest.fixture
def big_api():
    api = FakeGmailApi(threads={
        f"t{i}": [message(f"m{i}", sender="a@b.com", subject=f"S{i}",
                          to="me@z.com", date="Wed, 26 Aug 2026 10:00:00 +0000",
                          snippet="s", body="b", label_ids=["INBOX"])]
        for i in range(1200)})
    api.page_size = 500
    return api


def test_list_thread_ids_pages_past_the_first_500(big_api):
    client = LiveGmailClient(big_api)
    assert len(client.list_thread_ids(max_ids=1200)) == 1200
    assert len(big_api.queries) == 3
    assert big_api.gets == []


def test_list_thread_ids_hydrates_nothing(big_api):
    """The whole point: enumerating the mailbox must not cost a get per thread."""
    LiveGmailClient(big_api).list_thread_ids(max_ids=1000)
    assert big_api.gets == []


def test_max_ids_bounds_an_endless_mailbox(big_api):
    client = LiveGmailClient(big_api)
    assert len(client.list_thread_ids(max_ids=600)) == 600
    assert sum(q["maxResults"] for q in big_api.queries) == 600


def test_get_threads_hydrates_only_what_it_is_given(big_api):
    threads = LiveGmailClient(big_api).get_threads(["t3", "t1", "t2"])
    assert [t.id for t in threads] == ["t3", "t1", "t2"]
    assert sorted(big_api.gets) == ["t1", "t2", "t3"]


def test_get_threads_of_nothing_calls_nothing(big_api):
    assert LiveGmailClient(big_api).get_threads([]) == []
    assert big_api.gets == []


# --- paced, resumable hydration ---------------------------------------------

class _Recorder:
    """A client that records how many ids each get_threads call was handed."""

    def __init__(self, fail_after=None):
        self.batches, self._seen = [], 0
        self._fail_after = fail_after

    def get_threads(self, ids, *, workers=None):
        self.batches.append((list(ids), workers))
        out = []
        for tid in ids:
            self._seen += 1
            if self._fail_after is not None and self._seen > self._fail_after:
                raise RuntimeError("429 pretend")
            out.append(_Row(tid))
        return out


class _Row:
    def __init__(self, tid):
        self.id = tid

    def model_dump(self, mode="json"):
        return {"id": self.id, "subject": self.id, "sender": "a@b.com",
                "to": [], "date": "", "snippet": "s", "body": "b" * 10,
                "label_ids": []}


@pytest.fixture
def store(tmp_path, monkeypatch):
    from tools import ab_body
    monkeypatch.setattr(ab_body, "STORE", tmp_path)
    monkeypatch.setattr(ab_body, "CACHE", tmp_path / "ab_threads.json")
    monkeypatch.setattr(ab_body, "HYDRATE_PAUSE", 0)
    return ab_body


def test_hydration_is_chunked_not_one_burst(store):
    """Five workers on 200 threads is what breached the per-minute quota."""
    chosen = {f"t{i}": "primary/2026" for i in range(60)}
    client = _Recorder()
    rows = store._hydrate_paced(client, chosen, seed=0)
    assert len(rows) == 60
    assert [len(ids) for ids, _ in client.batches] == [25, 25, 10]
    assert {w for _, w in client.batches} == {store.HYDRATE_WORKERS}


def test_a_rate_limit_costs_one_chunk_not_the_run(store):
    """The first live 200-thread sample hydrated most of them and cached none,
    because one 403 near the end unwound the whole list."""
    chosen = {f"t{i}": "primary/2026" for i in range(60)}
    with pytest.raises(RuntimeError):
        store._hydrate_paced(_Recorder(fail_after=30), chosen, seed=0)
    assert len(store._cached_rows(chosen)) == 25


def test_a_resumed_run_only_fetches_what_is_missing(store):
    chosen = {f"t{i}": "primary/2026" for i in range(60)}
    with pytest.raises(RuntimeError):
        store._hydrate_paced(_Recorder(fail_after=30), chosen, seed=0)

    client = _Recorder()
    rows = store._hydrate_paced(client, chosen, seed=0)
    assert len(rows) == 60
    fetched = [tid for ids, _ in client.batches for tid in ids]
    assert len(fetched) == 35
    assert "t0" not in fetched


def test_the_cache_keeps_draw_order_across_a_resume(store):
    chosen = {f"t{i}": "primary/2026" for i in range(40)}
    with pytest.raises(RuntimeError):
        store._hydrate_paced(_Recorder(fail_after=10), chosen, seed=0)
    rows = store._hydrate_paced(_Recorder(), chosen, seed=0)
    assert [r["id"] for r in rows] == list(chosen)


def test_cached_rows_ignores_threads_this_draw_did_not_pick(store):
    store.CACHE.write_text('{"version": 2, "threads": [{"id": "other"}]}')
    assert store._cached_rows({"t1": "a"}) == {}


def test_a_corrupt_cache_is_refetched_not_fatal(store):
    store.CACHE.write_text("{not json")
    assert store._cached_rows({"t1": "a"}) == {}


# --- the CLI's expensive path is opt-in -------------------------------------

def test_caching_does_not_fall_through_into_the_model_run(store, monkeypatch,
                                                          capsys):
    """--sample is a two-minute read-only fetch; the arms behind it are hours
    of local GPU. Running both by accident is what happened on 2026-09-04."""
    ran = []
    monkeypatch.setattr(store, "sample",
                        lambda *a, **k: ran.append("sampled"))
    monkeypatch.setattr(store, "run_arm",
                        lambda *a, **k: ran.append("classified") or [])
    store.CACHE.write_text('{"version": 2, "strata": {}, "threads": '
                           '[{"id": "t1", "snippet": "s", "body": "b"}]}')

    assert store.main(["--sample", "5"]) == 0
    assert ran == ["sampled"]
    assert "--arms" in capsys.readouterr().out


def test_naming_the_arms_is_what_starts_the_run(store, monkeypatch):
    ran = []
    monkeypatch.setattr(store, "run_arm",
                        lambda raw, budget, label: ran.append(label) or [])
    monkeypatch.setattr(store, "compare", lambda results: None)
    monkeypatch.setattr(store, "RESULTS", store.STORE / "r.json")
    monkeypatch.setattr(store, "DISAGREEMENTS", store.STORE / "d.md")
    store.CACHE.write_text('{"version": 2, "strata": {}, "threads": '
                           '[{"id": "t1", "snippet": "s", "body": "b"}]}')

    assert store.main(["--arms", "snippet,full"]) == 0
    assert ran == ["snippet", "full"]
