"""One bot per install.

Two processes long-polling one Telegram token split the updates between them
and nothing errors - the digest just starts behaving as though the owner were
pressing buttons at random. These pin that the second one cannot start.
"""
import fcntl
import os
import subprocess
import sys
import textwrap

import pytest

from inbox_agent.single_instance import LOCK_NAME, AlreadyRunning, acquire


def test_the_first_process_gets_the_lock(tmp_path):
    handle = acquire(tmp_path)
    assert (tmp_path / LOCK_NAME).exists()
    handle.close()


def test_the_second_is_refused(tmp_path):
    held = acquire(tmp_path)
    with pytest.raises(AlreadyRunning):
        acquire(tmp_path)
    held.close()


def test_the_refusal_names_the_holding_pid(tmp_path):
    """"Refusing to start" is useless without something to go and look at."""
    held = acquire(tmp_path)
    with pytest.raises(AlreadyRunning) as exc:
        acquire(tmp_path)
    assert str(os.getpid()) in str(exc.value)
    held.close()


def test_a_refused_attempt_does_not_erase_the_holders_pid(tmp_path):
    """Opening "w" would truncate before the lock is attempted, so the second
    process would wipe the very pid its error message needs."""
    held = acquire(tmp_path)
    with pytest.raises(AlreadyRunning):
        acquire(tmp_path)
    assert (tmp_path / LOCK_NAME).read_text().strip() == str(os.getpid())
    held.close()


def test_releasing_lets_the_next_one_in(tmp_path):
    acquire(tmp_path).close()
    acquire(tmp_path).close()


def test_two_installs_with_different_stores_do_not_collide(tmp_path):
    """The lock is per install. Two store dirs are two bots by intention."""
    a, b = tmp_path / "a", tmp_path / "b"
    ha, hb = acquire(a), acquire(b)
    ha.close()
    hb.close()


def test_the_store_directory_is_created_if_absent(tmp_path):
    acquire(tmp_path / "does" / "not" / "exist").close()


def test_a_killed_process_leaves_no_stale_lock(tmp_path):
    """The reason this is a flock and not a pid file. The process that most
    needs to clean up is the one that died without the chance to - so the
    kernel does it instead."""
    script = textwrap.dedent(f"""
        import sys, time
        sys.path.insert(0, {str(os.getcwd())!r})
        from inbox_agent.single_instance import acquire
        h = acquire({str(tmp_path)!r})
        print("locked", flush=True)
        time.sleep(60)
    """)
    child = subprocess.Popen([sys.executable, "-c", script],
                             stdout=subprocess.PIPE, text=True)
    assert child.stdout.readline().strip() == "locked"
    with pytest.raises(AlreadyRunning):
        acquire(tmp_path)

    child.kill()
    child.wait(timeout=10)
    acquire(tmp_path).close()          # no cleanup ran; the lock is free anyway


def test_the_lock_is_held_by_the_handle_not_the_file(tmp_path):
    """Documents why callers must keep the handle referenced: garbage
    collecting it releases the lock and silently permits a second bot."""
    handle = acquire(tmp_path)
    fcntl.flock(handle, fcntl.LOCK_UN)
    acquire(tmp_path).close()
