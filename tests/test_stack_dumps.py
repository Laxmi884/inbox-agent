"""Making the next wedge diagnosable without a root password.

On 2026-09-11 the only way to learn where the bot was stuck was
`sudo py-spy dump`, which needs root on macOS and - with no TTY under launchd
or an agent tool call - a GUI auth dialog to type a password into. The stack
was worth it: `attempt: 1` in _with_backoff is what proved the hang was a
missing timeout rather than an exhausted retry ladder.

faulthandler is in the stdlib and costs one registration at startup. With it,
the same evidence is `kill -USR1 <pid>` and a look at the log - no password, no
dialog, and available to whoever is on call rather than only to the machine's
owner.
"""
import faulthandler
import signal

from inbox_agent import logging_setup


def test_a_signal_dumps_the_stack(tmp_path, monkeypatch):
    """The registration is real, not merely attempted."""
    monkeypatch.setattr(faulthandler, "register", _record := _Recorder())
    logging_setup.enable_stack_dumps()
    assert _record.signum == signal.SIGUSR1
    assert _record.all_threads is True, (
        "a one-thread dump would have missed nothing on 2026-09-11 by luck: "
        "the hang was on MainThread. The hydration pool is where the next one "
        "will be.")


def test_registration_never_stops_the_bot(monkeypatch):
    """Signals cannot be registered off the main thread, and a notebook or a
    test runner may well import this from one. A diagnostic aid that refuses
    to start the process it exists to diagnose is worse than none."""
    def boom(*a, **k):
        raise ValueError("signal only works in main thread")

    monkeypatch.setattr(faulthandler, "register", boom)
    logging_setup.enable_stack_dumps()          # must not raise


def test_configure_wires_it_up(monkeypatch, tmp_path):
    """Nobody remembers to call a diagnostic helper by hand. It rides along
    with the logging every entry point already configures."""
    called = []
    monkeypatch.setattr(logging_setup, "enable_stack_dumps",
                        lambda *a, **k: called.append(1))
    logging_setup.configure(log_file=tmp_path / "x.log")
    assert called, "the bot would start with no way to dump its own stack"


class _Recorder:
    signum = None
    all_threads = None

    def __call__(self, signum, file=None, all_threads=True, chain=False):
        self.signum, self.all_threads = signum, all_threads
