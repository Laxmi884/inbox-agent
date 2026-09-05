"""The launchd unit: the preflight, and the plist that is rendered not committed.

The bot has run on a hand-typed double-fork, which dies on reboot and had to be
restarted by hand twice in one session. These cover the two things that make
handing it to launchd safe rather than merely automatic: a preflight that
refuses to start a misconfigured bot, and a plist that carries THIS checkout's
paths instead of whoever's machine last committed one.
"""
import plistlib
import shutil
import subprocess
import textwrap
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent
RUN_BOT = REPO / "tools" / "run_bot.sh"
INSTALL = REPO / "tools" / "install-launchd.sh"


@pytest.fixture
def fake_python(tmp_path):
    """Stands in for the interpreter, recording what it was asked to run."""
    trace = tmp_path / "trace"
    fake = tmp_path / "python3"
    fake.write_text(textwrap.dedent(f"""\
        #!/bin/bash
        echo "$@" >> {trace}
        case "$*" in
          *doctor*) exit "${{DOCTOR_EXIT:-0}}" ;;
          *) exit 0 ;;
        esac
    """))
    fake.chmod(0o755)
    return fake, trace


def _run_bot(fake_python, doctor_exit):
    fake, trace = fake_python
    proc = subprocess.run(["bash", str(RUN_BOT)], capture_output=True, text=True,
                          env={"PATH": "/usr/bin:/bin", "HOME": str(fake.parent),
                               "INBOX_PYTHON": str(fake),
                               "DOCTOR_EXIT": str(doctor_exit)})
    return proc, (trace.read_text() if trace.exists() else "")


# --- the preflight ----------------------------------------------------------

def test_a_fatal_config_stops_the_bot_starting(fake_python):
    """The spec designed doctor's exit code for this. A bot started at login
    months later is not read by anyone the way a hand-started one is."""
    proc, trace = _run_bot(fake_python, doctor_exit=1)
    assert proc.returncode == 78                      # EX_CONFIG
    assert "telegram" not in trace


def test_a_sound_config_reaches_the_bot(fake_python):
    proc, trace = _run_bot(fake_python, doctor_exit=0)
    assert "inbox_agent.telegram" in trace


def test_the_preflight_runs_before_the_bot(fake_python):
    _, trace = _run_bot(fake_python, doctor_exit=0)
    lines = trace.strip().splitlines()
    assert "doctor" in lines[0]
    assert "telegram" in lines[-1]


def test_the_failure_says_why_on_stderr(fake_python):
    proc, _ = _run_bot(fake_python, doctor_exit=1)
    assert "doctor" in proc.stderr.lower()


def test_the_wrapper_sets_no_inbox_variables():
    """.env is the only source. A plist or wrapper that exported INBOX_* would
    be exactly the second place this project keeps re-learning not to have."""
    body = RUN_BOT.read_text()
    assert "export INBOX_GMAIL" not in body
    assert "INBOX_DRY_RUN=" not in body


# --- the rendered plist -----------------------------------------------------

@pytest.fixture
def rendered(tmp_path):
    proc = subprocess.run(["bash", str(INSTALL), "--dry-run"],
                          capture_output=True, text=True)
    assert proc.returncode == 0, proc.stderr
    return proc.stdout


def test_the_plist_parses(rendered):
    plistlib.loads(rendered.encode())


def test_no_placeholder_survives_rendering(rendered):
    """A half-rendered plist loads fine and points at nothing."""
    assert "__" not in rendered.split("<dict>", 1)[1]


def test_it_points_at_this_checkout(rendered):
    data = plistlib.loads(rendered.encode())
    assert data["WorkingDirectory"] == str(REPO)
    assert data["ProgramArguments"] == [str(RUN_BOT)]


def test_the_working_directory_is_where_dotenv_lives(rendered):
    """launchd gives a job no cwd. Without this, find_dotenv() finds nothing
    and the bot starts on defaults - snapshot, dry-run - looking normal."""
    data = plistlib.loads(rendered.encode())
    assert (Path(data["WorkingDirectory"]) / ".env.example").exists()


def test_the_interpreter_is_absolute(rendered):
    """Resolved at install time, because launchd has no PATH to look it up in."""
    data = plistlib.loads(rendered.encode())
    python = data["EnvironmentVariables"]["INBOX_PYTHON"]
    assert Path(python).is_absolute() and shutil.which(python)


def test_no_inbox_setting_is_baked_into_the_plist(rendered):
    """Only INBOX_PYTHON, which is where to find the interpreter, not what the
    agent should do. Anything else here would outrank .env and survive nothing."""
    env = plistlib.loads(rendered.encode())["EnvironmentVariables"]
    assert [k for k in env if k.startswith("INBOX_")] == ["INBOX_PYTHON"]


def test_restarts_are_throttled(rendered):
    """A broken config exits 78 from the preflight every time. Unthrottled,
    launchd would rerun doctor - which probes Ollama and reads the policy over
    the network - in a tight loop."""
    data = plistlib.loads(rendered.encode())
    assert data["KeepAlive"] is True
    assert data["ThrottleInterval"] >= 60


# --- macOS TCC --------------------------------------------------------------
#
# Found on 2026-09-05 by installing from ~/Documents/Projects: the job spawned,
# exited 126 "Operation not permitted" before a line of Python ran, and would
# have retried every 60s forever. A launchd user agent has no consent grant for
# the protected directories.

def _install(cwd_repo: Path, *args):
    return subprocess.run(["bash", str(cwd_repo / "tools" / "install-launchd.sh"),
                           *args], capture_output=True, text=True)


@pytest.fixture
def checkout(tmp_path):
    """A copy of the tooling, so the path under test is the fixture's."""
    def build(root: Path) -> Path:
        (root / "tools" / "launchd").mkdir(parents=True)
        for rel in ("tools/install-launchd.sh", "tools/run_bot.sh",
                    "tools/launchd/com.inbox-agent.bot.plist.template"):
            shutil.copy(REPO / rel, root / rel)
        (root / ".env.example").write_text("")
        return root
    return build


@pytest.mark.parametrize("protected", ["Documents", "Desktop", "Downloads"])
def test_a_tcc_protected_checkout_is_refused(checkout, tmp_path, protected,
                                             monkeypatch):
    root = checkout(tmp_path / protected / "repo")
    proc = subprocess.run(["bash", str(root / "tools/install-launchd.sh")],
                          capture_output=True, text=True,
                          env={"PATH": "/usr/bin:/bin", "HOME": str(tmp_path)})
    assert proc.returncode == 1
    assert "TCC-protected" in proc.stderr


def test_the_refusal_offers_both_ways_out(checkout, tmp_path):
    root = checkout(tmp_path / "Documents" / "repo")
    proc = subprocess.run(["bash", str(root / "tools/install-launchd.sh")],
                          capture_output=True, text=True,
                          env={"PATH": "/usr/bin:/bin", "HOME": str(tmp_path)})
    assert "Move the checkout" in proc.stderr
    assert "Full Disk Access" in proc.stderr


def test_an_unprotected_checkout_gets_past_the_tcc_gate(checkout, tmp_path):
    """It may still stop for another reason - a running bot - but not for this."""
    root = checkout(tmp_path / "dev" / "repo")
    proc = subprocess.run(["bash", str(root / "tools/install-launchd.sh")],
                          capture_output=True, text=True,
                          env={"PATH": "/usr/bin:/bin", "HOME": str(tmp_path)})
    assert "TCC-protected" not in proc.stderr


def test_dry_run_renders_even_from_a_protected_path():
    """Printing a plist harms nothing, and it is how these tests read it. The
    gate belongs on the install path, not on inspection."""
    proc = _install(REPO, "--dry-run")
    assert proc.returncode == 0
    assert "<plist" in proc.stdout
