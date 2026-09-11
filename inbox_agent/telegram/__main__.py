"""Run the bot:  python -m inbox_agent.telegram

Wires the same objects the notebook's Cell 2 wires, then polls. The graph,
audit chokepoint and preference store are untouched - only the renderer differs.
"""
from __future__ import annotations

import logging
import re
import sys
from datetime import datetime

from langgraph.checkpoint.sqlite import SqliteSaver

from ..audit import AuditLog
from ..config import (build_embeddings, build_gmail_client,
                      describe_body_budget, load_settings, mask, use_model)
from ..doctor import alert_text, health_alerts, oauth_check, tracing_check
from ..logging_setup import configure as configure_logging
from ..single_instance import AlreadyRunning, acquire
from ..graph import build_graph
from ..policy import load_policy
from ..schedule import ScheduleStore, Trigger
from ..store import DoneStore, HeldQueue, PreferenceStore, open_store
from .bot import Bot, HttpTransport, run_polling


def policy_categories(policy) -> list[str]:
    """The label buttons come from the policy, so the UI cannot offer a
    category the policy does not define. Same reasoning as deriving
    VALID_ACTION_KINDS from the Literal rather than hand-copying it: the
    taxonomy has exactly one source.
    """
    section = policy.text.split("## Categories", 1)
    if len(section) < 2:
        return []
    body = section[1].split("\n## ", 1)[0]
    return re.findall(r"^-\s+`([a-z_]+)`", body, re.M)


def _tracing_banner() -> str:
    """The banner's tracing line, as a string so it can be tested.

    Same reasoning as _embeddings_banner: a mode nobody can see is the failure
    this banner exists to prevent. The banner is printed BY the process, which
    makes it the only place that can honestly report what THIS process is
    doing rather than what .env currently says - and on 2026-09-04 those were
    different for seventeen hours. See doctor.tracing_check.
    """
    check = tracing_check()
    return (f"tracing   : {check.value}"
            + (f"   <- {check.note}" if check.level != "ok" else ""))


def _schedule_banner(settings, now: datetime | None = None) -> str:
    """What the schedule is, from the process that holds it.

    A .env edit does not reach a running bot - load_dotenv runs once, at import -
    so the banner is the honest source and doctor can only report the file.
    """
    if not settings.schedule:
        return "schedule  : off   <- runs only when you send /triage"
    slots = ", ".join(t.strftime("%H:%M") for t in settings.schedule)
    nxt = Trigger(slots=settings.schedule).next_slot(now or datetime.now())
    return f"schedule  : {slots} local (next {nxt.strftime('%H:%M')})"


def _embeddings_banner(settings, resolved: str) -> str:
    """The banner's embeddings line, as a string so it can be tested.

    Takes the RESOLVED mode as a required argument rather than probing again:
    on a machine whose `ollama serve` has died, resolving twice cost a second
    Ollama probe, a duplicated degrade warning, and a window in which the
    banner could report a mode other than what the store actually got. This
    is a pure formatter over (configured, resolved) - no side effects, no
    network. "none" alone would be ambiguous - it is also what a deliberate
    INBOX_EMBEDDINGS=none looks like - so a degrade says why.
    """
    if resolved == "none" and settings.embeddings == "auto":
        return ("embeddings: none   <- configured auto, but Ollama is not "
                "listening; rules still match exactly")
    return f"embeddings: {resolved}"


def main() -> int:
    # Before anything that logs, but after nothing: load_settings only reads
    # env, so the log file can be configured from the same file it reads.
    settings = load_settings()
    configure_logging(log_file=settings.log_file)

    missing = [n for n, v in (("INBOX_TG_TOKEN", settings.tg_token),
                              ("INBOX_TG_CHAT_ID", settings.tg_chat_id)) if not v]
    if missing:
        print(f"Refusing to start: {', '.join(missing)} not set.\n"
              "Get a token from @BotFather, and set INBOX_TG_CHAT_ID to your own\n"
              "Telegram user id - the bot answers that id and no other.")
        return 2

    # Held for the life of the process. This local is the only reference, and
    # dropping it releases the lock and silently permits a second bot - which
    # is why it is not an underscore-prefixed throwaway.
    try:
        instance_lock = acquire(settings.store_dir)      # noqa: F841 - see above
    except AlreadyRunning as exc:
        print(f"Refusing to start: {exc}")
        return 3

    policy = load_policy(settings)
    client = build_gmail_client(settings)
    log = AuditLog(settings.audit_log)
    # On disk, not in memory. The bot is the one caller whose lifetime is not
    # the run: it is restarted for a code change, a laptop lid, a crash. Rules
    # were merely lost that way; held items became unreachable, because
    # mark_triaged takes every processed thread out of the fetch query and
    # `/backlog` uses the same query. See open_store.
    embeddings_mode, embeddings = build_embeddings(settings.embeddings)
    prefs = PreferenceStore(open_store(settings.store_dir / "prefs.sqlite",
                                       embeddings))
    # Own namespace, own file: a held item is work in flight, not durable
    # preference knowledge, and build_graph now requires the queue explicitly
    # (task 4) rather than building one for itself. Separate files rather than
    # one, because only the rules want the embedding index - pointing it at a
    # held payload with no `text` field would cost embedding calls to index
    # nothing.
    work_store = open_store(settings.store_dir / "held.sqlite")
    held = HeldQueue(work_store)
    # Same store, own namespace (spec 2.1): a run report wants an embedding
    # index no more than a held item does, and a third sqlite file would be a
    # third object threaded through build_graph and Bot for no gain.
    done = DoneStore(work_store)
    llm = use_model("gemma") if settings.backend == "ollama" else None
    if llm is None:
        from ..config import get_llm
        llm = get_llm()

    cm = SqliteSaver.from_conn_string("inbox_agent/checkpoints.sqlite")
    checkpointer = cm.__enter__()

    graph = build_graph(client=client, prefs=prefs, policy=policy, llm=llm,
                        settings=settings, log=log, held=held, done=done,
                        checkpointer=checkpointer)

    categories = policy_categories(policy)
    transport = HttpTransport(settings.tg_token)
    # The same queue the graph fills: one instance, so the digest renders what
    # the run actually held rather than a second, permanently empty store.
    # The same PreferenceStore the graph reads rules from: a correction is a
    # store write, and two instances would let one land where nothing reads it.
    bot = Bot(transport=transport, graph=graph, settings=settings, held=held,
              done=done, prefs=prefs, client=client, log=log,
              categories=categories, policy_version=policy.version)

    print(f"backend   : {settings.backend}")
    # Which mailbox is in play is the single thing to be certain of before a
    # live run, so it is named here and named loudly when it is the real one.
    print(f"gmail     : {settings.gmail}"
          + ("   <- THE REAL MAILBOX" if settings.gmail == "live" else ""))
    print(f"dry_run   : {settings.dry_run}   <- nothing reaches Gmail while true")
    print(f"body      : {describe_body_budget(settings.body_budget)}")
    print(f"http      : {settings.http_timeout:g}s before a Gmail socket is abandoned")
    print(_tracing_banner())
    print(f"policy    : {policy.version}"
          + ("   <- DRIFTED from the committed policies/default.md"
             if policy.drifted else ""))
    print(f"mode      : {bot.mode}")
    print(_schedule_banner(settings))
    print(f"chat id   : {settings.tg_chat_id}  (the only authorised sender)")
    print(f"token     : {mask(settings.tg_token)}")
    print(f"categories: {categories}")
    # Live rules, not all rules. A retired rule is kept as part of the record
    # but can never fire, and counting it here would overstate what the agent
    # actually carries - the same overstatement the dead simplywall.st rule
    # already makes on its own.
    _live = [r for r in prefs.rules() if not r.overridden]
    _retired = len(prefs.rules()) - len(_live)
    print(f"store     : {settings.store_dir}  ({len(held.all())} held, "
          f"{len(_live)} rules carried over"
          + (f", {_retired} retired)" if _retired else ")"))
    print(_embeddings_banner(settings, embeddings_mode))
    # The refresh token dies seven days after consent whatever the bot does, and
    # this countdown has been computed correctly since it was written - on a
    # screen nobody opens. Say it here, and say it again on the phone below.
    _oauth = oauth_check(settings)
    print(f"oauth     : {_oauth.value}" + (f"   {_oauth.note}" if _oauth.note else ""))
    print("\nSend /triage in Telegram. Ctrl-C to stop.")

    # Everything above this line is a terminal banner, and the lesson this repo
    # keeps relearning is that a terminal banner is not where the owner is: the
    # same argument bot.py already makes about dry_run. Anything at warn or
    # fatal goes to the phone as well, once, at startup.
    try:
        alerts = alert_text(health_alerts(settings, policy=policy))
        if alerts:
            transport.send_message(settings.tg_chat_id, f"Bot started.\n{alerts}")
    except Exception:                 # never let the health notice stop the bot
        # `log` in this function is the AuditLog, not a logger.
        logging.getLogger(__name__).exception(
            "could not send the startup health alert")

    try:
        trigger = Trigger(slots=settings.schedule)
        store = ScheduleStore(settings.store_dir / "schedule.json")
        run_polling(bot, transport, trigger=trigger, store=store)
    except KeyboardInterrupt:
        print("\nstopped. any parked run is still in the checkpoint.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
