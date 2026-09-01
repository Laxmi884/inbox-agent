"""Run the bot:  python -m inbox_agent.telegram

Wires the same objects the notebook's Cell 2 wires, then polls. The graph,
audit chokepoint and preference store are untouched - only the renderer differs.
"""
from __future__ import annotations

import logging
import re
import sys

from langgraph.checkpoint.sqlite import SqliteSaver

from ..audit import AuditLog
from ..config import get_embeddings, load_settings, mask, use_model
from ..gmail import SnapshotGmailClient
from ..graph import build_graph
from ..policy import load_policy
from ..store import HeldQueue, PreferenceStore, build_store
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


def main() -> int:
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(name)s %(message)s")
    settings = load_settings()

    missing = [n for n, v in (("INBOX_TG_TOKEN", settings.tg_token),
                              ("INBOX_TG_CHAT_ID", settings.tg_chat_id)) if not v]
    if missing:
        print(f"Refusing to start: {', '.join(missing)} not set.\n"
              "Get a token from @BotFather, and set INBOX_TG_CHAT_ID to your own\n"
              "Telegram user id - the bot answers that id and no other.")
        return 2

    policy = load_policy(settings)
    client = SnapshotGmailClient(settings.snapshot_dir / "threads.json")
    log = AuditLog(settings.audit_log)
    prefs = PreferenceStore(build_store(get_embeddings()))
    # Own namespace, own store: a held item is work in flight, not durable
    # preference knowledge, and build_graph now requires the queue explicitly
    # (task 4) rather than building one for itself.
    held = HeldQueue(build_store())
    llm = use_model("gemma") if settings.backend == "ollama" else None
    if llm is None:
        from ..config import get_llm
        llm = get_llm()

    cm = SqliteSaver.from_conn_string("inbox_agent/checkpoints.sqlite")
    checkpointer = cm.__enter__()

    graph = build_graph(client=client, prefs=prefs, policy=policy, llm=llm,
                        settings=settings, log=log, held=held,
                        checkpointer=checkpointer)

    categories = policy_categories(policy)
    transport = HttpTransport(settings.tg_token)
    bot = Bot(transport=transport, graph=graph, settings=settings,
              categories=categories)

    print(f"backend   : {settings.backend}")
    print(f"dry_run   : {settings.dry_run}   <- nothing reaches Gmail while true")
    print(f"policy    : {policy.version}")
    print(f"mode      : {bot.mode}")
    print(f"chat id   : {settings.tg_chat_id}  (the only authorised sender)")
    print(f"token     : {mask(settings.tg_token)}")
    print(f"categories: {categories}")
    print("\nSend /triage in Telegram. Ctrl-C to stop.")

    try:
        run_polling(bot, transport)
    except KeyboardInterrupt:
        print("\nstopped. any parked run is still in the checkpoint.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
