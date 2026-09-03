"""The `inbox-agent` entry point.

Deliberately thin. `bot` delegates to inbox_agent.telegram.__main__.main, which
already wires the graph, the stores, the checkpointer and the startup banner -
a second copy of that wiring would drift from the first, and the wiring is
where the safety-relevant decisions are printed.

`python -m inbox_agent.telegram` keeps working and is not deprecated; this is
an addition, not a replacement.
"""
from __future__ import annotations

import sys
from typing import Optional, Sequence

_USAGE = """usage: inbox-agent <command>

  doctor   report the effective configuration, where each value came from,
           and what would stop the bot from starting
  bot      run the Telegram bot (same as: python -m inbox_agent.telegram)
"""


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = list(sys.argv[1:] if argv is None else argv)
    command = args[0] if args else ""

    if command == "doctor":
        from . import doctor
        return doctor.main(args[1:])

    if command == "bot":
        from .telegram.__main__ import main as bot_main
        return bot_main()

    # No default. Falling through to `bot` on a typo would start a run against
    # the real mailbox because someone mistyped a diagnostic command.
    print(_USAGE)
    return 2


if __name__ == "__main__":
    sys.exit(main())
