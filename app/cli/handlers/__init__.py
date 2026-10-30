"""Dispatch CLI invocations to command handlers."""

from __future__ import annotations

from app.cli.handlers import imports, timesnet

_DISPATCHERS = (
    imports.dispatch,
    timesnet.dispatch,
)


def dispatch(args, parser) -> int:
    for handler in _DISPATCHERS:
        result = handler(args, parser)
        if result is not None:
            return result
    parser.error(f"Unknown command: {args.command}")
