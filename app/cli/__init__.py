"""CLI for CSV import and TimesNet prepare/train."""

from __future__ import annotations

from app.cli.handlers import dispatch as dispatch_command
from app.cli.parsers import build_parser


def main() -> int:
    """Parse CLI args and dispatch to the selected command handler."""
    parser = build_parser()
    args = parser.parse_args()
    return dispatch_command(args, parser)
