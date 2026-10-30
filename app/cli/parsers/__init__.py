"""Register all top-level CLI subcommand parsers."""

from __future__ import annotations

import argparse

from app.cli.parsers import imports, timesnet


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="caltrans-trafmon",
        description="Traffic monitor command-line interface.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    imports.register(subparsers)
    timesnet.register(subparsers)

    return parser
