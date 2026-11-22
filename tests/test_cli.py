"""Offline contract for the CLI surface."""

from __future__ import annotations

import argparse

from app.cli.parsers import build_parser


def _top_level_commands(parser: argparse.ArgumentParser) -> set[str]:
    action = next(
        action for action in parser._actions if isinstance(action, argparse._SubParsersAction)
    )
    return set(action.choices)


def test_cli_exposes_only_current_import_and_model_commands():
    commands = _top_level_commands(build_parser())

    assert commands == {"import-csv", "timesnet"}
    assert {
        "import-api",
        "features-rebuild",
        "import-csv-legacy",
        "import-csv-kat",
        "import-csv-wro",
        "nhits",
        "lstm_ae",
    }.isdisjoint(commands)


def test_import_csv_parser_takes_a_file_path():
    parser = build_parser()

    args = parser.parse_args(["import-csv", "input.csv"])
    assert args.command == "import-csv"
    assert args.input_path == "input.csv"
    assert args.no_resume is False

