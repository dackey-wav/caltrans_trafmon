"""CLI parser for the 5-minute station CSV import."""

import argparse


def register(subparsers: argparse._SubParsersAction) -> None:
    csv_import = subparsers.add_parser(
        "import-csv",
        help="Import a comma-separated 5-minute station CSV into ClickHouse.",
    )
    csv_import.add_argument(
        "input_path",
        metavar="INPUT",
        help="CSV file, or a directory scanned recursively for CSV files.",
    )
    csv_import.add_argument(
        "--state-file",
        default=None,
        help="Importer state file. Unchanged files are skipped on the next run.",
    )
    csv_import.add_argument(
        "--no-resume",
        action="store_true",
        help="Import files even when the state file already records them.",
    )
