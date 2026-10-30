"""Dispatch handler for the station CSV import."""

from app.importer.importer_csv.formats.registry import PreflightError, get_format, preflight


def dispatch(args, parser) -> int | None:
    if args.command != "import-csv":
        return None

    spec = get_format()
    state_path = args.state_file or spec.default_state_path
    try:
        preflight(args.input_path, spec)
    except PreflightError as exc:
        parser.error(str(exc))
    print(f"[preflight] format={spec.format_id}")
    print(f"[preflight] destination={spec.destination_resolver()}")
    print(f"[preflight] state={state_path}")
    spec.runner(
        spec.adapter,
        input_path=args.input_path,
        state_path=state_path,
        resume=not args.no_resume,
    )
    return 0
