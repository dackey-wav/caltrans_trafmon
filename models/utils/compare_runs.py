"""Compare anomaly-model evaluation artifacts between runs."""

from __future__ import annotations

import argparse
import json
from collections.abc import Callable
from pathlib import Path


def compare_runs(model_name: str, runs: list[str], comparison_path_for: Callable[[str], Path]) -> int:
    if len(runs) < 2:
        raise SystemExit("Provide at least 2 runs in --runs")

    payloads: dict[str, dict] = {}
    for run in runs:
        path = comparison_path_for(run)
        if not path.exists():
            raise FileNotFoundError(f"comparison.json not found for run_id={run!r}: {path}")
        with open(path, "r", encoding="utf-8") as f:
            payloads[run] = json.load(f)

    print(f"{model_name} comparison")
    for run in runs:
        p = payloads[run]
        s = p.get("summary", {})
        q = p.get("score_quantiles_by_split", {}).get("test", {})
        print(
            f"- {run}: test_rate={s.get('anomaly_rate', 0.0) * 100:.2f}% "
            f"points={s.get('point_count', 0)} p95={q.get('p95', 0.0):.4f} "
            f"p99={q.get('p99', 0.0):.4f}"
        )

    baseline = payloads[runs[0]].get("summary", {}).get("anomaly_rate", 0.0)
    for run in runs[1:]:
        cur = payloads[run].get("summary", {}).get("anomaly_rate", 0.0)
        delta_pp = (cur - baseline) * 100.0
        print(f"  delta_vs_{runs[0]} -> {run}: {delta_pp:+.2f} pp")
    return 0


def main_from_config(model_name: str, comparison_path_for: Callable[[str], Path]) -> int:
    parser = argparse.ArgumentParser(description=f"Compare {model_name} run diagnostics")
    parser.add_argument("--runs", required=True, help="Comma-separated run ids")
    args = parser.parse_args()
    runs = [x.strip() for x in args.runs.split(",") if x.strip()]
    return compare_runs(model_name, runs, comparison_path_for)
