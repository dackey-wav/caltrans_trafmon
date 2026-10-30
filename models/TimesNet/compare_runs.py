"""Compare TimesNet evaluation artifacts between runs."""

from __future__ import annotations

from models.TimesNet import config as tc
from models.utils.compare_runs import main_from_config

if __name__ == "__main__":
    raise SystemExit(main_from_config("TimesNet", tc.comparison_path))
