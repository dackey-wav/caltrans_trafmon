"""TimesNet forecasting pipeline on a small synthetic station series."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

torch = pytest.importorskip("torch")

from models.TimesNet import config as tc  # noqa: E402
from models.TimesNet import runner  # noqa: E402
from models.TimesNet.data_prep import save_dataset, save_meta  # noqa: E402
from models.TimesNet.model import Model, TimesNetConfig  # noqa: E402
from models.TimesNet.train_timesnet import (  # noqa: E402
    FeaturePlan,
    basic_time_marks,
    chronological_split_rows,
    masked_loss,
    plan_features,
)
from models.utils.lazy_windows import forecast_window_starts  # noqa: E402
from models.utils.multivariate_data_prep import MultivariateDatasetMeta  # noqa: E402

pytestmark = pytest.mark.ml_cpu

TARGETS = ["y__flow", "y__speed", "y__occupancy"]
CONTEXT = ["is_weekend", "is_holiday", "is_shopping_sunday"]


def _synthetic_frame(days: int = 24) -> pd.DataFrame:
    idx = pd.date_range("2024-03-04 08:00", periods=days * 96, freq="15min", tz="UTC")
    local = idx.tz_convert("America/Los_Angeles")
    hour = local.hour + local.minute / 60
    weekend = (local.day_of_week >= 5).astype(float)
    rng = np.random.default_rng(0)
    peak = np.exp(-((hour - 8) ** 2) / 4) + np.exp(-((hour - 17) ** 2) / 4)
    flow = 100 + 300 * peak * (1 - 0.5 * weekend) + rng.normal(0, 10, len(idx))
    speed = 65 - 25 * peak * (1 - 0.5 * weekend) + rng.normal(0, 1, len(idx))
    occ = 0.05 + 0.15 * peak + rng.normal(0, 0.005, len(idx))
    frame = pd.DataFrame(
        {"y__flow": flow, "y__speed": speed, "y__occupancy": occ}, index=idx
    )
    frame.index.name = "ds"
    active = np.ones(len(idx), dtype=bool)
    active[500:540] = False
    for ch in ("flow", "speed", "occupancy"):
        frame[f"active__{ch}"] = active
    frame["is_weekend"] = weekend
    frame["is_holiday"] = 0.0
    frame["is_shopping_sunday"] = 0.0
    return frame


def test_model_output_shape():
    cfg = TimesNetConfig(seq_len=32, pred_len=8, n_mark=7, d_model=8, d_ff=16, e_layers=1, top_k=2, num_kernels=2)
    model = Model(cfg)
    out = model(torch.randn(4, 32, 3), torch.randn(4, 40, 7))
    assert out.shape == (4, 8, 3)


def test_window_starts_keep_horizon_inside_split():
    active = np.ones((200, 3), dtype=bool)
    active[150:170] = False
    starts = forecast_window_starts(200, 20, 10, target_lo=100, target_hi=200, active=active)
    assert starts.min() + 20 >= 100
    assert starts.max() + 30 <= 200
    for s in starts:
        assert active[s + 20 : s + 30].any()
    assert not any(s + 20 >= 150 and s + 30 <= 170 for s in starts)


def test_split_snaps_to_days():
    s = chronological_split_rows(96 * 30, "15min", 0.15, 0.15)
    assert s.val[0] % 96 == 0 and s.test[0] % 96 == 0
    assert s.train[1] == s.val[0] and s.val[1] == s.test[0] and s.test[1] == 96 * 30


def test_masked_loss_ignores_inactive():
    pred = torch.zeros(1, 2, 1)
    target = torch.tensor([[[1.0], [100.0]]])
    mask = torch.tensor([[[1.0], [0.0]]])
    assert float(masked_loss(pred, target, mask, "mae")) == pytest.approx(1.0)


def test_time_marks_use_local_clock():
    idx = pd.DatetimeIndex(["2024-07-01 15:00"], tz="UTC")
    marks = basic_time_marks(idx)
    assert marks[0, 3] == pytest.approx(8 / 23 - 0.5)


def test_compact_marks_drop_calendar_date():
    frame = _synthetic_frame(days=12)
    frame["month_sin"] = np.sin(np.arange(len(frame)) / 1000)
    plan = plan_features(frame, TARGETS, [*CONTEXT, "month_sin"], (0, len(frame)), "compact")
    assert plan.basic_marks == ["weekday", "hour", "minute"]
    assert "month_sin" not in plan.mark_columns
    assert plan.n_mark == 3 + len(plan.mark_columns)
    restored = FeaturePlan.from_dict(plan.to_dict())
    assert restored.basic_marks == plan.basic_marks


def test_residual_run_reconstructs_levels(tmp_path, monkeypatch):
    monkeypatch.setattr(tc, "ARTIFACTS_ROOT", tmp_path)
    monkeypatch.setenv("TIMESNET_DEVICE", "cpu")
    frame = _synthetic_frame(days=30)
    run_id = "residual"
    save_dataset(frame, tc.dataset_path(run_id))
    save_meta(
        MultivariateDatasetMeta(
            target_columns=TARGETS, context_columns=CONTEXT, freq="15min",
            date_from=None, date_to=None, rows=len(frame), n_targets=3, n_context=len(CONTEXT),
            ds_min=str(frame.index.min()), ds_max=str(frame.index.max()), imputed_rows=0,
        ),
        tc.dataset_meta_path(run_id),
    )
    runner.run_train(
        run_id, name="resid", window=96, pred_len=16, epochs=1, batch_size=64,
        d_model=8, e_layers=1, top_k=2, num_kernels=2, train_stride=8,
        target_mode="weekly_residual",
    )
    log = runner.load_train_log(tc.train_log_path(run_id, "resid"))
    assert log["hparams"]["target_mode"] == "weekly_residual"
    assert log["feature_plan"]["lag_steps"] == 96 * 7
    metrics = runner.load_metrics(tc.metrics_path(run_id, "resid"))
    assert metrics["overall"]["model"]["flow"]["mae"] > 0


def test_parse_grid():
    grid = runner.parse_grid("window=96,192; d_model=64 ;loss=mae")
    assert grid == {"window": [96, 192], "d_model": [64], "loss": ["mae"]}
    with pytest.raises(ValueError):
        runner.parse_grid("bogus=1")


def test_train_evaluate_compare_end_to_end(tmp_path, monkeypatch):
    monkeypatch.setattr(tc, "ARTIFACTS_ROOT", tmp_path)
    monkeypatch.setenv("TIMESNET_DEVICE", "cpu")
    frame = _synthetic_frame()
    run_id = "synthetic"
    save_dataset(frame, tc.dataset_path(run_id))
    save_meta(
        MultivariateDatasetMeta(
            target_columns=TARGETS,
            context_columns=CONTEXT,
            freq="15min",
            date_from=None,
            date_to=None,
            rows=len(frame),
            n_targets=3,
            n_context=len(CONTEXT),
            ds_min=str(frame.index.min()),
            ds_max=str(frame.index.max()),
            imputed_rows=0,
        ),
        tc.dataset_meta_path(run_id),
    )

    name = runner.run_train(
        run_id,
        name="tiny",
        window=96,
        pred_len=16,
        epochs=2,
        batch_size=64,
        d_model=8,
        e_layers=1,
        top_k=2,
        num_kernels=2,
        train_stride=4,
    )
    exp = tc.experiment_dir(run_id, name)
    for fname in ("model.pt", "scaler.json", "train_log.json", "forecast_metrics_test.json",
                  "forecast_test.parquet", "index.html"):
        assert (exp / fname).exists(), fname

    metrics = runner.load_metrics(exp / "forecast_metrics_test.json")
    assert set(metrics["overall"]) == {"model", "persistence", "daily", "weekly"}
    assert len(metrics["by_lead"]["model"]["flow"]["mae"]) == 16
    assert metrics["overall"]["weekly"]["flow"]["mae"] > 0
    log = runner.load_train_log(exp / "train_log.json")
    assert "is_shopping_sunday" in log["feature_plan"]["dropped_constant_columns"]

    plots = list((exp / "plots").glob("*.png"))
    assert any(p.name.startswith("test_example_") for p in plots)
    assert (exp / "plots" / "test_whole_split.png").exists()
    assert (exp / "plots" / "test_error_by_lead.png").exists()

    rows = runner.run_compare([run_id])
    assert len(rows) == 1 and rows[0]["experiment"] == "tiny"
