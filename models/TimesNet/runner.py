"""Thin CLI-facing orchestration for the TimesNet forecasting pipeline."""

from __future__ import annotations

import csv
import gc
import itertools
import time
from collections.abc import Sequence
from pathlib import Path

import numpy as np
import torch

from models.TimesNet import config as tc
from models.TimesNet.data_prep import (
    build_multivariate_frame,
    load_dataset,
    load_meta,
    load_scaler,
    save_dataset,
    save_meta,
    save_scaler,
)
from models.TimesNet.forecast_eval import (
    BASELINES,
    METHODS,
    evaluate_forecast,
    format_summary,
    lead_label,
    load_metrics,
    save_metrics,
    save_predictions,
)
from models.TimesNet.plots import (
    pick_weeks,
    plot_daily_overview,
    plot_error_by_lead,
    plot_example,
    plot_period,
    plot_train_log,
    write_index_html,
)
from models.TimesNet.train_timesnet import (
    FeaturePlan,
    SplitRows,
    _resolve_device,
    load_model,
    load_train_log,
    save_model,
    save_train_log,
    train_forecaster,
)
from models.utils.constants import (
    DEFAULT_CALENDAR_COLUMNS,
    default_pred_len_for,
    default_window_for,
    steps_per_day,
    steps_per_hour,
)
from models.utils.multivariate_grid import extract_active_mask

SWEEP_KEYS = {
    "window": int,
    "pred_len": int,
    "d_model": int,
    "d_ff": int,
    "e_layers": int,
    "top_k": int,
    "num_kernels": int,
    "batch_size": int,
    "learning_rate": float,
    "dropout": float,
    "loss": str,
    "mark_set": str,
    "target": str,
    "weight_decay": float,
    "epochs": int,
    "seed": int,
    "train_stride": int,
}


def run_prepare(
    run_id: str,
    date_from: str | None = None,
    date_to: str | None = None,
    freq: str = tc.DEFAULT_FREQ,
    calendar_columns: Sequence[str] = DEFAULT_CALENDAR_COLUMNS,
) -> None:
    started = time.time()
    print(f"[timesnet:prepare] run_id={run_id} freq={freq} table={tc.FEATURE_TABLE}")
    frame, meta = build_multivariate_frame(
        date_from=date_from,
        date_to=date_to,
        freq=freq,
        calendar_columns=calendar_columns,
    )
    ds_path = save_dataset(frame, tc.dataset_path(run_id))
    meta_path = save_meta(meta, tc.dataset_meta_path(run_id))

    elapsed = time.time() - started
    total_cells = meta.rows * max(meta.n_targets, 1)
    imputed_rate = (meta.imputed_rows / total_cells) if total_cells else 0.0
    print(
        f"[timesnet:prepare] rows={meta.rows} n_targets={meta.n_targets} "
        f"n_context={meta.n_context} imputed_rows={meta.imputed_rows} "
        f"imputed_rate={imputed_rate*100:.2f}% inactive_rows={meta.inactive_rows} "
        f"gap_imputed_rows={meta.gap_imputed_rows} ds_range={meta.ds_min}..{meta.ds_max}"
    )
    print(f"[timesnet:prepare] dataset={ds_path}")
    print(f"[timesnet:prepare] meta={meta_path}")
    print(f"[timesnet:prepare] elapsed_s={elapsed:.2f}")


def run_train(
    run_id: str,
    *,
    name: str | None = None,
    window: int | None = None,
    pred_len: int | None = None,
    epochs: int = tc.DEFAULT_EPOCHS,
    batch_size: int = tc.DEFAULT_BATCH_SIZE,
    learning_rate: float = tc.DEFAULT_LEARNING_RATE,
    patience: int = tc.DEFAULT_PATIENCE,
    val_fraction: float = tc.DEFAULT_VAL_FRACTION,
    test_fraction: float = tc.DEFAULT_TEST_FRACTION,
    d_model: int = tc.DEFAULT_D_MODEL,
    d_ff: int | None = None,
    e_layers: int = tc.DEFAULT_E_LAYERS,
    top_k: int = tc.DEFAULT_TOP_K,
    num_kernels: int = tc.DEFAULT_NUM_KERNELS,
    dropout: float = tc.DEFAULT_DROPOUT,
    seed: int = tc.DEFAULT_SEED,
    loss: str = tc.DEFAULT_LOSS,
    mark_set: str = "full",
    target_mode: str = "absolute",
    train_stride: int = 1,
    use_amp: bool = tc.DEFAULT_USE_AMP,
    weight_decay: float = tc.DEFAULT_WEIGHT_DECAY,
    grad_clip_norm: float = tc.DEFAULT_GRAD_CLIP_NORM,
    log_every: int = tc.DEFAULT_LOG_EVERY,
    oom_retry_enabled: bool = tc.DEFAULT_OOM_RETRY_ENABLED,
    oom_retry_min_batch: int = tc.DEFAULT_OOM_RETRY_MIN_BATCH,
    lr_scheduler_enabled: bool = tc.DEFAULT_LR_SCHEDULER_ENABLED,
    lr_scheduler_patience: int = tc.DEFAULT_LR_SCHEDULER_PATIENCE,
    lr_scheduler_factor: float = tc.DEFAULT_LR_SCHEDULER_FACTOR,
    lr_scheduler_min_lr: float = tc.DEFAULT_LR_SCHEDULER_MIN_LR,
    evaluate: bool = True,
) -> str:
    """Train one experiment on a prepared dataset; return the experiment name."""
    started = time.time()
    frame = load_dataset(tc.dataset_path(run_id))
    meta = load_meta(tc.dataset_meta_path(run_id))
    freq = meta.freq
    window = window or default_window_for(freq)
    pred_len = pred_len or default_pred_len_for(freq)
    d_ff = d_ff or 2 * d_model
    name = name or tc.default_experiment_name(freq, window, pred_len, d_model)
    print(f"[timesnet:train] run_id={run_id} experiment={name}")

    result = train_forecaster(
        frame,
        meta.target_columns,
        meta.context_columns,
        freq=freq,
        window=window,
        pred_len=pred_len,
        epochs=epochs,
        batch_size=batch_size,
        learning_rate=learning_rate,
        patience=patience,
        val_fraction=val_fraction,
        test_fraction=test_fraction,
        d_model=d_model,
        d_ff=d_ff,
        e_layers=e_layers,
        top_k=top_k,
        num_kernels=num_kernels,
        dropout=dropout,
        seed=seed,
        loss=loss,
        mark_set=mark_set,
        target_mode=target_mode,
        train_stride=train_stride,
        use_amp=use_amp,
        weight_decay=weight_decay,
        grad_clip_norm=grad_clip_norm,
        log_every=log_every,
        oom_retry_enabled=oom_retry_enabled,
        oom_retry_min_batch=oom_retry_min_batch,
        lr_scheduler_enabled=lr_scheduler_enabled,
        lr_scheduler_patience=lr_scheduler_patience,
        lr_scheduler_factor=lr_scheduler_factor,
        lr_scheduler_min_lr=lr_scheduler_min_lr,
    )

    log = result.log
    log["run_id"] = run_id
    log["experiment"] = name
    log["train_seconds"] = round(time.time() - started, 1)
    save_model(result.model, tc.model_path(run_id, name))
    save_scaler(result.scaler, columns=result.plan.target_columns, path=tc.scaler_path(run_id, name))
    log_path = save_train_log(log, tc.train_log_path(run_id, name))
    plot_train_log(log, tc.plots_dir(run_id, name) / "train_curve.png")
    print(
        f"[timesnet:train] epochs_run={log['epochs_run']} best_epoch={log['best_epoch']} "
        f"best_val_loss={log['best_val_loss']:.6f} params={log['n_parameters']:,} "
        f"peak_gpu_gb={log['peak_gpu_gb']} train_s={log['train_seconds']}"
    )
    print(f"[timesnet:train] log={log_path}")

    if evaluate:
        run_evaluate(run_id, name)
    return name


def run_evaluate(
    run_id: str,
    name: str,
    *,
    split: str = "test",
    batch_size: int = tc.DEFAULT_EVAL_BATCH_SIZE,
    n_examples: int = 6,
) -> dict:
    started = time.time()
    frame = load_dataset(tc.dataset_path(run_id))
    meta = load_meta(tc.dataset_meta_path(run_id))
    log = load_train_log(tc.train_log_path(run_id, name))
    plan = FeaturePlan.from_dict(log["feature_plan"])
    splits = SplitRows.from_dict(log["splits"])
    scaler, _ = load_scaler(tc.scaler_path(run_id, name))
    model = load_model(tc.model_path(run_id, name))
    freq = meta.freq
    window = model.configs.seq_len
    pred_len = model.configs.pred_len
    rows = splits.rows(split)
    device = _resolve_device()

    result = evaluate_forecast(
        model,
        frame,
        plan,
        scaler,
        freq=freq,
        window=window,
        pred_len=pred_len,
        rows=rows,
        split=split,
        device=device,
        batch_size=batch_size,
        use_amp=bool(log["hparams"].get("use_amp")),
        n_examples=n_examples,
    )
    metrics = result.metrics
    metrics["run_id"] = run_id
    metrics["experiment"] = name
    save_metrics(metrics, tc.metrics_path(run_id, name, split))
    save_predictions(result.predictions, tc.predictions_path(run_id, name, split))

    out_dir = tc.experiment_dir(run_id, name)
    pdir = tc.plots_dir(run_id, name)
    active = extract_active_mask(frame, plan.target_columns)
    channels = plan.channels
    sph = steps_per_hour(freq)
    spd = steps_per_day(freq)
    near_lead = min(sph, pred_len)
    plot_leads = [(near_lead, lead_label(freq, near_lead))]
    if pred_len != near_lead:
        plot_leads.append((pred_len, lead_label(freq, pred_len)))

    overview = [
        plot_error_by_lead(metrics, pdir / f"{split}_error_by_lead.png"),
        plot_daily_overview(
            frame, active, plan.target_columns, channels, result.predictions,
            near_lead, plot_leads[0][1], rows[0], rows[1],
            pdir / f"{split}_daily_overview.png",
        ),
    ]
    week_imgs = []
    for label, a, b in pick_weeks(frame, active, result.predictions, channels[0], near_lead, rows[0], rows[1], freq):
        slug = label.replace(" ", "_")
        week_imgs.append(
            plot_period(
                frame, active, plan.target_columns, channels, result.predictions,
                plot_leads, a, b, 7 * spd,
                f"{split}: {label}", pdir / f"{split}_{slug}.png",
            )
        )
    example_imgs = [
        plot_example(
            frame, active, plan.target_columns, channels, ex,
            history_steps=min(window, spd), steps_day=spd,
            out_path=pdir / f"{split}_example_{k:02d}.png",
        )
        for k, ex in enumerate(result.examples)
    ]
    train_curve = pdir / "train_curve.png"
    if not train_curve.exists():
        plot_train_log(log, train_curve)

    summary = format_summary(metrics)
    header = ["lead", "channel", *METHODS, "skill vs weekly"]
    table = []
    for bucket, row in metrics["by_lead_bucket"].items():
        for ch in channels:
            vals = [row[m][ch]["mae"] for m in METHODS]
            mm, mw = vals[0], vals[METHODS.index("weekly")]
            skill = "" if (mm is None or not mw) else f"{1 - mm / mw:+.1%}"
            table.append([bucket, ch, *("n/a" if v is None else f"{v:.4g}" for v in vals), skill])
    index = write_index_html(
        out_dir,
        f"TimesNet {run_id} / {name}",
        [
            ("Training", [train_curve]),
            ("Errors and overview", overview),
            ("Weeks (actual, model, weekly baseline)", week_imgs),
            ("Forecast examples", example_imgs),
        ],
        summary + "\n\nTable: MAE in original units by lead bucket.",
        table_rows=table,
        table_header=header,
    )
    print(f"[timesnet:eval] {summary}")
    print(f"[timesnet:eval] report={index}")
    print(f"[timesnet:eval] elapsed_s={time.time() - started:.1f}")
    return metrics


def _fmt(value, spec: str = ".4g") -> str:
    if value is None or (isinstance(value, float) and not np.isfinite(value)):
        return ""
    return format(value, spec)


def _experiment_row(run_id: str, exp_dir: Path, split: str) -> dict | None:
    metrics_file = exp_dir / f"forecast_metrics_{split}.json"
    log_file = exp_dir / "train_log.json"
    if not metrics_file.exists() or not log_file.exists():
        return None
    m = load_metrics(metrics_file)
    log = load_train_log(log_file)
    hp = log.get("hparams", {})
    sph = 60 // m["step_minutes"]
    ch = m["channels"]

    def model_mae(channel: str, bucket: str | None = None):
        if channel not in ch:
            return None
        src = m["by_lead_bucket"].get(bucket) if bucket else m["overall"]
        return None if src is None else src["model"][channel]["mae"]

    blocks = m.get("blocks_15min")
    flow_15 = (
        blocks["model"]["flow"]["mae"] if blocks and "flow" in ch else model_mae("flow")
    )
    flow_15_weekly = (
        blocks["weekly"]["flow"]["mae"] if blocks and "flow" in ch
        else (m["overall"]["weekly"]["flow"]["mae"] if "flow" in ch else None)
    )
    def to_hourly(v):
        return None if v is None else v * sph

    return {
        "run_id": run_id,
        "experiment": exp_dir.name,
        "freq": m["freq"],
        "window": m["window"],
        "pred_len": m["pred_len"],
        "d_model": hp.get("d_model"),
        "e_layers": hp.get("e_layers"),
        "loss": hp.get("loss"),
        "mark_set": hp.get("mark_set", "full"),
        "target": hp.get("target_mode", "absolute"),
        "learning_rate": hp.get("learning_rate"),
        "params": log.get("n_parameters"),
        "best_epoch": log.get("best_epoch"),
        "train_s": log.get("train_seconds"),
        "peak_gpu_gb": log.get("peak_gpu_gb"),
        "flow_mae_15min_block": flow_15,
        "flow_mae_15min_block_weekly": flow_15_weekly,
        "flow_mae_vph": to_hourly(model_mae("flow")),
        "flow_mae_vph_0_1h": to_hourly(model_mae("flow", "0-1h")),
        "flow_mae_vph_4_24h": to_hourly(model_mae("flow", "4-24h")),
        "speed_mae_mph": model_mae("speed"),
        "occupancy_mae": model_mae("occupancy"),
        "skill_flow_vs_weekly": m["skill_vs_weekly"].get("flow"),
        "skill_speed_vs_weekly": m["skill_vs_weekly"].get("speed"),
    }


def run_compare(run_ids: Sequence[str], *, split: str = "test", out_csv: Path | None = None) -> list[dict]:
    rows: list[dict] = []
    for run_id in run_ids:
        root = tc.experiments_root(run_id)
        if not root.exists():
            print(f"[timesnet:compare] no experiments under {root}")
            continue
        for exp_dir in sorted(p for p in root.iterdir() if p.is_dir()):
            row = _experiment_row(run_id, exp_dir, split)
            if row is not None:
                rows.append(row)
    if not rows:
        print("[timesnet:compare] nothing to compare")
        return rows
    rows.sort(key=lambda r: (r["flow_mae_15min_block"] is None, r["flow_mae_15min_block"] or 0.0))

    cols = [
        ("experiment", "s"), ("freq", "s"), ("window", "d"), ("pred_len", "d"), ("d_model", "d"),
        ("mark_set", "s"), ("target", "s"), ("learning_rate", "g"), ("best_epoch", "d"), ("params", ","), ("train_s", ".0f"), ("peak_gpu_gb", ".2f"),
        ("flow_mae_15min_block", ".2f"), ("flow_mae_15min_block_weekly", ".2f"),
        ("flow_mae_vph_0_1h", ".1f"), ("flow_mae_vph_4_24h", ".1f"),
        ("speed_mae_mph", ".2f"), ("occupancy_mae", ".4f"),
        ("skill_flow_vs_weekly", "+.1%"), ("skill_speed_vs_weekly", "+.1%"),
    ]
    table = [[c for c, _ in cols]]
    for r in rows:
        table.append([
            str(r[c]) if spec == "s" else _fmt(r[c], spec) for c, spec in cols
        ])
    widths = [max(len(row[i]) for row in table) for i in range(len(cols))]
    for row in table:
        print("  ".join(cell.rjust(w) for cell, w in zip(row, widths, strict=False)))

    out_csv = Path(out_csv) if out_csv else tc.ARTIFACTS_ROOT / f"compare_{split}.csv"
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"[timesnet:compare] baselines in every report: {', '.join(BASELINES)}")
    print(f"[timesnet:compare] csv={out_csv}")
    return rows


def parse_grid(spec: str) -> dict[str, list]:
    """Parse ``"window=96,192;d_model=64,128"`` into a dict of value lists."""
    grid: dict[str, list] = {}
    for part in spec.split(";"):
        part = part.strip()
        if not part:
            continue
        if "=" not in part:
            raise ValueError(f"Bad grid item {part!r}; expected key=v1,v2")
        key, values = part.split("=", 1)
        key = key.strip()
        if key not in SWEEP_KEYS:
            raise ValueError(f"Unsupported grid key {key!r}; allowed: {sorted(SWEEP_KEYS)}")
        cast = SWEEP_KEYS[key]
        grid[key] = [cast(v.strip()) for v in values.split(",") if v.strip()]
    if not grid:
        raise ValueError("Empty grid")
    return grid


def run_sweep(run_id: str, grid: dict[str, list], *, prefix: str = "sweep", **train_kwargs) -> list[str]:
    keys = list(grid)
    combos = list(itertools.product(*(grid[k] for k in keys)))
    print(f"[timesnet:sweep] run_id={run_id} experiments={len(combos)} grid={grid}")
    done: list[str] = []
    for i, combo in enumerate(combos, start=1):
        params = dict(zip(keys, combo, strict=True))
        call_params = {"target_mode" if k == "target" else k: v for k, v in params.items()}
        tag = "_".join(f"{k}{v}" for k, v in params.items())
        name = f"{prefix}_{tag}"
        if (tc.experiments_root(run_id) / name / "forecast_metrics_test.json").exists():
            print(f"[timesnet:sweep] [{i}/{len(combos)}] skip {name}: already evaluated")
            done.append(name)
            continue
        print(f"[timesnet:sweep] [{i}/{len(combos)}] {name}")
        try:
            run_train(run_id, name=name, **{**train_kwargs, **call_params})
            done.append(name)
        except Exception as exc:
            print(f"[timesnet:sweep] FAILED {name}: {type(exc).__name__}: {exc}")
        finally:
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    run_compare([run_id])
    return done
