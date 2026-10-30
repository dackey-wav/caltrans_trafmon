"""Thin CLI-facing orchestration for the TimesNet pipeline."""

from __future__ import annotations
import numpy as np
import json
import time
from typing import Sequence

import pandas as pd

from models.TimesNet import config as tc
from models.TimesNet.data_prep import (
    build_multivariate_class_frame,
    build_multivariate_frame,
    fit_robust_scaler,
    load_dataset,
    load_meta,
    load_scaler,
    save_dataset,
    save_meta,
    save_scaler,
)
from models.TimesNet.eval import build_anomaly_report, build_metrics_report, save_json
from models.TimesNet.plots import (
    plot_actual_with_anomalies,
    plot_location_class_summary,
    plot_reconstruction_vs_actual,
    plot_score_timeline,
    plot_train_log,
)
from models.utils.multivariate_grid import (
    extract_active_mask,
    fit_robust_scaler_masked,
    per_location_split_masks,
)
from models.TimesNet.domain_filter import apply_domain_filter, domain_filter_summary
from models.utils.runner_common import filter_by_date_range, resolve_threshold_profile
from models.utils.training import runtime_loader_defaults
from models.TimesNet.train_timesnet import (
    TrainSplits,
    _resolve_device,
    annotate_anomalies,
    annotate_combined,
    calibrate_thresholds,
    compute_scores,
    load_model,
    save_model,
    save_scores,
    save_thresholds,
    save_train_log,
    train_timesnet,
)


def run_prepare(
    run_id: str,
    locations: Sequence[str],
    date_from: str | None = None,
    date_to: str | None = None,
    freq: str = tc.DEFAULT_FREQ,
    calendar_columns: Sequence[str] = tc.DEFAULT_CALENDAR_COLUMNS,
    feature_table: str | None = None,
) -> None:
    started = time.time()
    table = feature_table or tc.FEATURE_TABLE
    print(
        f"[timesnet:prepare] run_id={run_id} locations={list(locations)} "
        f"freq={freq} feature_table={table}"
    )
    frame_kwargs: dict = {
        "locations": locations,
        "date_from": date_from,
        "date_to": date_to,
        "freq": freq,
        "calendar_columns": calendar_columns,
    }
    if feature_table is not None:
        frame_kwargs["feature_table"] = feature_table
    frame, meta = build_multivariate_frame(**frame_kwargs)
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


def run_prepare_class(
    run_id: str,
    locations: Sequence[str],
    date_from: str | None = None,
    date_to: str | None = None,
    freq: str = tc.DEFAULT_FREQ,
    calendar_columns: Sequence[str] = tc.DEFAULT_CALENDAR_COLUMNS,
    min_nonzero_hours_per_series: int = 0,
) -> None:
    started = time.time()
    print(f"[timesnet:prepare-class] run_id={run_id} locations={list(locations)} freq={freq}")
    frame, meta = build_multivariate_class_frame(
        locations=locations,
        date_from=date_from,
        date_to=date_to,
        freq=freq,
        calendar_columns=calendar_columns,
        min_nonzero_hours_per_series=min_nonzero_hours_per_series,
    )
    ds_path = save_dataset(frame, tc.dataset_path(run_id))
    meta_path = save_meta(meta, tc.dataset_meta_path(run_id))

    elapsed = time.time() - started
    total_cells = meta.rows * max(meta.n_targets, 1)
    imputed_rate = (meta.imputed_rows / total_cells) if total_cells else 0.0
    print(
        f"[timesnet:prepare-class] rows={meta.rows} n_targets={meta.n_targets} "
        f"n_context={meta.n_context} imputed_rows={meta.imputed_rows} "
        f"imputed_rate={imputed_rate*100:.2f}% inactive_rows={meta.inactive_rows} "
        f"gap_imputed_rows={meta.gap_imputed_rows} ds_range={meta.ds_min}..{meta.ds_max}"
    )
    print(f"[timesnet:prepare-class] dataset={ds_path}")
    print(f"[timesnet:prepare-class] meta={meta_path}")
    print(f"[timesnet:prepare-class] elapsed_s={elapsed:.2f}")


def run_train(
    run_id: str,
    *,
    window: int | None = None,
    epochs: int = tc.DEFAULT_EPOCHS,
    batch_size: int = tc.DEFAULT_BATCH_SIZE,
    learning_rate: float = tc.DEFAULT_LEARNING_RATE,
    patience: int = tc.DEFAULT_PATIENCE,
    val_fraction: float = tc.DEFAULT_VAL_FRACTION,
    test_fraction: float = tc.DEFAULT_TEST_FRACTION,
    d_model: int = tc.DEFAULT_D_MODEL,
    d_ff: int = tc.DEFAULT_D_FF,
    e_layers: int = tc.DEFAULT_E_LAYERS,
    top_k: int = tc.DEFAULT_TOP_K,
    num_kernels: int = tc.DEFAULT_NUM_KERNELS,
    dropout: float = tc.DEFAULT_DROPOUT,
    seed: int = tc.DEFAULT_SEED,
    num_workers: int | None = None,
    pin_memory: bool | None = None,
    persistent_workers: bool = tc.DEFAULT_PERSISTENT_WORKERS,
    use_amp: bool = tc.DEFAULT_USE_AMP,
    use_rmse_loss: bool = tc.DEFAULT_USE_RMSE_LOSS,
    weight_decay: float = tc.DEFAULT_WEIGHT_DECAY,
    grad_clip_norm: float = tc.DEFAULT_GRAD_CLIP_NORM,
    log_every: int = tc.DEFAULT_LOG_EVERY,
    oom_retry_enabled: bool = tc.DEFAULT_OOM_RETRY_ENABLED,
    oom_retry_min_batch: int = tc.DEFAULT_OOM_RETRY_MIN_BATCH,
    lr_scheduler_enabled: bool = tc.DEFAULT_LR_SCHEDULER_ENABLED,
    lr_scheduler_patience: int = tc.DEFAULT_LR_SCHEDULER_PATIENCE,
    lr_scheduler_factor: float = tc.DEFAULT_LR_SCHEDULER_FACTOR,
    lr_scheduler_min_lr: float = tc.DEFAULT_LR_SCHEDULER_MIN_LR,
) -> None:
    started = time.time()
    print(f"[timesnet:train] run_id={run_id}")

    frame = load_dataset(tc.dataset_path(run_id))
    meta = load_meta(tc.dataset_meta_path(run_id))

    if window is None:
        window = tc.default_window_for(meta.freq)

    column_order = list(meta.target_columns) + list(meta.context_columns)
    values = frame[column_order].to_numpy(dtype=np.float32)
    active = extract_active_mask(frame, meta.target_columns)
    train_m, _, _ = per_location_split_masks(
        active, val_fraction=val_fraction, test_fraction=test_fraction
    )
    n_tgt = len(meta.target_columns)
    n_ctx = len(meta.context_columns)
    fit_mask = np.zeros((len(frame), len(column_order)), dtype=bool)
    fit_mask[:, :n_tgt] = train_m & active
    if n_ctx:
        row_train = train_m.any(axis=1)
        fit_mask[:, n_tgt:] = row_train[:, None]
    scaler = fit_robust_scaler_masked(values, fit_mask)
    save_scaler(scaler, columns=column_order, path=tc.scaler_path(run_id))

    device = _resolve_device().type
    num_workers, pin_memory = runtime_loader_defaults(
        device,
        num_workers,
        pin_memory,
        default_gpu_num_workers=tc.DEFAULT_GPU_NUM_WORKERS,
        default_cpu_num_workers=tc.DEFAULT_CPU_NUM_WORKERS,
        default_gpu_pin_memory=tc.DEFAULT_GPU_PIN_MEMORY,
        default_cpu_pin_memory=tc.DEFAULT_CPU_PIN_MEMORY,
    )

    model, log, _splits = train_timesnet(
        frame=frame,
        target_columns=meta.target_columns,
        context_columns=meta.context_columns,
        scaler=scaler,
        window=window,
        data_freq=meta.freq,
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
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=persistent_workers,
        use_amp=use_amp,
        use_rmse_loss=use_rmse_loss,
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

    model_path = save_model(model, tc.model_path(run_id))
    log_path = save_train_log(log, tc.train_log_path(run_id))
    plot_train_log(log, tc.plots_dir(run_id) / "train_curve.png")

    elapsed = time.time() - started
    print(
        f"[timesnet:train] epochs_run={log['epochs_run']} best_val_loss={log['best_val_loss']:.6f} "
        f"device={log['device']} window={window} batch={log['hparams'].get('batch_size')}"
    )
    print(f"[timesnet:train] model={model_path}")
    print(f"[timesnet:train] log={log_path}")
    print(f"[timesnet:train] elapsed_s={elapsed:.2f}")


def run_gt_eval(
    run_id: str,
    *,
    eval_split: str = "test",
    gt_csv: str | None = None,
) -> None:
    from models.data.gt_anomalies.gt_eval import run_gt_eval_for_artifacts

    run_gt_eval_for_artifacts(
        tc.run_dir(run_id),
        gt_csv=gt_csv,
        eval_split=eval_split,
        output_path=tc.gt_eval_path(run_id),
    )


def run_eval(
    run_id: str,
    *,
    threshold_method: str = tc.DEFAULT_THRESHOLD_METHOD,
    threshold: float = tc.DEFAULT_ANOMALY_THRESHOLD,
    quantile: float = tc.DEFAULT_THRESHOLD_QUANTILE,
    threshold_profile: str | None = None,
    combined_robust_z: float | None = None,
    domain_filter: bool = True,
    score_smoothing_span: int | None = tc.DEFAULT_SCORE_SMOOTHING_SPAN,
    eval_split: str = "test",
    max_records: int | None = 5000,
    plot_top: int | None = None,
    include_train_in_score_plot: bool = False,
    zoom_to_eval: bool = True,
    include_full_span_plot: bool = True,
    date_from: str | None = None,
    date_to: str | None = None,
    gt_eval: bool = False,
    gt_csv: str | None = None,
) -> None:
    started = time.time()
    range_label = f" range={date_from or '*'}..{date_to or '*'}"
    print(f"[timesnet:eval] run_id={run_id}{range_label if date_from or date_to else ''}")

    frame = load_dataset(tc.dataset_path(run_id))
    meta = load_meta(tc.dataset_meta_path(run_id))
    scaler, _ = load_scaler(tc.scaler_path(run_id))
    model = load_model(tc.model_path(run_id), data_freq=meta.freq)

    with open(tc.train_log_path(run_id), "r", encoding="utf-8") as f:
        log = json.load(f)
    hp = log["hparams"]
    splits = TrainSplits(
        train=tuple(log["splits"]["train"]),
        val=tuple(log["splits"]["val"]),
        test=tuple(log["splits"]["test"]),
        window=int(hp["window"]),
    )

    scores = compute_scores(
        model=model,
        frame=frame,
        target_columns=meta.target_columns,
        context_columns=meta.context_columns,
        scaler=scaler,
        splits=splits,
        data_freq=meta.freq,
        batch_size=int(hp["batch_size"]),
        num_workers=int(hp.get("num_workers", 0)),
        pin_memory=bool(hp.get("pin_memory", False)),
        persistent_workers=bool(hp.get("persistent_workers", True)),
        use_amp=bool(hp.get("use_amp", False)),
        score_smoothing_span=score_smoothing_span,
        val_fraction=float(hp.get("val_fraction", tc.DEFAULT_VAL_FRACTION)),
        test_fraction=float(hp.get("test_fraction", tc.DEFAULT_TEST_FRACTION)),
    )
    save_scores(scores, tc.scores_path(run_id))

    threshold_method, threshold, quantile = resolve_threshold_profile(
        threshold_profile,
        threshold_method,
        threshold,
        quantile,
        preset_resolver=tc.threshold_preset,
    )
    pct_quantile = (
        float(quantile)
        if threshold_profile and threshold_method == "percentile"
        else tc.EVAL_PERCENTILE_QUANTILE
    )
    robust_z_gate = float(
        combined_robust_z
        if combined_robust_z is not None
        else (threshold if threshold_profile and threshold_method == "robust_z" else tc.EVAL_COMBINED_ROBUST_Z)
    )

    thresholds_pct = calibrate_thresholds(
        scores,
        method="percentile",
        quantile=pct_quantile,
        calibration_split="train",
    )
    thresholds_robust = calibrate_thresholds(
        scores,
        method="robust_z",
        threshold=robust_z_gate,
        calibration_split="train",
    )
    save_thresholds(thresholds_pct, tc.thresholds_percentile_path(run_id))
    save_thresholds(thresholds_robust, tc.thresholds_robust_path(run_id))
    save_thresholds(thresholds_pct, tc.thresholds_path(run_id))

    annotated_stat = annotate_combined(
        scores,
        thresholds_percentile=thresholds_pct,
        thresholds_robust=thresholds_robust,
        robust_z_confirm=robust_z_gate,
    )
    annotated = apply_domain_filter(annotated_stat, enabled=domain_filter)

    plot_frame, plot_scores = filter_by_date_range(
        frame,
        annotated,
        date_from=date_from,
        date_to=date_to,
    )
    _, plot_scores_stat = filter_by_date_range(
        frame,
        annotated_stat,
        date_from=date_from,
        date_to=date_to,
    )
    if plot_scores.empty and (date_from or date_to):
        raise ValueError(
            f"No score rows in eval date range {date_from!r}..{date_to!r}. "
            "Check --date-from/--date-to against the prepared dataset span."
        )

    report_stat = build_anomaly_report(
        annotated_scores=plot_scores_stat,
        thresholds=thresholds_pct,
        run_id=run_id,
        scope="timesnet",
        eval_split=eval_split,
        max_records=max_records,
        date_from=date_from,
        date_to=date_to,
        mode="combined",
        domain_filter_enabled=False,
    )
    save_json(report_stat, tc.anomalies_stat_path(run_id))

    anomaly_report = build_anomaly_report(
        annotated_scores=plot_scores,
        thresholds=thresholds_pct,
        run_id=run_id,
        scope="timesnet",
        eval_split=eval_split,
        max_records=max_records,
        date_from=date_from,
        date_to=date_to,
        mode="domain" if domain_filter else "combined",
        domain_filter_enabled=domain_filter,
    )
    save_json(anomaly_report, tc.anomalies_path(run_id))

    eval_slice = plot_scores[plot_scores["split"] == eval_split]
    n_stat = int(report_stat["summary"]["anomaly_count"])
    n_domain = int(anomaly_report["summary"]["anomaly_count"])
    filtered = n_stat - n_domain

    metrics_report = build_metrics_report(
        annotated_scores=plot_scores,
        extra={
            "model": "timesnet",
            "freq": meta.freq,
            "window": splits.window,
            "n_targets": meta.n_targets,
            "n_context": meta.n_context,
            "threshold_method": "combined",
            "threshold_profile": threshold_profile,
            "percentile_quantile": pct_quantile,
            "combined_robust_z": robust_z_gate,
            "domain_filter": domain_filter,
            "stat_summary": report_stat["summary"],
            "domain_summary": anomaly_report["summary"],
            "date_from": date_from,
            "date_to": date_to,
        },
    )
    save_json(metrics_report, tc.metrics_path(run_id))

    split_map = {"val": ("val",), "test": ("test",)}
    plot_score_timeline(
        plot_scores,
        tc.plots_dir(run_id) / "score",
        max_series=plot_top,
        splits_to_show=split_map.get(eval_split, ("test",)),
        include_train=include_train_in_score_plot,
    )

    plot_actual_with_anomalies(
        frame=plot_frame,
        target_columns=list(meta.target_columns),
        annotated_scores=plot_scores,
        output_dir=tc.anomaly_plots_dir(run_id),
        max_series=plot_top,
        eval_split=eval_split,
        zoom_to_eval=zoom_to_eval,
        include_full_span_copy=include_full_span_plot,
    )

    recon_paths = plot_reconstruction_vs_actual(
        frame=plot_frame,
        target_columns=list(meta.target_columns),
        annotated_scores=plot_scores,
        output_dir=tc.reconstruction_plots_dir(run_id),
        max_series=plot_top,
        eval_split=eval_split,
        zoom_to_eval=zoom_to_eval,
        include_full_span_copy=include_full_span_plot,
    )
    if recon_paths:
        print(
            f"[timesnet:eval] reconstruction_plots={len(recon_paths)} → "
            f"{tc.reconstruction_plots_dir(run_id)}"
        )

    has_class_targets = any("__cls" in c for c in meta.target_columns)
    if has_class_targets:
        cls_paths = plot_location_class_summary(
            frame=plot_frame,
            annotated_scores=plot_scores,
            target_columns=list(meta.target_columns),
            output_dir=tc.class_summary_plots_dir(run_id),
            eval_split=eval_split,
            zoom_to_eval=zoom_to_eval,
            max_locations=plot_top,
        )
        if cls_paths:
            print(
                f"[timesnet:eval] class_summary_plots={len(cls_paths)} → "
                f"{tc.class_summary_plots_dir(run_id)}"
            )

    per_series = anomaly_report.get("per_series", [])
    top_series = sorted(per_series, key=lambda r: r.get("rate", 0.0), reverse=True)[:10]
    comparison_payload = {
        "run_id": run_id,
        "eval_split": eval_split,
        "threshold": {
            "method": "combined",
            "profile": threshold_profile,
            "percentile_quantile": pct_quantile,
            "combined_robust_z": robust_z_gate,
            "domain_filter": domain_filter,
        },
        "summary_stat": report_stat["summary"],
        "summary_domain": anomaly_report["summary"],
        "top_series_by_rate": top_series,
        "score_quantiles_by_split": {
            row["split"]: {
                "p95": row["score_p95"],
                "p99": row["score_p99"],
                "anomaly_rate": row["anomaly_rate"],
            }
            for row in metrics_report["splits"]
        },
    }
    save_json(comparison_payload, tc.comparison_path(run_id))

    elapsed = time.time() - started
    summary = anomaly_report["summary"]
    dom = domain_filter_summary(eval_slice) if domain_filter else {}
    print(
        f"[timesnet:eval] eval_split={eval_split} combined "
        f"(q={pct_quantile}, z>={robust_z_gate}) "
        f"profile={threshold_profile or '-'}"
    )
    print(
        f"[timesnet:eval] stat_flagged={n_stat} domain_flagged={n_domain} "
        f"filtered={filtered} of {summary['point_count']} eval points "
        f"({summary['anomaly_rate'] * 100:.2f}%) series={summary['series_count']}"
    )
    if domain_filter and dom.get("by_kind"):
        print(f"[timesnet:eval] domain_kinds={dom['by_kind']}")
    for row in metrics_report["splits"]:
        print(
            f"[timesnet:eval] split={row['split']:<5} points={row['points']} "
            f"anomaly_rate={row['anomaly_rate'] * 100:.2f}% "
            f"score_p95={row['score_p95']:.4f} score_p99={row['score_p99']:.4f}"
        )
    print(f"[timesnet:eval] anomalies={tc.anomalies_path(run_id)}")
    print(f"[timesnet:eval] anomalies_stat={tc.anomalies_stat_path(run_id)}")
    print(f"[timesnet:eval] scores={tc.scores_path(run_id)}")
    print(f"[timesnet:eval] comparison={tc.comparison_path(run_id)}")
    if gt_eval:
        try:
            run_gt_eval(run_id, eval_split=eval_split, gt_csv=gt_csv)
        except FileNotFoundError as exc:
            print(f"[timesnet:eval] gt_eval skipped: {exc}")
    print(f"[timesnet:eval] elapsed_s={elapsed:.2f}")
