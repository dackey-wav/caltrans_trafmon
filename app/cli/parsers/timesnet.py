"""TimesNet CLI parser registration."""

import argparse

from models.TimesNet import config as tc


def register(subparsers: argparse._SubParsersAction) -> None:
    timesnet = subparsers.add_parser("timesnet", help="TimesNet anomaly-detection tools")
    timesnet_sub = timesnet.add_subparsers(dest="timesnet_command", required=True)

    tn_prepare = timesnet_sub.add_parser(
        "prepare",
        help="Build a multivariate wide-format dataset for TimesNet.",
    )
    tn_prepare.add_argument("--locations", required=True)
    tn_prepare.add_argument("--date-from", default=None)
    tn_prepare.add_argument("--date-to", default=None)
    tn_prepare.add_argument("--run-id", default=None)
    tn_prepare.add_argument(
        "--freq",
        choices=list(tc.SUPPORTED_FREQS),
        default=tc.DEFAULT_FREQ,
    )
    tn_prepare.add_argument(
        "--feature-table",
        default=tc.FEATURE_TABLE,
        help=(f"ClickHouse feature table for counts (default: {tc.FEATURE_TABLE})."),
    )

    tn_prepare_class = timesnet_sub.add_parser(
        "prepare-class",
        help="Build class-level multivariate dataset for TimesNet.",
    )
    tn_prepare_class.add_argument("--locations", required=True)
    tn_prepare_class.add_argument("--date-from", default=None)
    tn_prepare_class.add_argument("--date-to", default=None)
    tn_prepare_class.add_argument("--run-id", default=None)
    tn_prepare_class.add_argument(
        "--min-nonzero-hours-per-series",
        type=int,
        default=24,
    )
    tn_prepare_class.add_argument(
        "--freq",
        choices=list(tc.SUPPORTED_FREQS),
        default=tc.DEFAULT_FREQ,
    )

    tn_train = timesnet_sub.add_parser("train", help="Fit TimesNet on the prepared dataset.")
    tn_train.add_argument("--run-id", required=True)
    tn_train.add_argument("--window", type=int, default=None)
    tn_train.add_argument("--epochs", type=int, default=tc.DEFAULT_EPOCHS)
    tn_train.add_argument("--batch-size", type=int, default=tc.DEFAULT_BATCH_SIZE)
    tn_train.add_argument("--learning-rate", type=float, default=tc.DEFAULT_LEARNING_RATE)
    tn_train.add_argument("--patience", type=int, default=tc.DEFAULT_PATIENCE)
    tn_train.add_argument("--val-fraction", type=float, default=tc.DEFAULT_VAL_FRACTION)
    tn_train.add_argument("--test-fraction", type=float, default=tc.DEFAULT_TEST_FRACTION)
    tn_train.add_argument("--d-model", type=int, default=tc.DEFAULT_D_MODEL)
    tn_train.add_argument("--d-ff", type=int, default=tc.DEFAULT_D_FF)
    tn_train.add_argument("--e-layers", type=int, default=tc.DEFAULT_E_LAYERS)
    tn_train.add_argument("--top-k", type=int, default=tc.DEFAULT_TOP_K)
    tn_train.add_argument("--num-kernels", type=int, default=tc.DEFAULT_NUM_KERNELS)
    tn_train.add_argument("--dropout", type=float, default=tc.DEFAULT_DROPOUT)
    tn_train.add_argument("--seed", type=int, default=tc.DEFAULT_SEED)
    tn_train.add_argument("--num-workers", type=int, default=None)
    tn_train.add_argument("--pin-memory", action="store_true", default=None)
    tn_train.add_argument("--no-pin-memory", action="store_true", default=False)
    tn_train.add_argument("--no-persistent-workers", action="store_true", default=False)
    tn_train.add_argument("--use-amp", action="store_true", default=tc.DEFAULT_USE_AMP)
    tn_train.add_argument("--use-rmse-loss", action="store_true", default=tc.DEFAULT_USE_RMSE_LOSS)
    tn_train.add_argument("--no-rmse-loss", action="store_true", default=False)
    tn_train.add_argument("--weight-decay", type=float, default=tc.DEFAULT_WEIGHT_DECAY)
    tn_train.add_argument("--grad-clip-norm", type=float, default=tc.DEFAULT_GRAD_CLIP_NORM)
    tn_train.add_argument("--log-every", type=int, default=tc.DEFAULT_LOG_EVERY)
    tn_train.add_argument("--no-oom-retry", action="store_true", default=False)
    tn_train.add_argument("--oom-retry-min-batch", type=int, default=tc.DEFAULT_OOM_RETRY_MIN_BATCH)
    tn_train.add_argument("--no-lr-scheduler", action="store_true", default=False)
    tn_train.add_argument(
        "--lr-scheduler-patience", type=int, default=tc.DEFAULT_LR_SCHEDULER_PATIENCE
    )
    tn_train.add_argument(
        "--lr-scheduler-factor", type=float, default=tc.DEFAULT_LR_SCHEDULER_FACTOR
    )

    tn_eval = timesnet_sub.add_parser(
        "eval", help="Score timeline and emit anomalies.json + plots."
    )
    tn_eval.add_argument("--run-id", required=True)
    tn_eval.add_argument(
        "--threshold-profile",
        choices=sorted(tc.THRESHOLD_PRESETS.keys()),
        default=None,
    )
    tn_eval.add_argument(
        "--threshold-method",
        choices=list(tc.SUPPORTED_THRESHOLD_METHODS),
        default=tc.DEFAULT_THRESHOLD_METHOD,
    )
    tn_eval.add_argument("--threshold", type=float, default=tc.DEFAULT_ANOMALY_THRESHOLD)
    tn_eval.add_argument("--quantile", type=float, default=tc.DEFAULT_THRESHOLD_QUANTILE)
    tn_eval.add_argument("--eval-split", choices=("val", "test"), default="test")
    tn_eval.add_argument(
        "--date-from",
        default=None,
        help="Inclusive UTC lower bound for reports/plots (e.g. 2024-06-01 00:00:00).",
    )
    tn_eval.add_argument(
        "--date-to", default=None, help="Exclusive UTC upper bound for reports/plots."
    )
    tn_eval.add_argument("--max-records", type=int, default=5000)
    tn_eval.add_argument(
        "--plot-top",
        type=int,
        default=0,
        help="Max locations/series to plot (0 = all locations).",
    )
    tn_eval.add_argument("--include-train-in-score-plot", action="store_true", default=False)
    tn_eval.add_argument("--score-smoothing-span", type=int, default=None)
    tn_eval.add_argument("--no-zoom-to-eval", action="store_true", default=False)
    tn_eval.add_argument(
        "--no-full-span-plot",
        action="store_true",
        default=False,
        help="Disable saving full-span companion anomaly/recon plots.",
    )
    tn_eval.add_argument(
        "--combined-robust-z",
        type=float,
        default=None,
        help=(
            "Robust-z confirmation gate for combined statistical filter "
            f"(default: config EVAL_COMBINED_ROBUST_Z={tc.EVAL_COMBINED_ROBUST_Z})."
        ),
    )
    tn_eval.add_argument(
        "--gt-eval",
        action="store_true",
        help="Write gt_eval.json after scoring (or use gt-eval for GT-only).",
    )
    tn_eval.add_argument(
        "--gt-csv",
        default=None,
        help="Override path to ground-truth events CSV for --gt-eval.",
    )
    tn_eval.add_argument(
        "--no-domain-filter",
        action="store_true",
        default=False,
        help="Skip domain gates (traffic intensity / class share / lane shift).",
    )

    tn_gt_eval = timesnet_sub.add_parser(
        "gt-eval",
        help="GT recall only — uses existing scores + anomaly JSON from a prior eval.",
    )
    tn_gt_eval.add_argument("--run-id", required=True)
    tn_gt_eval.add_argument("--eval-split", choices=("val", "test"), default="test")
    tn_gt_eval.add_argument("--gt-csv", default=None)
