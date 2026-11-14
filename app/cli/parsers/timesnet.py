"""TimesNet CLI parser registration."""

import argparse

from models.TimesNet import config as tc
from models.utils.constants import SUPPORTED_FREQS


def _add_train_args(p: argparse.ArgumentParser, *, sweep: bool) -> None:
    p.add_argument("--run-id", required=True, help="Prepared dataset run id.")
    if not sweep:
        p.add_argument("--name", default=None, help="Experiment name (default: built from params).")
        p.add_argument("--window", type=int, default=None, help="Input steps (15min: 192, 5min: 576).")
        p.add_argument("--pred-len", type=int, default=None, help="Horizon steps (15min: 96, 5min: 288).")
        p.add_argument("--d-model", type=int, default=tc.DEFAULT_D_MODEL)
        p.add_argument("--d-ff", type=int, default=None, help="Default: 2 * d_model.")
        p.add_argument("--e-layers", type=int, default=tc.DEFAULT_E_LAYERS)
        p.add_argument("--top-k", type=int, default=tc.DEFAULT_TOP_K)
        p.add_argument("--num-kernels", type=int, default=tc.DEFAULT_NUM_KERNELS)
        p.add_argument("--batch-size", type=int, default=tc.DEFAULT_BATCH_SIZE)
        p.add_argument("--learning-rate", type=float, default=tc.DEFAULT_LEARNING_RATE)
        p.add_argument("--dropout", type=float, default=tc.DEFAULT_DROPOUT)
        p.add_argument("--loss", choices=list(tc.SUPPORTED_LOSSES), default=tc.DEFAULT_LOSS)
        p.add_argument(
            "--mark-set",
            choices=["full", "compact"],
            default="full",
            help="compact drops month, day of month and week of year from the time marks.",
        )
        p.add_argument(
            "--target",
            choices=["absolute", "weekly_residual"],
            default="absolute",
            help="weekly_residual learns the correction to the value one week earlier.",
        )
        p.add_argument("--epochs", type=int, default=tc.DEFAULT_EPOCHS)
        p.add_argument("--seed", type=int, default=tc.DEFAULT_SEED)
        p.add_argument("--train-stride", type=int, default=1, help="Use every N-th train window.")
        p.add_argument("--no-eval", action="store_true", default=False, help="Skip test evaluation.")
    p.add_argument("--patience", type=int, default=tc.DEFAULT_PATIENCE)
    p.add_argument("--val-fraction", type=float, default=tc.DEFAULT_VAL_FRACTION)
    p.add_argument("--test-fraction", type=float, default=tc.DEFAULT_TEST_FRACTION)
    p.add_argument("--no-amp", action="store_true", default=False, help="Disable bf16/fp16 autocast.")
    p.add_argument("--weight-decay", type=float, default=tc.DEFAULT_WEIGHT_DECAY)
    p.add_argument("--grad-clip-norm", type=float, default=tc.DEFAULT_GRAD_CLIP_NORM)
    p.add_argument("--log-every", type=int, default=tc.DEFAULT_LOG_EVERY)
    p.add_argument("--no-oom-retry", action="store_true", default=False)
    p.add_argument("--oom-retry-min-batch", type=int, default=tc.DEFAULT_OOM_RETRY_MIN_BATCH)
    p.add_argument("--no-lr-scheduler", action="store_true", default=False)
    p.add_argument("--lr-scheduler-patience", type=int, default=tc.DEFAULT_LR_SCHEDULER_PATIENCE)
    p.add_argument("--lr-scheduler-factor", type=float, default=tc.DEFAULT_LR_SCHEDULER_FACTOR)


def register(subparsers: argparse._SubParsersAction) -> None:
    timesnet = subparsers.add_parser("timesnet", help="TimesNet traffic forecasting")
    timesnet_sub = timesnet.add_subparsers(dest="timesnet_command", required=True)

    tn_prepare = timesnet_sub.add_parser(
        "prepare",
        help="Build the station dataset (targets, activity masks, calendar).",
    )
    tn_prepare.add_argument("--date-from", default=None)
    tn_prepare.add_argument("--date-to", default=None)
    tn_prepare.add_argument("--run-id", default=None)
    tn_prepare.add_argument("--freq", choices=list(SUPPORTED_FREQS), default=tc.DEFAULT_FREQ)

    tn_train = timesnet_sub.add_parser(
        "train", help="Train one forecaster on a prepared dataset, then evaluate on test."
    )
    _add_train_args(tn_train, sweep=False)

    tn_eval = timesnet_sub.add_parser(
        "evaluate", help="Score a trained experiment against baselines and draw plots."
    )
    tn_eval.add_argument("--run-id", required=True)
    tn_eval.add_argument("--name", required=True)
    tn_eval.add_argument("--split", choices=["val", "test"], default="test")
    tn_eval.add_argument("--batch-size", type=int, default=tc.DEFAULT_EVAL_BATCH_SIZE)
    tn_eval.add_argument("--examples", type=int, default=6)

    tn_compare = timesnet_sub.add_parser(
        "compare", help="Table of all evaluated experiments for one or more datasets."
    )
    tn_compare.add_argument("--run-id", nargs="+", required=True)
    tn_compare.add_argument("--split", choices=["val", "test"], default="test")
    tn_compare.add_argument("--out", default=None, help="CSV path.")

    tn_sweep = timesnet_sub.add_parser(
        "sweep", help="Train and evaluate every combination of a parameter grid."
    )
    _add_train_args(tn_sweep, sweep=True)
    tn_sweep.add_argument(
        "--grid",
        required=True,
        help='Example: "window=96,192;pred_len=96;d_model=64,128". '
        "Keys: window, pred_len, d_model, d_ff, e_layers, top_k, num_kernels, "
        "batch_size, learning_rate, dropout, loss, mark_set, target, epochs, seed, train_stride.",
    )
    tn_sweep.add_argument("--prefix", default="sweep")
