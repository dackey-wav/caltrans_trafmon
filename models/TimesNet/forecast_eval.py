"""Rolling-origin forecast evaluation in original units.

Every row of the split serves as a forecast origin. The model competes with
three baselines that need no training:

    persistence  last observed value repeated over the horizon
    daily        the value one day before each target step
    weekly       the value one week before each target step

All methods are scored on the same cells: the target, the last input step, and
the values one day and one week earlier are all real observations. Baselines
therefore never get scored on forward-filled gap values. 5-minute runs also get scores on 15-minute blocks so the
two frequencies compare directly.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from models.TimesNet.model import Model
from models.TimesNet.train_timesnet import (
    FeaturePlan,
    amp_dtype,
    autocast,
    build_model_inputs,
)
from models.utils.constants import LOCAL_TZ, step_minutes, steps_per_day, steps_per_hour
from models.utils.lazy_windows import forecast_window_starts

METHODS = ("model", "persistence", "daily", "weekly")
BASELINES = ("persistence", "daily", "weekly")
CHANNEL_UNITS = {
    "flow": "veh per step",
    "speed": "mph",
    "occupancy": "fraction",
}
SUM_CHANNELS = ("flow",)


@dataclass
class EvalResult:
    metrics: dict
    predictions: pd.DataFrame
    examples: list[dict]


def lead_buckets(freq: str, pred_len: int) -> list[tuple[str, int, int]]:
    """(label, first lead, last lead) in steps, 1-based and inclusive."""
    sph = steps_per_hour(freq)
    edges = [("0-1h", 1, sph), ("1-4h", sph + 1, 4 * sph), ("4-24h", 4 * sph + 1, 24 * sph)]
    edges.append((">24h", 24 * sph + 1, pred_len))
    out = []
    for label, lo, hi in edges:
        hi = min(hi, pred_len)
        if lo <= hi:
            out.append((label, lo, hi))
    return out


def stored_leads(freq: str, pred_len: int) -> list[int]:
    sph = steps_per_hour(freq)
    return sorted({k for k in (1, sph, 4 * sph, pred_len) if 1 <= k <= pred_len})


def lead_label(freq: str, lead: int) -> str:
    minutes = lead * step_minutes(freq)
    if minutes < 60:
        return f"+{minutes} min"
    hours = minutes / 60
    return f"+{hours:g} h"


def _pick_example_positions(
    index: pd.DatetimeIndex,
    starts: np.ndarray,
    window: int,
    pred_len: int,
    active: np.ndarray,
    n_examples: int,
    min_coverage: float = 0.8,
) -> list[int]:
    """Spread origins over the split: local midnight, well-observed horizon first."""
    if n_examples <= 0 or len(starts) == 0:
        return []
    origin_ts = pd.DatetimeIndex(index[starts + window])
    if origin_ts.tz is not None:
        origin_ts = origin_ts.tz_convert(LOCAL_TZ)
    obs = active.any(axis=1).astype(np.int64)
    csum = np.concatenate([[0], np.cumsum(obs)])
    lo = starts + window
    coverage = (csum[lo + pred_len] - csum[lo]) / pred_len
    midnight = (origin_ts.hour == 0) & (origin_ts.minute == 0)
    pool = np.arange(len(starts))
    for cond in (midnight & (coverage >= min_coverage), coverage >= min_coverage, midnight):
        hits = np.flatnonzero(cond)
        if len(hits) >= n_examples:
            pool = hits
            break
    picks = np.linspace(0, len(pool) - 1, num=min(n_examples, len(pool))).round().astype(int)
    return sorted({int(pool[p]) for p in picks})


def _agg_blocks(arr: np.ndarray, block: int, sum_mask: np.ndarray) -> np.ndarray:
    """(B, H, C) -> (B, H // block, C): sums for count channels, means for the rest."""
    b, h, c = arr.shape
    nb = h // block
    blocks = arr[:, : nb * block].reshape(b, nb, block, c)
    return np.where(sum_mask[None, None, :], blocks.sum(axis=2), blocks.mean(axis=2))


def _score(abs_sum: np.ndarray, sq_sum: np.ndarray, cnt: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    n = np.maximum(cnt, 1.0)
    mae = np.where(cnt > 0, abs_sum / n, np.nan)
    rmse = np.where(cnt > 0, np.sqrt(sq_sum / n), np.nan)
    return mae, rmse


def _nan_to_none(value):
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def evaluate_forecast(
    model: Model,
    frame: pd.DataFrame,
    plan: FeaturePlan,
    scaler: dict[str, np.ndarray],
    *,
    freq: str,
    window: int,
    pred_len: int,
    rows: tuple[int, int],
    split: str = "test",
    device: torch.device | None = None,
    batch_size: int = 512,
    use_amp: bool = True,
    n_examples: int = 6,
) -> EvalResult:
    device = device or torch.device("cpu")
    dtype = amp_dtype(device, use_amp)
    model = model.to(device).eval()

    inputs = build_model_inputs(frame, plan, scaler)
    raw = frame[plan.target_columns].to_numpy(dtype=np.float64)
    active = inputs.active
    channels = plan.channels
    n_ch = len(channels)
    med = scaler["median"].astype(np.float64)
    sc = scaler["scale"].astype(np.float64)

    lo, hi = rows
    starts = forecast_window_starts(len(frame), window, pred_len, target_lo=lo, target_hi=hi)
    if len(starts) == 0:
        raise RuntimeError(f"No forecast origins in rows {rows} for window={window} pred_len={pred_len}")

    spd = steps_per_day(freq)
    week = 7 * spd
    block = max(1, 15 // step_minutes(freq))
    sum_mask = np.array([ch in SUM_CHANNELS for ch in channels])
    leads = stored_leads(freq, pred_len)
    lead_idx = np.asarray(leads) - 1
    example_pos = set(
        _pick_example_positions(frame.index, starts, window, pred_len, active, n_examples)
    )

    abs_sum = {m: np.zeros((pred_len, n_ch)) for m in METHODS}
    sq_sum = {m: np.zeros((pred_len, n_ch)) for m in METHODS}
    cnt = np.zeros((pred_len, n_ch))
    nb = pred_len // block
    babs = {m: np.zeros(n_ch) for m in METHODS}
    bsq = {m: np.zeros(n_ch) for m in METHODS}
    bcnt = np.zeros(n_ch)
    kept = np.empty((len(starts), len(leads), n_ch), dtype=np.float32)
    examples: list[dict] = []

    arange_w = np.arange(window)
    arange_wh = np.arange(window + pred_len)
    arange_h = np.arange(pred_len)

    with torch.no_grad():
        for i in range(0, len(starts), batch_size):
            idx = starts[i : i + batch_size]
            x = torch.from_numpy(inputs.scaled[idx[:, None] + arange_w]).to(device)
            mk = torch.from_numpy(inputs.marks[idx[:, None] + arange_wh]).to(device)
            with autocast(device, dtype):
                out = model(x, mk)
            out_np = out.float().cpu().numpy().astype(np.float64)
            tgt = idx[:, None] + window + arange_h
            truth = raw[tgt]
            if model.configs.residual:
                # The network emits a correction to last week, in scaled units.
                weekly_level = raw[np.maximum(tgt - week, 0)]
                pred = np.maximum(weekly_level + out_np * sc, 0.0)
            else:
                pred = np.maximum(out_np * sc + med, 0.0)
            preds = {
                "model": pred,
                "persistence": np.repeat(raw[idx + window - 1][:, None, :], pred_len, axis=1),
                "daily": raw[np.maximum(tgt - spd, 0)],
                "weekly": raw[np.maximum(tgt - week, 0)],
            }
            valid = (
                active[tgt]
                & (tgt - week >= 0)[:, :, None]
                & active[np.maximum(tgt - week, 0)]
                & active[np.maximum(tgt - spd, 0)]
                & active[idx + window - 1][:, None, :]
            )
            vf = valid.astype(np.float64)
            cnt += vf.sum(axis=0)
            for m in METHODS:
                err = np.nan_to_num(preds[m] - truth)
                abs_sum[m] += (np.abs(err) * vf).sum(axis=0)
                sq_sum[m] += (err**2 * vf).sum(axis=0)

            if nb > 0:
                bvalid = valid[:, : nb * block].reshape(len(idx), nb, block, n_ch).all(axis=2)
                bvf = bvalid.astype(np.float64)
                bcnt += bvf.sum(axis=(0, 1))
                btruth = _agg_blocks(truth, block, sum_mask)
                for m in METHODS:
                    berr = np.nan_to_num(_agg_blocks(preds[m], block, sum_mask) - btruth)
                    babs[m] += (np.abs(berr) * bvf).sum(axis=(0, 1))
                    bsq[m] += (berr**2 * bvf).sum(axis=(0, 1))

            kept[i : i + len(idx)] = pred[:, lead_idx, :]
            for j in range(len(idx)):
                if i + j in example_pos:
                    examples.append(
                        {
                            "origin_row": int(idx[j] + window),
                            "pred": pred[j].astype(np.float32),
                            "weekly": preds["weekly"][j].astype(np.float32),
                        }
                    )

    metrics = _assemble_metrics(
        abs_sum, sq_sum, cnt, babs, bsq, bcnt,
        channels=channels, freq=freq, window=window, pred_len=pred_len,
        split=split, n_origins=len(starts), block=block,
        split_ds=[str(frame.index[lo]), str(frame.index[hi - 1])],
    )

    records = []
    for k, lead in enumerate(leads):
        target_rows = starts + window + lead - 1
        for j, ch in enumerate(channels):
            records.append(
                pd.DataFrame(
                    {
                        "ds": frame.index[target_rows],
                        "lead_steps": lead,
                        "lead_minutes": lead * step_minutes(freq),
                        "channel": ch,
                        "y_true": raw[target_rows, j].astype(np.float32),
                        "y_pred": kept[:, k, j],
                        "active": active[target_rows, j],
                    }
                )
            )
    predictions = pd.concat(records, ignore_index=True)
    return EvalResult(metrics=metrics, predictions=predictions, examples=examples)


def _assemble_metrics(
    abs_sum, sq_sum, cnt, babs, bsq, bcnt,
    *, channels, freq, window, pred_len, split, n_origins, block, split_ds,
) -> dict:
    sph = steps_per_hour(freq)

    def per_channel(a: np.ndarray, s: np.ndarray, c: np.ndarray) -> dict:
        mae, rmse = _score(a, s, c)
        return {
            ch: {
                "mae": _nan_to_none(float(mae[j])),
                "rmse": _nan_to_none(float(rmse[j])),
                "n": int(c[j]),
            }
            for j, ch in enumerate(channels)
        }

    overall = {m: per_channel(abs_sum[m].sum(0), sq_sum[m].sum(0), cnt.sum(0)) for m in METHODS}
    by_bucket = {}
    for label, b_lo, b_hi in lead_buckets(freq, pred_len):
        sl = slice(b_lo - 1, b_hi)
        by_bucket[label] = {
            m: per_channel(abs_sum[m][sl].sum(0), sq_sum[m][sl].sum(0), cnt[sl].sum(0))
            for m in METHODS
        }
    by_lead = {}
    for m in METHODS:
        mae, rmse = _score(abs_sum[m], sq_sum[m], cnt)
        by_lead[m] = {
            ch: {
                "mae": [_nan_to_none(float(v)) for v in mae[:, j]],
                "rmse": [_nan_to_none(float(v)) for v in rmse[:, j]],
            }
            for j, ch in enumerate(channels)
        }

    def skill(base: str) -> dict:
        out = {}
        for ch in channels:
            mm = overall["model"][ch]["mae"]
            mb = overall[base][ch]["mae"]
            out[ch] = None if (mm is None or not mb) else 1.0 - mm / mb
        return out

    flow_hourly = {}
    if "flow" in channels:
        for m in METHODS:
            row = overall[m]["flow"]
            flow_hourly[m] = {
                "mae": None if row["mae"] is None else row["mae"] * sph,
                "rmse": None if row["rmse"] is None else row["rmse"] * sph,
            }

    return {
        "split": split,
        "split_ds": split_ds,
        "freq": freq,
        "step_minutes": step_minutes(freq),
        "window": window,
        "pred_len": pred_len,
        "horizon_hours": pred_len * step_minutes(freq) / 60,
        "n_origins": int(n_origins),
        "channels": list(channels),
        "units": {ch: CHANNEL_UNITS.get(ch, "") for ch in channels},
        "overall": overall,
        "by_lead_bucket": by_bucket,
        "by_lead": by_lead,
        "lead_minutes": [int((k + 1) * step_minutes(freq)) for k in range(pred_len)],
        "flow_veh_per_hour": flow_hourly,
        "blocks_15min": {
            m: per_channel(babs[m], bsq[m], bcnt) for m in METHODS
        } if block > 1 else None,
        "skill_vs_weekly": skill("weekly"),
        "skill_vs_daily": skill("daily"),
        "skill_vs_persistence": skill("persistence"),
    }


def save_metrics(metrics: dict, path: Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)
    return path


def load_metrics(path: Path) -> dict:
    with open(Path(path), encoding="utf-8") as f:
        return json.load(f)


def save_predictions(predictions: pd.DataFrame, path: Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    predictions.to_parquet(path, index=False)
    return path


def format_summary(metrics: dict) -> str:
    """Short human-readable table of overall MAE and lead buckets."""
    lines = [
        f"split={metrics['split']} {metrics['split_ds'][0]} .. {metrics['split_ds'][1]} "
        f"origins={metrics['n_origins']} horizon={metrics['horizon_hours']:g}h"
    ]
    for ch in metrics["channels"]:
        unit = metrics["units"].get(ch, "")
        cells = []
        for m in METHODS:
            v = metrics["overall"][m][ch]["mae"]
            cells.append(f"{m}={v:.4g}" if v is not None else f"{m}=n/a")
        sk = metrics["skill_vs_weekly"].get(ch)
        sk_txt = f" skill_vs_weekly={sk:+.1%}" if sk is not None else ""
        lines.append(f"  MAE {ch} [{unit}]: " + " ".join(cells) + sk_txt)
    for label, row in metrics["by_lead_bucket"].items():
        cells = []
        for ch in metrics["channels"]:
            mm = row["model"][ch]["mae"]
            mw = row["weekly"][ch]["mae"]
            if mm is None or mw is None:
                continue
            cells.append(f"{ch} {mm:.4g} (weekly {mw:.4g})")
        lines.append(f"  lead {label}: " + "; ".join(cells))
    return "\n".join(lines)
