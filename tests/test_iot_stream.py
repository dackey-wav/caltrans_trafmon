"""Stream logic of the IoT predictor node: bucketing, gaps, duplicates, scoring."""

import math

import pandas as pd
import pytest
from app.iot.stream_state import StreamState

pytestmark = pytest.mark.unit

START = pd.Timestamp("2025-06-02T07:00:00Z")  # 00:00 PDT


def msg(ts: pd.Timestamp, flow=100.0, speed=60.0, occ=0.05, observed=100.0) -> dict:
    return {
        "schema_version": 1,
        "station_id": "1",
        "timestamp": ts.tz_convert("America/Los_Angeles").strftime("%Y-%m-%d %H:%M:%S"),
        "ts_utc": ts.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "flow_veh_5min": flow,
        "speed_mph": speed,
        "occupancy_fraction": occ,
        "observed_percent": observed,
    }


def feed(state: StreamState, n: int, start=START, **kw) -> list:
    closed = []
    for i in range(n):
        closed += state.ingest(msg(start + pd.Timedelta(minutes=5 * i), **kw))
    return closed


def test_buckets_close_on_last_slot_and_aggregate_like_training():
    state = StreamState()
    closed = feed(state, 3, flow=10.0, speed=50.0)
    assert closed == [START]
    values, active = state.history[START]
    assert active and values["flow"] == 30.0 and values["speed"] == 50.0


def test_duplicates_ignored_and_naive_local_time_accepted():
    state = StreamState()
    m = msg(START)
    state.ingest(m)
    state.ingest(m)
    del m["ts_utc"]
    state.ingest(m)
    assert state.duplicates == 2 and state.received == 1


def test_missing_rows_and_partial_observation_make_buckets_inactive():
    state = StreamState()
    feed(state, 2)  # bucket 0 is missing its last row
    third = START + pd.Timedelta(minutes=45)
    closed = state.ingest(msg(third, observed=50.0))  # skips buckets 1 and 2 entirely
    assert closed == [START, START + pd.Timedelta(minutes=15), START + pd.Timedelta(minutes=30)]
    assert [a for _, a in state.history.values()] == [False, False, False]
    assert math.isnan(state.history[START][0]["flow"])
    state.ingest(msg(third + pd.Timedelta(minutes=5)))
    closed = state.ingest(msg(third + pd.Timedelta(minutes=10)))
    assert closed == [third] and state.history[third][1] is False


def test_whole_bucket_lost_after_a_cleanly_closed_one_keeps_grid_contiguous():
    state = StreamState()
    feed(state, 3)  # bucket 0 closes on its last slot
    closed = feed(state, 3, start=START + pd.Timedelta(minutes=30))  # bucket 1 never arrives
    assert closed == [START + pd.Timedelta(minutes=15), START + pd.Timedelta(minutes=30)]
    assert state.window(3) is not None and state.history[START + pd.Timedelta(minutes=15)][1] is False


def test_forecast_scored_at_the_right_lead_against_weekly_baseline():
    state = StreamState()
    week = pd.Timedelta(days=7)
    feed(state, 3, flow=50.0)  # last week's bucket at START
    state.history.move_to_end(START)
    # jump a week ahead; intermediate buckets are filled as gaps
    later = START + week
    origin = later - pd.Timedelta(minutes=15)
    fc = pd.DataFrame(
        {"flow": [90.0, 999.0], "speed": [60.0, 60.0], "occupancy": [0.05, 0.05]},
        index=pd.DatetimeIndex([later, later + pd.Timedelta(minutes=15)]),
    )
    state.ingest(msg(origin))  # closes all gap buckets up to origin
    state.add_forecast(fc)
    state.ingest(msg(origin + pd.Timedelta(minutes=5)))
    state.ingest(msg(origin + pd.Timedelta(minutes=10)))
    closed = feed(state, 3, start=later, flow=40.0)
    assert closed == [later]
    result = state.evaluate(later)
    assert result["active"] and result["actual"]["flow"] == 120.0
    assert result["predictions"]["15min"]["flow"] == 90.0
    assert result["weekly_baseline"]["flow"] == 150.0
    row = state.errors.summary()["0-1h"]["flow"]
    assert row["mae"] == 30.0 and row["weekly_mae"] == 30.0 and row["skill_vs_weekly"] == 0.0
    assert result["anomaly"]["warming_up"] is True


def test_restarted_source_resets_state_and_snapshot_round_trips(tmp_path):
    state = StreamState()
    feed(state, 30)
    path = tmp_path / "state.pkl"
    state.save(path)
    restored = StreamState.load(path)
    assert list(restored.history) == list(state.history) and restored.last_ts == state.last_ts
    # an old timestamp right after the snapshot is a QoS 1 redelivery ...
    restored.ingest(msg(START + pd.Timedelta(minutes=5 * 29)))
    assert restored.duplicates == 1 and restored.resets == 0
    # ... but jumping back by more than an hour means the replay started over
    closed = feed(restored, 3)
    assert restored.resets == 1 and closed == [START] and len(restored.history) == 1


def test_new_replay_session_clears_state_and_dashboard_history(tmp_path):
    state = StreamState()
    for i in range(6):
        state.ingest(msg(START + pd.Timedelta(minutes=5 * i)) | {"session": "a"})
    state.recent.append({"t": "x"})
    path = tmp_path / "s.pkl"
    state.save(path)
    restored = StreamState.load(path)
    assert restored.session == "a" and list(restored.recent) == [{"t": "x"}]
    # the dashboard started a new session at a later date: fresh state, no resets of counters
    restored.ingest(msg(START + pd.Timedelta(days=30)) | {"session": "b"})
    assert restored.session == "b" and restored.resets == 1
    assert len(restored.history) == 0 and len(restored.recent) == 0


def test_window_requires_enough_history():
    state = StreamState()
    feed(state, 9)
    assert state.window(4) is None
    assert list(state.window(3).index) == [START + pd.Timedelta(minutes=15 * i) for i in range(3)]
