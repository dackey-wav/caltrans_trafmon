"""The live forecaster reproduces the offline evaluation for the same origin."""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

pytestmark = pytest.mark.ml_cpu

EXP = Path("artifacts/forecast/timesnet/ds15/experiments/sweep_window192_mark_setcompact_learning_rate0.0001")


@pytest.mark.skipif(not (EXP / "model.pt").exists(), reason="trained experiment not present")
def test_live_forecast_matches_offline_predictions():
    from app.iot.forecaster import LiveForecaster
    from models.utils.multivariate_data_prep import load_dataset

    fc = LiveForecaster(EXP)
    frame = load_dataset(EXP.parent.parent / "dataset.parquet")
    offline = pd.read_parquet(EXP / "forecast_test.parquet")
    lead1 = offline[(offline.lead_steps == 1) & (offline.channel == "flow")].set_index("ds")
    target = lead1.index[500]
    end = frame.index.get_loc(target)  # first forecast step
    hist = frame.iloc[end - fc.info.window : end][[f"y__{c}" for c in fc.info.channels]]
    hist.columns = fc.info.channels
    active = frame.iloc[end - fc.info.window : end][[f"active__{c}" for c in fc.info.channels]]
    hist = hist.where(active.to_numpy().astype(bool))  # inactive steps arrive as gaps live

    out = fc.forecast(hist)
    assert out.index[0] == target and len(out) == fc.info.pred_len
    np.testing.assert_allclose(out.loc[target, "flow"], lead1.loc[target, "y_pred"], rtol=0.05)
