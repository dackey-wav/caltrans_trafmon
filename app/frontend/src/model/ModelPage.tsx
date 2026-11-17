import { useEffect, useMemo, useState } from 'react';
import ReactECharts from 'echarts-for-react';
import { fetchModel, fetchStation, type ModelInfo, type StationInfo } from '../api';
import { StationMap } from './StationMap';
import { CHANNEL_DISPLAY } from '../live/chart';
import type { Channel } from '../live/types';
import { CHANNELS } from '../live/types';

const METHOD_STYLE: Record<string, { label: string; color: string; dash?: string }> = {
  model: { label: 'TimesNet', color: '#ff9830' },
  weekly: { label: 'Same time last week', color: '#8a8f98', dash: 'dotted' },
  daily: { label: 'Same time yesterday', color: '#b877d9', dash: 'dashed' },
  persistence: { label: 'Last value repeated', color: '#56d0e0', dash: 'dashed' },
};

const AXIS = {
  axisLine: { lineStyle: { color: '#2c3235' } },
  axisLabel: { color: '#9aa0a6', fontSize: 10 },
  splitLine: { lineStyle: { color: '#1f2328' } },
};

const TOOLTIP = {
  trigger: 'axis',
  backgroundColor: '#111217',
  borderColor: '#2c3235',
  textStyle: { color: '#d8d9da', fontSize: 12 },
};

function day(iso: string) {
  return iso.slice(0, 10);
}

function coverageOption(station: StationInfo) {
  return {
    backgroundColor: 'transparent',
    animation: false,
    tooltip: { ...TOOLTIP, valueFormatter: (v: number) => `${v.toFixed(1)} %` },
    grid: { left: 8, right: 12, top: 16, bottom: 8, containLabel: true },
    xAxis: { type: 'category', data: station.monthly.map((m) => m.month), ...AXIS, splitLine: { show: false } },
    yAxis: { type: 'value', max: 100, name: '% observed', nameTextStyle: { color: '#6e7681', fontSize: 10 }, ...AXIS },
    series: [
      {
        type: 'bar',
        name: 'Fully observed 5-min rows',
        data: station.monthly.map((m) => ({
          value: m.observed_share * 100,
          itemStyle: { color: m.observed_share >= 0.95 ? '#73bf69' : m.observed_share >= 0.7 ? '#fade2a' : '#f2495c' },
        })),
        barMaxWidth: 18,
      },
    ],
  };
}

function leadOption(model: ModelInfo, channel: Channel) {
  const d = CHANNEL_DISPLAY[channel];
  const leads = model.test.lead_minutes.map((m) => (m % 60 ? `${(m / 60).toFixed(2)} h` : `${m / 60} h`));
  const methods = model.test.methods.filter((m) => m in METHOD_STYLE && m !== 'persistence');
  return {
    backgroundColor: 'transparent',
    animation: false,
    tooltip: { ...TOOLTIP, valueFormatter: (v: number) => `${v.toFixed(d.digits)} ${d.unit}` },
    legend: { bottom: 0, textStyle: { color: '#9aa0a6', fontSize: 11 } },
    grid: { left: 8, right: 16, top: 24, bottom: 36, containLabel: true },
    xAxis: { type: 'category', data: leads, name: 'forecast lead', nameLocation: 'middle', nameGap: 26,
      nameTextStyle: { color: '#6e7681', fontSize: 10 }, ...AXIS, splitLine: { show: false } },
    yAxis: { type: 'value', name: `MAE, ${d.unit}`, nameTextStyle: { color: '#6e7681', fontSize: 10 }, ...AXIS },
    series: methods.map((m) => ({
      name: METHOD_STYLE[m].label,
      type: 'line',
      showSymbol: false,
      smooth: 0.2,
      lineStyle: { width: m === 'model' ? 2.2 : 1.4, color: METHOD_STYLE[m].color, type: METHOD_STYLE[m].dash ?? 'solid' },
      itemStyle: { color: METHOD_STYLE[m].color },
      data: model.test.by_lead[m][channel].map((v) => v * d.factor),
    })),
  };
}

export function ModelPage() {
  const [station, setStation] = useState<StationInfo | null>(null);
  const [model, setModel] = useState<ModelInfo | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [channel, setChannel] = useState<Channel>('flow');

  useEffect(() => {
    fetchStation().then(setStation).catch((e: Error) => setError(e.message));
    fetchModel().then(setModel).catch((e: Error) => setError(e.message));
  }, []);

  const leadChart = useMemo(() => (model ? leadOption(model, channel) : null), [model, channel]);
  const d = CHANNEL_DISPLAY[channel];

  return (
    <div className="model-page">
      {error && <div className="error-banner">{error}</div>}
      <section className="panel station-panel">
        <h4>Station</h4>
        {station ? (
          <>
            <div className="facts">
              <div><span>PeMS VDS</span><b>{station.station_id}</b></div>
              <div><span>Road</span><b>I-{station.freeway} {station.direction} · {station.lane_type === 'ML' ? 'mainline' : station.lane_type}</b></div>
              <div><span>Data</span><b>{day(station.first)} → {day(station.last)}</b></div>
              <div><span>Observed (not imputed)</span><b>{(station.observed_share * 100).toFixed(1)} %</b></div>
            </div>
            {station.latitude != null && station.longitude != null && (
              <StationMap lat={station.latitude} lon={station.longitude}
                label={`VDS ${station.station_id} · I-${station.freeway} ${station.direction}`} />
            )}
            <h5>Share of fully observed 5-minute rows per month</h5>
            <div className="coverage-chart">
              <ReactECharts option={coverageOption(station)} notMerge style={{ height: '100%' }} />
            </div>
          </>
        ) : (
          <p className="muted">Loading…</p>
        )}
      </section>

      <section className="panel model-panel">
        <h4>Model in the predictor node</h4>
        {model ? (
          <>
            <div className="facts">
              <div><span>Architecture</span><b>TimesNet, {(model.parameters / 1e6).toFixed(1)} M parameters</b></div>
              <div><span>Input → output</span><b>{Number(model.hparams.window) / 4} h → {Number(model.hparams.pred_len) / 4} h, 15-min steps</b></div>
              <div><span>Train</span><b>{day(model.split_ds.train[0])} → {day(model.split_ds.train[1])}</b></div>
              <div><span>Validation</span><b>{day(model.split_ds.val[0])} → {day(model.split_ds.val[1])}</b></div>
              <div><span>Test</span><b>{day(model.split_ds.test[0])} → {day(model.split_ds.test[1])}</b></div>
              <div><span>Training</span><b>best epoch {model.best_epoch} of {model.epochs_run}, {Math.round(model.train_seconds / 60)} min on {model.device}</b></div>
            </div>
            <div className="panel-toolbar">
              <h5>Test-period error by forecast lead</h5>
              <div className="segmented">
                {CHANNELS.map((c) => (
                  <button key={c} className={c === channel ? 'active' : ''} onClick={() => setChannel(c)}>
                    {CHANNEL_DISPLAY[c].label}
                  </button>
                ))}
              </div>
            </div>
            <div className="lead-chart">
              {leadChart && <ReactECharts option={leadChart} notMerge style={{ height: '100%' }} />}
            </div>
            <table className="metrics-table">
              <thead>
                <tr>
                  <th>MAE, {d.unit}</th>
                  {Object.keys(model.test.by_lead_bucket).map((b) => <th key={b}>{b}</th>)}
                  <th>all leads</th>
                </tr>
              </thead>
              <tbody>
                {model.test.methods.filter((m) => m in METHOD_STYLE).map((m) => (
                  <tr key={m} className={m === 'model' ? 'row-model' : ''}>
                    <td>{METHOD_STYLE[m].label}</td>
                    {Object.values(model.test.by_lead_bucket).map((per, i) => (
                      <td key={i}>{(per[m][channel] * d.factor).toFixed(d.digits)}</td>
                    ))}
                    <td>{(model.test.overall[m][channel] * d.factor).toFixed(d.digits)}</td>
                  </tr>
                ))}
              </tbody>
            </table>
            <p className="muted small">
              Offline evaluation on the held-out test split ({model.test.n_origins.toLocaleString()} forecast
              origins, only fully observed steps scored). Skill vs last week:{' '}
              {CHANNELS.map((c) => `${CHANNEL_DISPLAY[c].label.toLowerCase()} ${(model.test.skill_vs_weekly[c] * 100).toFixed(0)} %`).join(', ')}.
            </p>
          </>
        ) : (
          <p className="muted">Loading…</p>
        )}
      </section>
    </div>
  );
}
