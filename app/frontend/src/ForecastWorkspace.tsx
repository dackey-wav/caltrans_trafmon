import { useCallback, useEffect, useMemo, useState } from 'react';
import ReactECharts from 'echarts-for-react';
import { ArrowLeft, BarChart2 } from 'lucide-react';
import { fetchForecastChart, fetchForecastRun, fetchForecastSeries, fetchGtEvents } from './api';
import {
  buildForecastChartOption,
  buildForecastResidualOption,
  CHART_WINDOW_LABELS,
  DEFAULT_TIMESNET_CHART_LAYERS,
  seriesDisplayName,
  seriesLocationId,
  type ChartWindowPreset,
  type TimesnetChartLayers,
} from './charts';
import { KpiCard } from './models/KpiCard';
import type {
  ForecastChartResponse,
  ForecastRunManifest,
  ForecastSeriesRow,
  GtEvent,
} from './types';

type ViewMode = 'series' | 'charts';
type SortKey = 'smape' | 'mae' | 'rmse' | 'flags';

const CHART_FILL_STYLE = { height: '100%', width: '100%' };

type Props = {
  modelId: string;
  modelLabel: string;
  runId: string | null;
};

function num(value: number | undefined | null, digits = 2): string {
  if (value == null || Number.isNaN(value)) return '—';
  return Number(value).toFixed(digits);
}

function smapeStr(value: number | undefined | null): string {
  if (value == null || Number.isNaN(value)) return '—';
  return `${Number(value).toFixed(1)}%`;
}

export function ForecastWorkspace({ modelId, modelLabel, runId }: Props) {
  const [manifest, setManifest] = useState<ForecastRunManifest | null>(null);
  const [manifestLoading, setManifestLoading] = useState(false);
  const [scope, setScope] = useState<string | null>(null);
  const [perSeries, setPerSeries] = useState<ForecastSeriesRow[]>([]);
  const [seriesLoading, setSeriesLoading] = useState(false);
  const [selectedSeriesId, setSelectedSeriesId] = useState<string | null>(null);
  const [chart, setChart] = useState<ForecastChartResponse | null>(null);
  const [chartLoading, setChartLoading] = useState(false);
  const [view, setView] = useState<ViewMode>('series');
  const [sortKey, setSortKey] = useState<SortKey>('smape');
  const [gtOpen, setGtOpen] = useState(false);
  const [gtEvents, setGtEvents] = useState<GtEvent[]>([]);
  const [chartLayers, setChartLayers] = useState<TimesnetChartLayers>(DEFAULT_TIMESNET_CHART_LAYERS);
  const [chartWindow, setChartWindow] = useState<ChartWindowPreset>('test');
  const [error, setError] = useState<string | null>(null);

  const loadSeries = useCallback(
    (rid: string, sc: string | undefined) => {
      setSeriesLoading(true);
      fetchForecastSeries(modelId, rid, sc)
        .then((res) => {
          setScope(res.scope);
          setPerSeries(res.per_series);
        })
        .catch((e) => {
          setError(e instanceof Error ? e.message : String(e));
          setPerSeries([]);
        })
        .finally(() => setSeriesLoading(false));
    },
    [modelId],
  );

  useEffect(() => {
    void Promise.resolve().then(() => {
      if (!runId) {
        setManifest(null);
        setPerSeries([]);
        setSelectedSeriesId(null);
        setChart(null);
        return;
      }
      setManifestLoading(true);
      setError(null);
      setView('series');
      setSelectedSeriesId(null);
      setChart(null);
      fetchForecastRun(modelId, runId)
        .then((man) => {
          setManifest(man);
          const sc = man.default_scope || man.scopes?.[0];
          setScope(sc ?? null);
          loadSeries(runId, sc);
        })
        .catch((e) => {
          setError(e instanceof Error ? e.message : String(e));
          setManifest(null);
          setPerSeries([]);
        })
        .finally(() => setManifestLoading(false));
    });
  }, [modelId, runId, loadSeries]);

  const loadChart = useCallback(
    (uid: string, sc: string | undefined, window: ChartWindowPreset = chartWindow) => {
      if (!runId) return;
      setChartLoading(true);
      setChart(null);
      fetchForecastChart(modelId, runId, uid, sc, window)
        .then(setChart)
        .catch(() => setChart(null))
        .finally(() => setChartLoading(false));
    },
    [modelId, runId, chartWindow],
  );

  const selectSeries = useCallback(
    (uid: string) => {
      if (selectedSeriesId === uid) {
        setSelectedSeriesId(null);
        setChart(null);
        return;
      }
      setSelectedSeriesId(uid);
      loadChart(uid, scope ?? undefined);
    },
    [selectedSeriesId, scope, loadChart],
  );

  const changeScope = useCallback(
    (sc: string) => {
      if (!runId || sc === scope) return;
      setScope(sc);
      loadSeries(runId, sc);
      if (selectedSeriesId) loadChart(selectedSeriesId, sc);
    },
    [runId, scope, selectedSeriesId, loadSeries, loadChart],
  );

  useEffect(() => {
    if (view !== 'charts' || !selectedSeriesId) return;
    void Promise.resolve().then(() =>
      loadChart(selectedSeriesId, scope ?? undefined, chartWindow),
    );
  }, [view, selectedSeriesId, scope, chartWindow, loadChart]);

  const gtPrefix = useMemo(() => {
    if (selectedSeriesId) return seriesLocationId(selectedSeriesId);
    if (perSeries.length > 0) return seriesLocationId(perSeries[0].unique_id);
    return null;
  }, [selectedSeriesId, perSeries]);

  useEffect(() => {
    if (!gtPrefix) {
      void Promise.resolve().then(() => setGtEvents([]));
      return;
    }
    fetchGtEvents(modelId, { locationPrefix: gtPrefix })
      .then((res) => setGtEvents(res.events))
      .catch(() => setGtEvents([]));
  }, [modelId, gtPrefix]);

  const sortedSeries = useMemo(() => {
    const rows = [...perSeries];
    rows.sort((a, b) => {
      if (sortKey === 'flags') return b.n_anomalies - a.n_anomalies;
      return (b[sortKey] ?? 0) - (a[sortKey] ?? 0);
    });
    return rows;
  }, [perSeries, sortKey]);

  const forecastOption = useMemo(
    () =>
      chart
        ? buildForecastChartOption(
            chart,
            seriesDisplayName(chart.unique_id),
            gtEvents,
            chartLayers,
            chartWindow,
          )
        : null,
    [chart, gtEvents, chartLayers, chartWindow],
  );
  const residualOption = useMemo(
    () =>
      chart
        ? buildForecastResidualOption(
            chart,
            seriesDisplayName(chart.unique_id),
            gtEvents,
            chartLayers,
            chartWindow,
          )
        : null,
    [chart, gtEvents, chartLayers, chartWindow],
  );

  if (manifestLoading && !manifest) {
    return <div className="empty-state">Loading run…</div>;
  }
  if (!runId) {
    return <div className="empty-state">Select a run from the sidebar.</div>;
  }
  if (!manifest) {
    return <div className="empty-state">{error ?? 'No forecasting artifacts for this run.'}</div>;
  }

  const meta = manifest.dataset_meta;
  const macro = manifest.macro;
  const delta = manifest.delta;
  const inCharts = view === 'charts';
  const showDetail = !inCharts && !!selectedSeriesId;
  const totalFlags = perSeries.reduce((acc, r) => acc + (r.n_anomalies || 0), 0);
  const scopeLabel = (sc: string) => (sc === 'baseline' ? 'Baseline' : modelLabel);

  return (
    <div
      className={`models-layout${showDetail ? ' models-layout--series' : ''}${inCharts ? ' models-layout--charts' : ''}`}
    >
      <div className="models-body models-scroll">
        <div className="models-run-header">
          <div className="models-run-title">
            {inCharts ? (
              <button type="button" className="models-back-btn" onClick={() => setView('series')}>
                <ArrowLeft size={14} />
                Series list
              </button>
            ) : (
              <>
                <span className="models-run-chip models-run-chip--forecast">{modelLabel}</span>
                <code className="models-run-id">{manifest.run_id}</code>
              </>
            )}
          </div>
          {!inCharts && (
            <div className="models-run-meta">
              <span>forecasting</span>
              {meta?.ds_min && meta?.ds_max && (
                <span>
                  {meta.ds_min.slice(0, 10)} → {meta.ds_max.slice(0, 10)}
                </span>
              )}
              {meta?.unique_id_count != null && <span>{meta.unique_id_count} series</span>}
            </div>
          )}
        </div>

        {error && !inCharts && <div className="models-empty-hint">{error}</div>}

        {!inCharts && (
          <>
            {manifest.scopes.length > 1 && (
              <div className="models-mode-toggle">
                <span className="models-mode-label">Forecast</span>
                {manifest.scopes.map((sc) => (
                  <button
                    key={sc}
                    type="button"
                    className={`models-mode-btn ${scope === sc ? 'active' : ''}`}
                    onClick={() => changeScope(sc)}
                  >
                    {scopeLabel(sc)}
                  </button>
                ))}
              </div>
            )}

            <div className="models-kpi-row">
              <KpiCard
                label="MAE"
                value={num(macro?.mae)}
                sub={
                  delta?.mae != null
                    ? `${delta.mae >= 0 ? '+' : ''}${num(delta.mae)} vs baseline`
                    : undefined
                }
              />
              <KpiCard label="RMSE" value={num(macro?.rmse)} />
              <KpiCard label="sMAPE" value={smapeStr(macro?.smape)} />
              <KpiCard
                label="Residual anomalies"
                value={String(totalFlags)}
                sub={
                  manifest.anomaly_params?.threshold != null
                    ? `robust z ≥ ${manifest.anomaly_params.threshold}`
                    : 'residual z-score'
                }
              />
            </div>

            {gtPrefix && (
              <div className="models-metrics-collapsible">
                <button
                  type="button"
                  className="models-metrics-toggle"
                  onClick={() => setGtOpen((o) => !o)}
                  aria-expanded={gtOpen}
                >
                  {gtOpen ? 'Hide' : 'Show'} ground truth ({gtEvents.length} nearby)
                </button>
                {gtOpen && (
                  <div className="models-gt-table-wrap">
                    <table className="models-table models-table-compact">
                      <thead>
                        <tr>
                          <th>Date</th>
                          <th>Type</th>
                          <th>Description</th>
                        </tr>
                      </thead>
                      <tbody>
                        {gtEvents.slice(0, 40).map((ev) => (
                          <tr key={ev.event_id}>
                            <td>{ev.date_local}</td>
                            <td>{ev.event_type}</td>
                            <td className="models-gt-desc">{ev.description}</td>
                          </tr>
                        ))}
                      </tbody>
                    </table>
                    {gtEvents.length === 0 && (
                      <p className="models-empty-hint">No GT events for this location.</p>
                    )}
                  </div>
                )}
              </div>
            )}

            <div className="models-actions">
              {selectedSeriesId && (
                <button
                  type="button"
                  className="btn-secondary models-action-btn"
                  onClick={() => setView('charts')}
                >
                  <BarChart2 size={14} />
                  Charts for selected series
                </button>
              )}
            </div>

            <div className="models-series-table-wrap models-series-table-wrap--primary">
              <div className="models-section-head models-section-head--compact">
                <span className="models-section-label">Per-series forecast quality</span>
                <label className="models-location-picker">
                  Sort
                  <select
                    className="models-select"
                    value={sortKey}
                    onChange={(e) => setSortKey(e.target.value as SortKey)}
                  >
                    <option value="smape">sMAPE</option>
                    <option value="mae">MAE</option>
                    <option value="rmse">RMSE</option>
                    <option value="flags">Anomalies</option>
                  </select>
                </label>
              </div>
              <table className="models-table">
                <thead>
                  <tr>
                    <th>Series</th>
                    <th>MAE</th>
                    <th>RMSE</th>
                    <th>sMAPE</th>
                    <th>Flags</th>
                  </tr>
                </thead>
                <tbody>
                  {sortedSeries.map((row) => (
                    <tr
                      key={row.unique_id}
                      className={row.unique_id === selectedSeriesId ? 'selected' : ''}
                      onClick={() => selectSeries(row.unique_id)}
                    >
                      <td className="models-series-id">{seriesDisplayName(row.unique_id)}</td>
                      <td>{num(row.mae)}</td>
                      <td>{num(row.rmse)}</td>
                      <td>{smapeStr(row.smape)}</td>
                      <td>{row.n_anomalies}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
              {sortedSeries.length === 0 && !seriesLoading && (
                <p className="models-empty-hint">No per-series metrics in this run.</p>
              )}
              {seriesLoading && <p className="models-empty-hint">Loading series…</p>}
            </div>
          </>
        )}

        {inCharts && (
          <div className="models-chart-layers">
            <span className="models-mode-label">Window</span>
            <label className="models-location-picker">
              <select
                className="models-select"
                value={chartWindow}
                onChange={(e) => setChartWindow(e.target.value as ChartWindowPreset)}
              >
                {(Object.keys(CHART_WINDOW_LABELS) as ChartWindowPreset[]).map((key) => (
                  <option key={key} value={key}>
                    {CHART_WINDOW_LABELS[key]}
                  </option>
                ))}
              </select>
            </label>
            <span className="models-mode-label">Overlays</span>
            <label className="models-layer-check">
              <input
                type="checkbox"
                checked={chartLayers.splitBands}
                onChange={(e) => setChartLayers((l) => ({ ...l, splitBands: e.target.checked }))}
              />
              Split bands
            </label>
            <label className="models-layer-check">
              <input
                type="checkbox"
                checked={chartLayers.gtBands}
                onChange={(e) => setChartLayers((l) => ({ ...l, gtBands: e.target.checked }))}
              />
              GT bands (blue)
            </label>
            <label className="models-layer-check">
              <input
                type="checkbox"
                checked={chartLayers.modelBands}
                onChange={(e) => setChartLayers((l) => ({ ...l, modelBands: e.target.checked }))}
              />
              Anomaly bands (red)
            </label>
            <label className="models-layer-check">
              <input
                type="checkbox"
                checked={chartLayers.modelPoints}
                onChange={(e) => setChartLayers((l) => ({ ...l, modelPoints: e.target.checked }))}
              />
              Anomaly points
            </label>
          </div>
        )}

        {inCharts && selectedSeriesId && (
          <section className="models-charts-view">
            <div className="models-section-head models-section-head--compact">
              <span className="models-section-label">{seriesDisplayName(selectedSeriesId)}</span>
            </div>
            <div className="models-series-chart-stack">
              <div className="models-chart-panel models-chart-panel--fill">
                {chartLoading && <div className="models-chart-loading">Loading…</div>}
                {!chartLoading && forecastOption && (
                  <div className="models-chart-echarts-host">
                    <ReactECharts option={forecastOption} style={CHART_FILL_STYLE} notMerge lazyUpdate />
                  </div>
                )}
                {!chartLoading && !forecastOption && (
                  <div className="models-empty-hint">No forecast data for this series.</div>
                )}
              </div>
              {!chartLoading && residualOption && (
                <div className="models-chart-panel models-chart-panel--fill models-chart-panel--score">
                  <div className="models-chart-panel-title">Residual</div>
                  <div className="models-chart-echarts-host">
                    <ReactECharts option={residualOption} style={CHART_FILL_STYLE} notMerge lazyUpdate />
                  </div>
                </div>
              )}
            </div>
          </section>
        )}
      </div>

      {showDetail && (
        <aside className="models-detail models-detail--open models-scroll">
          <div className="models-detail-header">
            {seriesDisplayName(selectedSeriesId!)}
            <div className="models-detail-sub">
              {chartLoading ? 'Loading…' : `${chart?.anomalies.length ?? 0} flagged points`}
            </div>
          </div>
          <div className="models-anomaly-list">
            {chartLoading && <p className="models-empty-hint">Loading…</p>}
            {!chartLoading && (!chart || chart.anomalies.length === 0) && (
              <p className="models-empty-hint">No residual anomalies for this series.</p>
            )}
            {!chartLoading &&
              chart?.anomalies.map((a, i) => {
                const sev = String(a.severity ?? '').toLowerCase();
                const sevClass =
                  sev === 'alert' ? 'sev-alert' : sev === 'warning' ? 'sev-warn' : '';
                return (
                  <div key={`${a.ds}-${i}`} className="map-anomaly-card">
                    {a.ds && <div className="time">{a.ds.slice(0, 16).replace('T', ' ')}</div>}
                    {a.severity && <div className={`sev ${sevClass}`}>{String(a.severity)}</div>}
                    <div className="summary">
                      residual {num(a.residual)}
                      {a.score != null && ` · z ${num(a.score)}`}
                    </div>
                  </div>
                );
              })}
          </div>
        </aside>
      )}
    </div>
  );
}
