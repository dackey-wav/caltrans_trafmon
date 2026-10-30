import { useCallback, useEffect, useMemo, useState } from 'react';
import ReactECharts from 'echarts-for-react';
import { ArrowLeft, BarChart2 } from 'lucide-react';
import {
  fetchGtEvents,
  fetchTimesnetLocationChart,
  fetchTimesnetScores,
} from './api';
import {
  buildTimesnetForecastChartOption,
  buildTimesnetLocationStackOption,
  buildTimesnetScoreChartOption,
  CHART_WINDOW_LABELS,
  DEFAULT_TIMESNET_CHART_LAYERS,
  MODEL_CHART_WINDOW_PRESETS,
  seriesClassId,
  seriesDisplayName,
  seriesLocationId,
  type ChartWindowPreset,
  type TimesnetChartLayers,
} from './charts';
import { KpiCard } from './models/KpiCard';
import type {
  AnomalyMode,
  GtEvent,
  TimesnetAnomaliesListResponse,
  TimesnetLocationChartResponse,
  TimesnetPerSeriesRow,
  TimesnetRunManifest,
  TimesnetScoresResponse,
} from './types';

type ViewMode = 'series' | 'charts';
type ChartScope = 'location' | 'series';

const CHART_FILL_STYLE = { height: '100%', width: '100%' };

type Props = {
  modelId: string;
  modelLabel?: string;
  runId: string | null;
  manifest: TimesnetRunManifest | null;
  perSeries: TimesnetPerSeriesRow[];
  selectedSeriesId: string | null;
  onSelectSeries: (uniqueId: string) => void;
  anomalyList: TimesnetAnomaliesListResponse | null;
  anomalyLoading: boolean;
  manifestLoading: boolean;
  anomalyMode: AnomalyMode;
  onAnomalyModeChange: (mode: AnomalyMode) => void;
};

function pct(rate: number | undefined | null): string {
  if (rate == null || Number.isNaN(rate)) return '—';
  return `${(rate * 100).toFixed(2)}%`;
}

export function ModelsWorkspace({
  modelId,
  modelLabel,
  runId,
  manifest,
  perSeries,
  selectedSeriesId,
  onSelectSeries,
  anomalyList,
  anomalyLoading,
  manifestLoading,
  anomalyMode,
  onAnomalyModeChange,
}: Props) {
  const [view, setView] = useState<ViewMode>('series');
  const [chartScope, setChartScope] = useState<ChartScope>('location');
  const [selectedLocationId, setSelectedLocationId] = useState<string | null>(null);
  const [locationChart, setLocationChart] = useState<TimesnetLocationChartResponse | null>(null);
  const [locationChartLoading, setLocationChartLoading] = useState(false);
  const [scores, setScores] = useState<TimesnetScoresResponse | null>(null);
  const [scoresLoading, setScoresLoading] = useState(false);
  const [metricsOpen, setMetricsOpen] = useState(false);
  const [gtOpen, setGtOpen] = useState(false);
  const [gtEvents, setGtEvents] = useState<GtEvent[]>([]);
  const [chartLayers, setChartLayers] = useState<TimesnetChartLayers>(DEFAULT_TIMESNET_CHART_LAYERS);
  const [chartWindow, setChartWindow] = useState<ChartWindowPreset>('test');

  const isLocationWide = useMemo(
    () =>
      manifest?.schema_version === 'location_wide_v1' ||
      perSeries.some((row) => seriesClassId(row.unique_id) != null),
    [manifest?.schema_version, perSeries],
  );

  const locationOptions = useMemo(() => {
    if (!isLocationWide) return [];
    const fromMeta = manifest?.dataset_meta?.locations ?? [];
    if (fromMeta.length > 0) return fromMeta;
    const locs = new Set<string>();
    for (const row of perSeries) {
      if (seriesClassId(row.unique_id) != null) {
        locs.add(seriesLocationId(row.unique_id));
      }
    }
    return [...locs].sort();
  }, [manifest, perSeries, isLocationWide]);

  useEffect(() => {
    if (!manifest || locationOptions.length === 0) return;
    void Promise.resolve().then(() => {
      setSelectedLocationId((previous) => previous ?? locationOptions[0]);
    });
  }, [manifest, locationOptions]);

  useEffect(() => {
    void Promise.resolve().then(() => setView('series'));
  }, [runId]);

  const loadLocationChart = useCallback(
    (locationId: string, window: ChartWindowPreset = chartWindow) => {
      if (!runId) return;
      setLocationChartLoading(true);
      setLocationChart(null);
      fetchTimesnetLocationChart(runId, locationId, {
        mode: anomalyMode,
        modelId,
        window,
      })
        .then(setLocationChart)
        .catch(() => setLocationChart(null))
        .finally(() => setLocationChartLoading(false));
    },
    [runId, anomalyMode, modelId, chartWindow],
  );

  const loadSeriesCharts = useCallback(
    (uniqueId: string, window: ChartWindowPreset = chartWindow) => {
      if (!runId) return;
      setScoresLoading(true);
      setScores(null);
      fetchTimesnetScores(runId, uniqueId, anomalyMode, modelId, window)
        .then(setScores)
        .catch(() => setScores(null))
        .finally(() => setScoresLoading(false));
    },
    [
      runId,
      anomalyMode,
      modelId,
      chartWindow,
    ],
  );

  const gtPrefix = useMemo(() => {
    if (chartScope === 'location' && selectedLocationId) {
      return seriesLocationId(selectedLocationId);
    }
    if (selectedSeriesId) {
      return seriesLocationId(selectedSeriesId);
    }
    return null;
  }, [chartScope, selectedLocationId, selectedSeriesId]);

  useEffect(() => {
    if (!gtPrefix) {
      void Promise.resolve().then(() => setGtEvents([]));
      return;
    }
    fetchGtEvents(modelId, { locationPrefix: gtPrefix })
      .then((res) => setGtEvents(res.events))
      .catch(() => setGtEvents([]));
  }, [modelId, gtPrefix]);

  const filteredGtEvents = useMemo(() => gtEvents, [gtEvents]);

  const gtHitByEventId = useMemo(() => {
    const block = manifest?.gt_eval?.modes?.[anomalyMode];
    if (!block?.events) return new Map<number, boolean>();
    return new Map(block.events.map((e) => [e.event_id, e.hit]));
  }, [manifest?.gt_eval, anomalyMode]);

  const openLocationCharts = useCallback(() => {
    const loc = selectedLocationId ?? locationOptions[0];
    if (!loc) return;
    setSelectedLocationId(loc);
    setChartScope('location');
    setView('charts');
    loadLocationChart(loc);
  }, [selectedLocationId, locationOptions, loadLocationChart]);

  const openSeriesCharts = useCallback(() => {
    if (!selectedSeriesId) return;
    setChartScope('series');
    setView('charts');
    loadSeriesCharts(selectedSeriesId);
  }, [selectedSeriesId, loadSeriesCharts]);

  useEffect(() => {
    if (view !== 'charts') return;
    if (chartScope === 'location' && selectedLocationId) {
      void Promise.resolve().then(() => loadLocationChart(selectedLocationId));
    } else if (chartScope === 'series' && selectedSeriesId) {
      void Promise.resolve().then(() => loadSeriesCharts(selectedSeriesId));
    }
  }, [
    anomalyMode,
    chartWindow,
    view,
    chartScope,
    selectedLocationId,
    selectedSeriesId,
    loadLocationChart,
    loadSeriesCharts,
  ]);

  const backToSeries = useCallback(() => {
    setView('series');
  }, []);

  const scoreChartOption = useMemo(
    () =>
      scores
        ? buildTimesnetScoreChartOption(
            scores,
            seriesDisplayName(scores.unique_id),
            filteredGtEvents,
            chartLayers,
            chartWindow,
          )
        : null,
    [scores, filteredGtEvents, chartLayers, chartWindow],
  );

  const forecastChartOption = useMemo(
    () =>
      scores
        ? buildTimesnetForecastChartOption(
            scores,
            seriesDisplayName(scores.unique_id),
            filteredGtEvents,
            chartLayers,
            chartWindow,
          )
        : null,
    [scores, filteredGtEvents, chartLayers, chartWindow],
  );

  const chartWindowPresets = MODEL_CHART_WINDOW_PRESETS;

  const locationActualOption = useMemo(
    () =>
      locationChart
        ? buildTimesnetLocationStackOption(
            locationChart,
            'actual',
            filteredGtEvents,
            chartLayers,
            chartWindow,
          )
        : null,
    [locationChart, filteredGtEvents, chartLayers, chartWindow],
  );

  const locationPredOption = useMemo(
    () =>
      locationChart
        ? buildTimesnetLocationStackOption(
            locationChart,
            'predicted',
            filteredGtEvents,
            chartLayers,
            chartWindow,
          )
        : null,
    [locationChart, filteredGtEvents, chartLayers, chartWindow],
  );

  const seriesWithAnomalies = useMemo(
    () => perSeries.filter((row) => row.n_anomalies > 0),
    [perSeries],
  );
  // Reconstruction models can flag nothing; still let the user browse series.
  const hasAnomalies = seriesWithAnomalies.length > 0;
  const sortedSeries = useMemo(
    () =>
      (hasAnomalies ? seriesWithAnomalies : [...perSeries]).sort(
        (a, b) => b.rate - a.rate || b.score_max - a.score_max,
      ),
    [hasAnomalies, seriesWithAnomalies, perSeries],
  );

  const combinedRobustZ = manifest?.combined_robust_z ?? 10;

  const thresholdLabel = useMemo(() => {
    if (anomalyMode === 'combined') {
      return `combined · percentile 0.999 ∧ robust_z ${combinedRobustZ}`;
    }
    return 'percentile · q=0.999';
  }, [anomalyMode, combinedRobustZ]);

  const otherModeRate = useMemo(() => {
    if (anomalyMode === 'combined') {
      return manifest?.summary_percentile?.anomaly_rate;
    }
    return manifest?.summary_combined?.anomaly_rate;
  }, [anomalyMode, manifest?.summary_percentile, manifest?.summary_combined]);

  const gtSummary = useMemo(() => {
    const s = manifest?.gt_eval?.modes?.[anomalyMode]?.summary;
    if (!s || s.total === 0) return null;
    const scope = manifest?.gt_eval?.scope;
    const scopeHint =
      scope?.gt_events_in_scope != null && scope?.gt_events_total_csv != null
        ? ` · ${scope.gt_events_in_scope}/${scope.gt_events_total_csv} in test scope`
        : '';
    return `${(s.hit_rate * 100).toFixed(0)}% recall (${s.hits}/${s.total})${scopeHint}`;
  }, [manifest?.gt_eval, anomalyMode]);

  if (manifestLoading && !manifest) {
    return <div className="empty-state">Loading run…</div>;
  }

  if (!manifest) {
    return <div className="empty-state">Select a run from the sidebar.</div>;
  }

  const meta = manifest.dataset_meta;
  const summary = manifest.summary;
  const inCharts = view === 'charts';
  const showDetail = !inCharts && !!selectedSeriesId;


  return (
    <div
      className={`models-layout${showDetail ? ' models-layout--series' : ''}${inCharts ? ' models-layout--charts' : ''}`}
    >
      <div className="models-body models-scroll">
        <div className="models-run-header">
          <div className="models-run-title">
            {inCharts ? (
              <button type="button" className="models-back-btn" onClick={backToSeries}>
                <ArrowLeft size={14} />
                Series list
              </button>
            ) : (
              <>
                <span className="models-run-chip">{modelLabel ?? 'TimesNet'}</span>
                <code className="models-run-id">{manifest.run_id}</code>
              </>
            )}
          </div>
          {!inCharts && thresholdLabel && (
            <div className="models-run-meta">{thresholdLabel}</div>
          )}
          {!inCharts && meta && (
            <div className="models-run-meta">
              {meta.freq && <span>{meta.freq}</span>}
              {meta.ds_min && meta.ds_max && (
                <span>
                  {meta.ds_min.slice(0, 10)} → {meta.ds_max.slice(0, 10)}
                </span>
              )}
              {meta.n_targets != null && <span>{meta.n_targets} targets</span>}
            </div>
          )}
        </div>

        {!inCharts && (
          <>
            <div className="models-mode-toggle">
              <span className="models-mode-label">Threshold</span>
              <button
                type="button"
                className={`models-mode-btn ${anomalyMode === 'percentile' ? 'active' : ''}`}
                onClick={() => onAnomalyModeChange('percentile')}
              >
                Percentile
              </button>
              <button
                type="button"
                className={`models-mode-btn ${anomalyMode === 'combined' ? 'active' : ''}`}
                onClick={() => onAnomalyModeChange('combined')}
              >
                Combined
              </button>
            </div>

            <div className="models-kpi-row">
              <KpiCard
                label={`Test rate (${anomalyMode})`}
                value={pct(
                  anomalyMode === 'combined'
                    ? manifest?.summary_combined?.anomaly_rate ?? summary?.anomaly_rate
                    : manifest?.summary_percentile?.anomaly_rate ?? summary?.anomaly_rate,
                )}
                sub={
                  otherModeRate != null
                    ? `other mode ${pct(otherModeRate)}`
                    : summary?.anomaly_count != null
                      ? `${summary.anomaly_count} flags`
                      : undefined
                }
              />
              <KpiCard
                label="Series"
                value={String(summary?.series_count ?? perSeries.length)}
                sub={
                  summary?.point_count != null ? `${summary.point_count} points` : undefined
                }
              />
              {gtSummary && (
                <KpiCard label="GT recall (test)" value={gtSummary} sub="confirmed events in test split" />
              )}
            </div>

            {gtPrefix && (
              <div className="models-metrics-collapsible">
                <button
                  type="button"
                  className="models-metrics-toggle"
                  onClick={() => setGtOpen((o) => !o)}
                  aria-expanded={gtOpen}
                >
                  {gtOpen ? 'Hide' : 'Show'} ground truth ({filteredGtEvents.length} nearby)
                </button>
                {gtOpen && (
                  <div className="models-gt-table-wrap">
                    <table className="models-table models-table-compact">
                      <thead>
                        <tr>
                          <th>Date</th>
                          <th>Type</th>
                          <th>Hit</th>
                          <th>Description</th>
                        </tr>
                      </thead>
                      <tbody>
                        {filteredGtEvents.slice(0, 40).map((ev) => (
                          <tr key={ev.event_id}>
                            <td>{ev.date_local}</td>
                            <td>{ev.event_type}</td>
                            <td>
                              {gtHitByEventId.has(ev.event_id)
                                ? gtHitByEventId.get(ev.event_id)
                                  ? 'yes'
                                  : 'no'
                                : '—'}
                            </td>
                            <td className="models-gt-desc">{ev.description}</td>
                          </tr>
                        ))}
                      </tbody>
                    </table>
                    {filteredGtEvents.length === 0 && (
                      <p className="models-empty-hint">No GT events for this location.</p>
                    )}
                  </div>
                )}
              </div>
            )}

            <div className="models-actions">
              {locationOptions.length > 0 && (
                <button
                  type="button"
                  className="btn-secondary models-action-btn"
                  onClick={openLocationCharts}
                >
                  <BarChart2 size={14} />
                  All classes at location
                </button>
              )}
              {selectedSeriesId && (
                <button
                  type="button"
                  className="btn-secondary models-action-btn"
                  onClick={openSeriesCharts}
                >
                  <BarChart2 size={14} />
                  Charts for selected series
                </button>
              )}
            </div>

            {manifest.metrics?.splits && manifest.metrics.splits.length > 0 && (
              <div className="models-metrics-collapsible">
                <button
                  type="button"
                  className="models-metrics-toggle"
                  onClick={() => setMetricsOpen((o) => !o)}
                  aria-expanded={metricsOpen}
                >
                  {metricsOpen ? 'Hide' : 'Show'} split metrics
                </button>
                {metricsOpen && (
                  <div className="models-splits-table-wrap">
                    <table className="models-table models-table-compact">
                      <thead>
                        <tr>
                          <th>Split</th>
                          <th>Rate</th>
                          <th>Points</th>
                          <th>p95</th>
                        </tr>
                      </thead>
                      <tbody>
                        {manifest.metrics.splits.map((row) => (
                          <tr key={row.split}>
                            <td>{row.split}</td>
                            <td>{pct(row.anomaly_rate)}</td>
                            <td>{row.points}</td>
                            <td>
                              {row.score_p95 != null && typeof row.score_p95 === 'number'
                                ? row.score_p95.toFixed(3)
                                : '—'}
                            </td>
                          </tr>
                        ))}
                      </tbody>
                    </table>
                  </div>
                )}
              </div>
            )}

            <div className="models-series-table-wrap models-series-table-wrap--primary">
              <div className="models-section-label">
                {hasAnomalies ? 'Series with anomalies' : 'Series'}
              </div>
              <table className="models-table">
                <thead>
                  <tr>
                    <th>Series</th>
                    <th>Rate</th>
                    <th>Flags</th>
                    <th>Score max</th>
                  </tr>
                </thead>
                <tbody>
                  {sortedSeries.map((row) => (
                    <tr
                      key={row.unique_id}
                      className={row.unique_id === selectedSeriesId ? 'selected' : ''}
                      onClick={() => onSelectSeries(row.unique_id)}
                    >
                      <td className="models-series-id">{seriesDisplayName(row.unique_id)}</td>
                      <td>{pct(row.rate)}</td>
                      <td>{row.n_anomalies}</td>
                      <td>{row.score_max != null ? Number(row.score_max).toFixed(3) : '—'}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
              {sortedSeries.length === 0 && (
                <p className="models-empty-hint">No series in this run.</p>
              )}
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
                {chartWindowPresets.map((key) => (
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
                onChange={(e) =>
                  setChartLayers((l) => ({ ...l, splitBands: e.target.checked }))
                }
              />
              Split bands
            </label>
            <label className="models-layer-check">
              <input
                type="checkbox"
                checked={chartLayers.gtBands}
                onChange={(e) =>
                  setChartLayers((l) => ({ ...l, gtBands: e.target.checked }))
                }
              />
              GT bands (blue)
            </label>
            <label className="models-layer-check">
              <input
                type="checkbox"
                checked={chartLayers.modelBands}
                onChange={(e) =>
                  setChartLayers((l) => ({ ...l, modelBands: e.target.checked }))
                }
              />
              Model bands (red)
            </label>
            <label className="models-layer-check">
              <input
                type="checkbox"
                checked={chartLayers.modelPoints}
                onChange={(e) =>
                  setChartLayers((l) => ({ ...l, modelPoints: e.target.checked }))
                }
              />
              Model points
            </label>
          </div>
        )}

        {inCharts && chartScope === 'location' && locationOptions.length > 0 && (
          <section className="models-charts-view">
            <div className="models-section-head">
              <span className="models-section-label">
                All classes at location
                {locationChart?.flags_in_split != null && (
                  <span className="models-chart-flags-hint">
                    {' '}
                    · {locationChart.flags_in_split} flags on {locationChart.split ?? 'test'}
                  </span>
                )}
              </span>
              <label className="models-location-picker">
                Location
                <select
                  className="models-select"
                  value={selectedLocationId ?? ''}
                  onChange={(e) => {
                    setSelectedLocationId(e.target.value);
                    loadLocationChart(e.target.value);
                  }}
                >
                  {locationOptions.map((loc) => (
                    <option key={loc} value={loc}>
                      {seriesDisplayName(loc)}
                    </option>
                  ))}
                </select>
              </label>
            </div>
            <div className="models-location-charts">
              <div className="models-chart-panel models-chart-panel--location">
                {locationChartLoading && (
                  <div className="models-chart-loading">Loading…</div>
                )}
                {!locationChartLoading && locationActualOption && (
                  <div className="models-chart-echarts-host">
                    <ReactECharts
                      option={locationActualOption}
                      style={CHART_FILL_STYLE}
                      notMerge
                      lazyUpdate
                    />
                  </div>
                )}
                {!locationChartLoading && !locationActualOption && (
                  <div className="models-empty-hint">No class-level chart data for this location.</div>
                )}
              </div>
              <div className="models-chart-panel models-chart-panel--location">
                {!locationChartLoading && locationPredOption && (
                  <div className="models-chart-echarts-host">
                    <ReactECharts
                      option={locationPredOption}
                      style={CHART_FILL_STYLE}
                      notMerge
                      lazyUpdate
                    />
                  </div>
                )}
              </div>
            </div>
          </section>
        )}

        {inCharts && chartScope === 'series' && selectedSeriesId && (
          <section className="models-charts-view">
            <div className="models-section-head models-section-head--compact">
              <span className="models-section-label">
                {seriesDisplayName(selectedSeriesId)}
                {scores?.flags_in_split != null && (
                  <span className="models-chart-flags-hint">
                    {' '}
                    · {scores.flags_in_split} flags on test
                    {scores.flags_visible != null &&
                    scores.flags_visible < scores.flags_in_split
                      ? ` (${scores.flags_visible} on chart)`
                      : ''}
                  </span>
                )}
              </span>
            </div>
            <div className="models-series-chart-stack">
              <div className="models-chart-panel models-chart-panel--fill">
                {scoresLoading && <div className="models-chart-loading">Loading…</div>}
                {!scoresLoading && forecastChartOption && (
                  <div className="models-chart-echarts-host">
                    <ReactECharts
                      option={forecastChartOption}
                      style={CHART_FILL_STYLE}
                      notMerge
                      lazyUpdate
                    />
                  </div>
                )}
                {!scoresLoading && !forecastChartOption && scoreChartOption && (
                  <div className="models-chart-echarts-host">
                    <ReactECharts
                      option={scoreChartOption}
                      style={CHART_FILL_STYLE}
                      notMerge
                      lazyUpdate
                    />
                  </div>
                )}
              </div>
              {!scoresLoading && forecastChartOption && scoreChartOption && (
                <div className="models-chart-panel models-chart-panel--fill models-chart-panel--score">
                  <div className="models-chart-panel-title">Anomaly score</div>
                  <div className="models-chart-echarts-host">
                    <ReactECharts
                      option={scoreChartOption}
                      style={CHART_FILL_STYLE}
                      notMerge
                      lazyUpdate
                    />
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
              {anomalyLoading
                ? 'Loading…'
                : `${anomalyList?.records.length ?? 0} flagged points`}
            </div>
          </div>
          <div className="models-anomaly-list">
            {anomalyLoading && (
              <p className="models-empty-hint">Loading anomalies…</p>
            )}
            {!anomalyLoading && (!anomalyList || anomalyList.records.length === 0) && (
              <p className="models-empty-hint">No anomalies for this series.</p>
            )}
            {!anomalyLoading &&
              anomalyList?.records.map((a, i) => {
                const kind = String(a.anomaly_kind ?? '').toLowerCase();
                const sev = String(a.severity ?? '').toLowerCase();
                const sevClass =
                  sev === 'alert' ? 'sev-alert' : sev === 'warning' ? 'sev-warn' : '';
                return (
                  <div key={`${a.ds}-${i}`} className="map-anomaly-card">
                    {a.ds && <div className="time">{a.ds}</div>}
                    {kind === 'macro' || kind === 'micro' ? (
                      <div className={`sev ${kind === 'macro' ? 'sev-alert' : 'sev-warn'}`}>
                        {kind}
                      </div>
                    ) : null}
                    {a.severity && (
                      <div className={`sev ${sevClass}`}>{String(a.severity)}</div>
                    )}
                    <div className="summary">
                      score {a.score?.toFixed(4) ?? '—'}
                      {a.z != null && ` · z ${a.z.toFixed(2)}`}
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
