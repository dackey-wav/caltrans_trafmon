import { useCallback, useEffect, useMemo, useState } from 'react';
import ReactECharts from 'echarts-for-react';
import {
  Activity,
  ChevronRight,
  ChevronLeft,
} from 'lucide-react';
import {
  fetchClassTraffic,
  fetchMap,
  fetchModelsLocal,
  fetchTimesnetAnomalies,
  fetchTimesnetRun,
  fetchTimesnetSeries,
  fetchDatasets,
  fetchTraffic,
} from './api';
import { buildClassChartOption, buildCountChartOption } from './charts';
import { ModelsWorkspace } from './ModelsWorkspace';
import { ModelPicker } from './ModelPicker';
import { ForecastWorkspace } from './ForecastWorkspace';
import { StationMap } from './StationMap';
import { ModelsSidebar } from './models/ModelsSidebar';
import { TrafficSidebar } from './traffic/TrafficSidebar';
import type {
  AnomalyMode,
  AppSection,
  ChartMode,
  ClassStationSelection,
  LocalModelInfo,
  MapResponse,
  TimesnetAnomaliesListResponse,
  TimesnetPerSeriesRow,
  TimesnetRunManifest,
  RangePresetId,
  DatasetInfo,
  StationGroup,
} from './types';
import './index.css';

const PRESET_ORDER: RangePresetId[] = ['15m', '1h', '1d', '7d', '30d'];

function App() {
  const [datasets, setDatasets] = useState<DatasetInfo[]>([]);
  const [presetMeta, setPresetMeta] = useState<Record<string, { label: string }>>({});
  const [expandedDatasets, setExpandedDatasets] = useState<Record<string, boolean>>({});
  const [expandedBases, setExpandedBases] = useState<Record<string, boolean>>({});

  const [appSection, setAppSection] = useState<AppSection>('traffic');
  const [chartMode, setChartMode] = useState<ChartMode | null>(null);
  const [sidebarPanel, setSidebarPanel] = useState<'menu' | 'locations'>('menu');

  const [models, setModels] = useState<LocalModelInfo[]>([]);
  const [selectedModelId, setSelectedModelId] = useState<string | null>(null);
  const [selectedRunId, setSelectedRunId] = useState<string | null>(null);
  const [runManifest, setRunManifest] = useState<TimesnetRunManifest | null>(null);
  const [perSeries, setPerSeries] = useState<TimesnetPerSeriesRow[]>([]);
  const [selectedSeriesId, setSelectedSeriesId] = useState<string | null>(null);
  const [seriesAnomalies, setSeriesAnomalies] = useState<TimesnetAnomaliesListResponse | null>(null);
  const [modelsLoading, setModelsLoading] = useState(false);
  const [manifestLoading, setManifestLoading] = useState(false);
  const [anomalyLoading, setAnomalyLoading] = useState(false);
  const [anomalyMode, setAnomalyMode] = useState<AnomalyMode>('combined');

  const [activeDataset, setActiveDataset] = useState<string | null>(null);
  const [selectedLocations, setSelectedLocations] = useState<Set<string>>(new Set());
  const [classSelection, setClassSelection] = useState<ClassStationSelection | null>(null);

  const [rangePreset, setRangePreset] = useState<RangePresetId>('7d');
  const [endDate, setEndDate] = useState('2024-12-31');
  const [endHour, setEndHour] = useState(12);

  const [chartOption, setChartOption] = useState<Record<string, unknown> | null>(null);
  const [metaLine, setMetaLine] = useState('');
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const [mapData, setMapData] = useState<MapResponse | null>(null);
  const [mapLoading, setMapLoading] = useState(false);
  const [mapStationId, setMapStationId] = useState<string | null>(null);


  useEffect(() => {
    fetchDatasets()
      .then((data) => {
        setDatasets(data.datasets);
        setPresetMeta(data.range_presets);
        if (data.datasets.length > 0) {
          setExpandedDatasets({ [data.datasets[0].id]: true });
        }
      })
      .catch((e) => setError(String(e)));
  }, []);

  const mapDataset = useMemo(() => {
    if (datasets.some((d) => d.id === 'dataset-a')) return 'dataset-a';
    return datasets[0]?.id ?? 'dataset-a';
  }, [datasets]);

  const selectedModel = useMemo(
    () => models.find((m) => m.id === selectedModelId) ?? null,
    [models, selectedModelId],
  );
  const modelFamily = selectedModel?.family ?? null;
  const modelRuns = useMemo(() => selectedModel?.runs ?? [], [selectedModel]);

  const openMode = (mode: ChartMode) => {
    setAppSection('traffic');
    setChartMode(mode);
    setMapStationId(null);
    setChartOption(null);
    setMetaLine('');
    if (mode === 'map') {
      setSidebarPanel('menu');
      setSelectedLocations(new Set());
      setClassSelection(null);
    } else if (mode === 'classes') {
      setSidebarPanel('locations');
      setSelectedLocations(new Set());
      setClassSelection(null);
    } else {
      setSidebarPanel('locations');
      setClassSelection(null);
    }
  };

  const resetRunState = () => {
    setSelectedRunId(null);
    setRunManifest(null);
    setPerSeries([]);
    setSelectedSeriesId(null);
    setSeriesAnomalies(null);
  };

  const openModels = () => {
    setAppSection('models');
    setChartMode(null);
    setSidebarPanel('menu');
    setChartOption(null);
    setMetaLine('');
    setError(null);
    setSelectedModelId(null);
    resetRunState();
  };

  const selectModel = (modelId: string) => {
    setSelectedModelId(modelId);
    resetRunState();
    setError(null);
  };

  const backToModels = () => {
    setSelectedModelId(null);
    resetRunState();
    setError(null);
  };

  const backToTrafficMenu = () => {
    setAppSection('traffic');
    setSidebarPanel('menu');
  };

  const backToMenu = () => {
    setSidebarPanel('menu');
  };

  const loadModelsIndex = useCallback(() => {
    setModelsLoading(true);
    setError(null);
    fetchModelsLocal()
      .then((data) => {
        setModels(data.models);
      })
      .catch((e) => {
        setError(String(e));
        setModels([]);
      })
      .finally(() => setModelsLoading(false));
  }, []);

  const loadAnomalyRun = useCallback(
    (runId: string, modelId: string) => {
      setSelectedRunId(runId);
      setSelectedSeriesId(null);
      setSeriesAnomalies(null);
      setManifestLoading(true);
      setError(null);
      Promise.all([fetchTimesnetRun(runId, modelId), fetchTimesnetSeries(runId, anomalyMode, modelId)])
        .then(([manifest, series]) => {
          setRunManifest(manifest);
          setPerSeries(series.per_series);
        })
        .catch((e) => {
          setError(String(e));
          setRunManifest(null);
          setPerSeries([]);
        })
        .finally(() => setManifestLoading(false));
    },
    [anomalyMode],
  );

  const selectRun = useCallback(
    (runId: string) => {
      if (modelFamily === 'forecasting') {
        // ForecastWorkspace loads its own manifest/series/charts.
        setSelectedRunId(runId);
        setRunManifest(null);
        setPerSeries([]);
        setSelectedSeriesId(null);
        setSeriesAnomalies(null);
        setError(null);
        return;
      }
      if (selectedModelId) loadAnomalyRun(runId, selectedModelId);
    },
    [modelFamily, selectedModelId, loadAnomalyRun],
  );

  const handleAnomalyModeChange = useCallback(
    (mode: AnomalyMode) => {
      setAnomalyMode(mode);
      if (!selectedRunId || !selectedModelId) return;
      fetchTimesnetSeries(selectedRunId, mode, selectedModelId)
        .then((series) => setPerSeries(series.per_series))
        .catch(() => {});
      if (selectedSeriesId) {
        setAnomalyLoading(true);
        fetchTimesnetAnomalies(selectedRunId, selectedSeriesId, 200, mode, selectedModelId)
          .then(setSeriesAnomalies)
          .catch(() => setSeriesAnomalies(null))
          .finally(() => setAnomalyLoading(false));
      }
    },
    [selectedRunId, selectedSeriesId, selectedModelId],
  );

  const selectSeries = useCallback(
    (uniqueId: string) => {
      if (!selectedRunId || !selectedModelId) return;
      if (selectedSeriesId === uniqueId) {
        setSelectedSeriesId(null);
        setSeriesAnomalies(null);
        return;
      }
      setSelectedSeriesId(uniqueId);
      setAnomalyLoading(true);
      setError(null);
      fetchTimesnetAnomalies(selectedRunId, uniqueId, 200, anomalyMode, selectedModelId)
        .then(setSeriesAnomalies)
        .catch((e) => {
          setError(e instanceof Error ? e.message : String(e));
          setSeriesAnomalies(null);
        })
        .finally(() => setAnomalyLoading(false));
    },
    [selectedRunId, selectedSeriesId, anomalyMode, selectedModelId],
  );

  useEffect(() => {
    if (appSection !== 'models') return;
    void Promise.resolve().then(loadModelsIndex);
  }, [appSection, loadModelsIndex]);

  useEffect(() => {
    if (appSection !== 'models' || !selectedModelId || selectedRunId) return;
    if (modelRuns.length > 0) {
      void Promise.resolve().then(() => selectRun(modelRuns[0].run_id));
    }
  }, [appSection, selectedModelId, modelRuns, selectedRunId, selectRun]);

  const toggleDataset = (id: string) => {
    setExpandedDatasets((prev) => ({ ...prev, [id]: !prev[id] }));
  };

  const toggleBase = (baseId: string) => {
    setExpandedBases((prev) => ({ ...prev, [baseId]: !prev[baseId] }));
  };

  const setSelectionForBase = useCallback(
    (datasetId: string, laneIds: string[], checked: boolean) => {
      setSelectedLocations((prev) => {
        const next = new Set(prev);
        if (checked) {
          if (activeDataset && activeDataset !== datasetId) {
            next.clear();
          }
          laneIds.forEach((id) => next.add(id));
          setActiveDataset(datasetId);
        } else {
          laneIds.forEach((id) => next.delete(id));
        }
        return next;
      });
    },
    [activeDataset],
  );

  const toggleLane = (datasetId: string, locationId: string) => {
    setSelectedLocations((prev) => {
      const next = new Set(prev);
      if (next.has(locationId)) {
        next.delete(locationId);
      } else {
        if (activeDataset && activeDataset !== datasetId && next.size > 0) {
          next.clear();
        }
        next.add(locationId);
        setActiveDataset(datasetId);
      }
      return next;
    });
  };

  const selectClassStation = (datasetId: string, station: StationGroup) => {
    const laneIds = station.lanes.map((l) => l.location_id);
    setClassSelection({
      datasetId,
      baseId: station.base_id,
      label: station.label,
      laneIds,
      laneScope: 'all',
    });
    setActiveDataset(datasetId);
    setChartOption(null);
    setMetaLine('');
  };

  const setClassLaneScope = (scope: 'all' | string) => {
    setClassSelection((prev) => (prev ? { ...prev, laneScope: scope } : prev));
    setChartOption(null);
    setMetaLine('');
  };

  const shiftEndDate = (days: number) => {
    const d = new Date(`${endDate}T12:00:00Z`);
    d.setUTCDate(d.getUTCDate() + days);
    setEndDate(d.toISOString().slice(0, 10));
  };

  const shiftEndHour = (hours: number) => {
    const d = new Date(`${endDate}T${String(endHour).padStart(2, '0')}:00:00Z`);
    d.setUTCHours(d.getUTCHours() + hours);
    setEndDate(d.toISOString().slice(0, 10));
    setEndHour(d.getUTCHours());
  };

  const shiftRange = (delta: number) => {
    if (rangePreset === '1h') {
      shiftEndHour(delta);
    } else {
      shiftEndDate(delta);
    }
  };

  const formatMeta = (m: {
    points: number;
    resolution: string;
    date_from: string;
    date_to: string;
  }) => {
    const from = m.date_from.slice(0, 16).replace('T', ' ');
    const to = m.date_to.slice(0, 16).replace('T', ' ');
    return `${m.points} points · ${m.resolution} · ${from} → ${to}`;
  };

  const loadData = useCallback(() => {
    if (chartMode === 'count') {
      if (!activeDataset || selectedLocations.size === 0) return;
      setLoading(true);
      setError(null);
      fetchTraffic({
        dataset: activeDataset,
        stations: Array.from(selectedLocations),
        rangePreset,
        endDate,
        endHour,
      })
        .then((data) => {
          if (!data.timestamps?.length) {
            setChartOption(null);
            setMetaLine('No data for selection');
            return;
          }
          setChartOption(buildCountChartOption(data));
          setMetaLine(data.meta ? formatMeta(data.meta) : '');
        })
        .catch((e) => {
          setError(String(e));
          setChartOption(null);
        })
        .finally(() => setLoading(false));
      return;
    }

    if (chartMode === 'classes' && classSelection) {
      const stations =
        classSelection.laneScope === 'all'
          ? classSelection.laneIds
          : [classSelection.laneScope];
      setLoading(true);
      setError(null);
      fetchClassTraffic({
        dataset: classSelection.datasetId,
        stations,
        rangePreset,
        endDate,
        endHour,
      })
        .then((data) => {
          if (!data.timestamps?.length) {
            setChartOption(null);
            setMetaLine('No data for selection');
            return;
          }
          setChartOption(buildClassChartOption(data));
          setMetaLine(data.meta ? formatMeta(data.meta) : '');
        })
        .catch((e) => {
          setError(String(e));
          setChartOption(null);
        })
        .finally(() => setLoading(false));
    }
  }, [chartMode, activeDataset, selectedLocations, classSelection, rangePreset, endDate, endHour]);

  const loadMap = useCallback(() => {
    setMapLoading(true);
    setError(null);
    fetchMap(mapDataset)
      .then(setMapData)
      .catch((e) => {
        setError(String(e));
        setMapData(null);
      })
      .finally(() => setMapLoading(false));
  }, [mapDataset]);

  useEffect(() => {
    if (chartMode !== 'map') return;
    void Promise.resolve().then(loadMap);
  }, [chartMode, loadMap]);

  const selectMapStation = useCallback((id: string | null) => {
    setMapStationId(id);
  }, []);

  const canRefresh =
    chartMode === 'count'
      ? selectedLocations.size > 0
      : chartMode === 'classes' && classSelection !== null;

  const selectedCount = selectedLocations.size;

  const datasetLabel = useMemo(
    () => datasets.find((d) => d.id === activeDataset)?.label ?? '',
    [datasets, activeDataset],
  );

  const classLaneLabel = useMemo(() => {
    if (!classSelection || classSelection.laneScope === 'all') return 'all lanes';
    const lane = classSelection.laneIds.find((id) => id === classSelection.laneScope);
    if (!lane) return classSelection.laneScope;
    const part = lane.split('-').pop();
    return part ?? lane;
  }, [classSelection]);

  const selectedMapStation = useMemo(
    () => mapData?.stations.find((s) => s.station_id === mapStationId),
    [mapData, mapStationId],
  );

  const emptyHint =
    appSection === 'models'
      ? 'Select a model, then a run'
      : chartMode === null
      ? 'Choose a view in the sidebar'
      : chartMode === 'map'
        ? 'Use Reload map if needed'
        : chartMode === 'count'
          ? selectedCount === 0
            ? 'Select one or more locations, then Refresh'
            : 'Click Refresh to load traffic'
          : !classSelection
            ? 'Select a location, then Refresh'
            : 'Click Refresh to load class breakdown';

  return (
    <div className="dashboard-container">
      <aside className="sidebar">
        <div className="sidebar-header">
          <Activity className="logo-icon" size={18} />
          <span>Traffic</span>
        </div>

        <div className="sidebar-viewport">
          {appSection === 'models' ? (
            <ModelsSidebar
              models={models}
              selectedModel={selectedModel}
              selectedRunId={selectedRunId}
              loading={modelsLoading}
              onBackToTraffic={backToTrafficMenu}
              onBackToModels={backToModels}
              onSelectModel={selectModel}
              onSelectRun={selectRun}
            />
          ) : (
            <TrafficSidebar
              chartMode={chartMode}
              panel={sidebarPanel}
              datasets={datasets}
              expandedDatasets={expandedDatasets}
              expandedBases={expandedBases}
              selectedLocations={selectedLocations}
              classSelection={classSelection}
              onOpenMode={openMode}
              onOpenModels={openModels}
              onBack={backToMenu}
              onToggleDataset={toggleDataset}
              onToggleBase={toggleBase}
              onSetBaseSelection={setSelectionForBase}
              onToggleLane={toggleLane}
              onSelectClassStation={selectClassStation}
              onSetClassLaneScope={setClassLaneScope}
            />
          )}
        </div>
      </aside>

      <main className="main-content">
        <header
          className={`topbar${chartMode === 'map' ? ' topbar--map' : ''}${appSection === 'models' ? ' topbar--models' : ''}`}
        >
          <div className="controls">
            {appSection === 'models' ? (
              <button
                type="button"
                className="btn-primary"
                onClick={loadModelsIndex}
                disabled={modelsLoading}
              >
                {modelsLoading ? 'Loading…' : 'Refresh runs'}
              </button>
            ) : chartMode === 'map' ? (
              <button
                type="button"
                className="btn-primary"
                onClick={loadMap}
                disabled={mapLoading}
              >
                {mapLoading ? 'Loading…' : 'Reload map'}
              </button>
            ) : (
              <>
                <div className="preset-group">
                  {PRESET_ORDER.map((id) => (
                    <button
                      key={id}
                      type="button"
                      className={`preset-btn ${rangePreset === id ? 'active' : ''}`}
                      onClick={() => setRangePreset(id)}
                      title={presetMeta[id]?.label ?? id}
                      disabled={chartMode === null}
                    >
                      {id}
                    </button>
                  ))}
                </div>

                <div className="date-nav">
                  <button
                    type="button"
                    className="icon-btn"
                    onClick={() => shiftRange(-1)}
                    aria-label={rangePreset === '1h' ? 'Previous hour' : 'Previous day'}
                    disabled={chartMode === null}
                  >
                    <ChevronLeft size={16} />
                  </button>
                  <div className="date-field">
                    <span className="date-label">End</span>
                    <div className="date-inputs">
                      <input
                        type="date"
                        className="date-input"
                        value={endDate}
                        onChange={(e) => setEndDate(e.target.value)}
                        disabled={chartMode === null}
                      />
                      {rangePreset === '1h' && (
                        <input
                          type="number"
                          className="hour-input"
                          min={0}
                          max={23}
                          value={endHour}
                          onChange={(e) => {
                            const v = Number(e.target.value);
                            if (!Number.isNaN(v)) setEndHour(Math.min(23, Math.max(0, v)));
                          }}
                          disabled={chartMode === null}
                          aria-label="End hour UTC"
                        />
                      )}
                    </div>
                  </div>
                  <button
                    type="button"
                    className="icon-btn"
                    onClick={() => shiftRange(1)}
                    aria-label={rangePreset === '1h' ? 'Next hour' : 'Next day'}
                    disabled={chartMode === null}
                  >
                    <ChevronRight size={16} />
                  </button>
                </div>

                <button
                  type="button"
                  className="btn-primary"
                  onClick={loadData}
                  disabled={loading || !canRefresh}
                >
                  {loading ? 'Loading…' : 'Refresh'}
                </button>
              </>
            )}
          </div>

          {appSection === 'models' && selectedRunId && (
            <div className="meta-line subtle">
              Run <code style={{ fontSize: '0.8rem' }}>{selectedRunId}</code>
              {runManifest?.has_scores === false && ' · scores.parquet missing'}
            </div>
          )}
          {chartMode === 'map' && mapData?.message && (
            <div className="meta-line">{mapData.message}</div>
          )}
          {chartMode === 'map' && mapData?.source_note && (
            <div className="meta-line subtle">{mapData.source_note}</div>
          )}
          {appSection === 'traffic' && chartMode !== 'map' && metaLine && (
            <div className="meta-line">{metaLine}</div>
          )}
          {chartMode === 'count' && datasetLabel && selectedCount > 0 && (
            <div className="meta-line subtle">
              {datasetLabel} · {selectedCount} series
            </div>
          )}
          {chartMode === 'classes' && classSelection && (
            <div className="meta-line subtle">
              {datasets.find((d) => d.id === classSelection.datasetId)?.label} ·{' '}
              {classSelection.label} · {classLaneLabel}
            </div>
          )}
        </header>

        {error && <div className="banner-error">{error}</div>}

        <div
          className={`chart-container${chartMode === 'map' ? ' chart-container--map' : ''}${appSection === 'models' ? ' chart-container--models' : ''}`}
        >
          {appSection === 'models' ? (
            <div className="chart-card models-shell">
              {!selectedModel ? (
                <ModelPicker models={models} loading={modelsLoading} onSelect={selectModel} />
              ) : modelFamily === 'forecasting' ? (
                <ForecastWorkspace
                  modelId={selectedModel.id}
                  modelLabel={selectedModel.label}
                  runId={selectedRunId}
                />
              ) : (
                <ModelsWorkspace
                  modelId={selectedModel.id}
                  modelLabel={selectedModel.label}
                  runId={selectedRunId}
                  manifest={runManifest}
                  perSeries={perSeries}
                  selectedSeriesId={selectedSeriesId}
                  onSelectSeries={selectSeries}
                  anomalyList={seriesAnomalies}
                  anomalyLoading={anomalyLoading}
                  manifestLoading={manifestLoading}
                  anomalyMode={anomalyMode}
                  onAnomalyModeChange={handleAnomalyModeChange}
                />
              )}
            </div>
          ) : chartMode === 'map' ? (
            <div className="chart-card map-shell">
              {mapLoading && <div className="map-loading">Loading map…</div>}
              {!mapLoading && mapData && mapData.stations.length > 0 && (
                <div className="map-split">
                  <div className="map-split-main">
                    <StationMap
                      stations={mapData.stations}
                      selectedStationId={mapStationId}
                      onSelectStation={selectMapStation}
                    />
                  </div>
                  <aside className="map-detail">
                    {mapStationId && selectedMapStation ? (
                      <>
                        <div className="map-detail-header">
                          Station {selectedMapStation.station_id}
                          <div className="map-detail-sub">
                            {[selectedMapStation.road, selectedMapStation.place]
                              .filter(Boolean)
                              .join(' · ')}
                          </div>
                        </div>
                        <div className="map-detail-scroll">
                          {selectedMapStation.segment && (
                            <p
                              style={{
                                fontSize: '0.75rem',
                                color: 'var(--text-muted)',
                                margin: '0 0 12px',
                                lineHeight: 1.4,
                              }}
                            >
                              {selectedMapStation.segment}
                            </p>
                          )}
                          {selectedMapStation.anomalies.length === 0 ? (
                            <p style={{ color: 'var(--text-faint)', fontSize: '0.82rem', margin: 0 }}>
                              No anomaly records for this station. Add entries under{' '}
                              <code style={{ fontSize: '0.7rem' }}>by_station</code> in{' '}
                              <code style={{ fontSize: '0.7rem' }}>dataset_a_map_anomalies.json</code>.
                            </p>
                          ) : (
                            selectedMapStation.anomalies.map((a, i) => {
                              const sev = String(a.severity ?? '').toLowerCase();
                              const sevClass =
                                sev === 'alert' ? 'sev-alert' : sev === 'warning' ? 'sev-warn' : '';
                              return (
                                <div key={String(a.id ?? `a-${i}`)} className="map-anomaly-card">
                                  {a.ds_utc && <div className="time">{a.ds_utc}</div>}
                                  {a.severity && (
                                    <div className={`sev ${sevClass}`}>{String(a.severity)}</div>
                                  )}
                                  {a.summary && <div className="summary">{String(a.summary)}</div>}
                                  {Array.isArray(a.location_ids) && a.location_ids.length > 0 && (
                                    <div className="map-anomaly-locs">
                                      {a.location_ids.join(', ')}
                                    </div>
                                  )}
                                </div>
                              );
                            })
                          )}
                        </div>
                      </>
                    ) : (
                      <div className="map-detail-placeholder">
                        Click a station pin to see anomaly details. Badge = number of records in
                        config for that station.
                      </div>
                    )}
                  </aside>
                </div>
              )}
              {!mapLoading && mapData && mapData.stations.length === 0 && (
                <div className="empty-state">{mapData.message ?? 'No stations for this dataset.'}</div>
              )}
              {!mapLoading && !mapData && <div className="empty-state">Could not load map.</div>}
            </div>
          ) : (
            <div className="chart-card">
              {chartOption ? (
                <ReactECharts
                  option={chartOption}
                  style={{ height: '100%', width: '100%' }}
                  notMerge
                  lazyUpdate
                />
              ) : (
                <div className="empty-state">{emptyHint}</div>
              )}
            </div>
          )}
        </div>
      </main>
    </div>
  );
}

export default App;
