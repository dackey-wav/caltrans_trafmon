export type LaneInfo = {
  location_id: string;
  lane: string;
};

export type StationGroup = {
  base_id: string;
  label: string;
  lanes: LaneInfo[];
};

export type DatasetInfo = {
  id: string;
  label: string;
  table: string;
  station_count: number;
  location_count: number;
  stations: StationGroup[];
};

/** @deprecated Use DatasetInfo */
export type RegionInfo = DatasetInfo;

export type RangePresetId = '15m' | '1h' | '1d' | '7d' | '30d';

export type RangePresetMeta = {
  hours: number;
  resolution: string;
  label: string;
};

export type VehicleClassMeta = {
  labels: Record<string, string>;
  colors: Record<string, string>;
};

export type DatasetsResponse = {
  datasets: DatasetInfo[];
  range_presets: Record<RangePresetId, RangePresetMeta>;
  max_points_hint: number;
  vehicle_classes?: VehicleClassMeta;
};

/** @deprecated Use DatasetsResponse */
export type RegionsResponse = DatasetsResponse & { regions?: DatasetInfo[] };

export type TrafficResponse = {
  timestamps: string[];
  series: Record<string, number[]>;
  meta?: {
    dataset: string;
    resolution: string;
    range_preset?: string;
    date_from: string;
    date_to: string;
    points: number;
    series_count: number;
  };
};

export type ClassTrafficResponse = {
  timestamps: string[];
  classes: number[];
  series: Record<string, number[]>;
  class_labels?: Record<string, string>;
  class_colors?: Record<string, string>;
  meta?: {
    dataset: string;
    resolution: string;
    range_preset?: string;
    date_from: string;
    date_to: string;
    points: number;
    class_count: number;
    station_count: number;
  };
};

export type ChartMode = 'count' | 'classes' | 'map';

export type AppSection = 'traffic' | 'models';

export type AnomalyMode = 'percentile' | 'combined';

export type ModelFamily = 'anomaly' | 'forecasting';

export type LocalModelRunSummary = {
  run_id: string;
  modified_at?: string | null;
  /** anomaly family */
  anomaly_rate?: number | null;
  series_count?: number | null;
  eval_split?: string | null;
  /** forecasting family */
  mae?: number | null;
  rmse?: number | null;
  smape?: number | null;
  has_eval?: boolean | null;
};

export type LocalModelInfo = {
  id: string;
  label: string;
  family: ModelFamily;
  tagline?: string;
  focus?: string;
  artifacts_root: string;
  run_count?: number;
  runs: LocalModelRunSummary[];
};

export type ModelsLocalResponse = {
  models: LocalModelInfo[];
};

export type ForecastMacro = {
  series_count?: number;
  mae?: number;
  rmse?: number;
  smape?: number;
};

export type ForecastRunManifest = {
  model: string;
  label?: string;
  family: 'forecasting';
  run_id: string;
  artifacts: string[];
  dataset_meta?: {
    locations?: string[];
    ds_min?: string;
    ds_max?: string;
    unique_id_count?: number;
    rows?: number;
  } | null;
  comparison?: Record<string, unknown> | null;
  macro?: ForecastMacro | null;
  macro_model?: ForecastMacro | null;
  macro_baseline?: ForecastMacro | null;
  delta?: { mae?: number; rmse?: number; smape?: number } | null;
  scopes: string[];
  default_scope: string;
  has_anomalies?: boolean;
  anomaly_params?: { method?: string; threshold?: number; max_records?: number | null } | null;
  anomaly_scopes?: string[];
  has_domain_anomalies?: boolean;
  domain_summary?: Record<string, unknown> | null;
  has_predictions?: boolean;
};

export type ForecastSeriesRow = {
  unique_id: string;
  n: number;
  mae: number;
  rmse: number;
  smape: number;
  n_anomalies: number;
  rate: number;
};

export type ForecastSeriesResponse = {
  run_id: string;
  model: string;
  scope: string;
  scopes: string[];
  per_series: ForecastSeriesRow[];
  macro?: ForecastMacro | null;
};

export type ForecastChartResponse = {
  run_id: string;
  model: string;
  scope: string;
  scopes: string[];
  unique_id: string;
  timestamps: string[];
  actual: number[];
  predicted: number[];
  baseline: number[];
  residual: number[];
  score: number[];
  severity: Array<string | null>;
  is_anomaly: boolean[];
  anomalies: Array<{ ds: string; score: number; residual: number; severity?: string | null }>;
  points: number;
  points_in_window?: number;
  points_total: number;
  window?: string;
  split_ranges: Array<{ split: string; start: number; end: number }>;
};

export type TimesnetPerSeriesRow = {
  unique_id: string;
  n: number;
  n_anomalies: number;
  rate: number;
  score_max: number;
  score_median: number;
  cutoff: number;
};

export type GtEvent = {
  event_id: number;
  source_sheet?: string;
  date_local: string;
  site_id: string;
  direction?: string;
  gt_location_key: string;
  road_location?: string;
  time_start_local?: string;
  time_end_local?: string;
  time_start_utc: string;
  time_end_utc: string;
  event_type: string;
  hourly_visible: string;
  description: string;
  match_prefixes: string;
};

export type GtEventsResponse = {
  csv_path: string;
  count: number;
  events: GtEvent[];
};

export type GtEvalModeBlock = {
  summary: { total: number; hits: number; misses: number; hit_rate: number };
  summary_hourly_visible?: { total: number; hits: number; misses: number; hit_rate: number };
  events: Array<{
    event_id: number;
    hit: boolean;
    date_local: string;
    site_id: string;
    gt_location_key: string;
    event_type: string;
    hourly_visible: string;
  }>;
};

export type GtEvalScope = {
  start?: string | null;
  end?: string | null;
  locations_count?: number;
  gt_events_total_csv?: number;
  gt_events_in_scope?: number;
};

export type GtEvalPayload = {
  run_id?: string;
  run_dir?: string;
  gt_csv?: string;
  eval_split?: string;
  window_pad_hours?: number;
  scope?: GtEvalScope;
  modes: Record<string, GtEvalModeBlock>;
};

export type TimesnetRunManifest = {
  model: string;
  run_id: string;
  schema_version?: string;
  artifacts: string[];
  has_scores: boolean;
  anomaly_modes?: string[];
  default_mode?: string;
  has_gt_eval?: boolean;
  summary_percentile?: TimesnetRunManifest['summary'];
  summary_combined?: TimesnetRunManifest['summary'];
  combined_robust_z?: number;
  gt_eval?: GtEvalPayload | null;
  comparison?: Record<string, unknown> | null;
  metrics?: {
    splits?: Array<{
      split: string;
      points: number;
      series: number;
      anomalies: number;
      anomaly_rate: number;
      score_mean: number;
      score_p95: number;
      score_p99: number;
    }>;
    meta?: {
      model_family?: string;
      [key: string]: unknown;
    };
    extra?: Record<string, unknown>;
  } | null;
  dataset_meta?: {
    locations?: string[];
    freq?: string;
    date_from?: string | null;
    date_to?: string | null;
    ds_min?: string;
    ds_max?: string;
    n_targets?: number;
    rows?: number;
  } | null;
  thresholds?: Record<string, unknown> | null;
  summary?: {
    series_count?: number;
    point_count?: number;
    anomaly_count?: number;
    anomaly_rate?: number;
  } | null;
  threshold?: {
    method?: string;
    profile?: string;
    threshold?: number | null;
    quantile?: number | null;
  } | null;
  top_series_by_rate?: TimesnetPerSeriesRow[];
  score_quantiles_by_split?: Record<
    string,
    { p95?: number; p99?: number; anomaly_rate?: number }
  >;
};

export type TimesnetSeriesListResponse = {
  run_id: string;
  mode?: string;
  per_series: TimesnetPerSeriesRow[];
  summary?: TimesnetRunManifest['summary'];
};

export type TimesnetScoresResponse = {
  run_id: string;
  mode?: string;
  unique_id: string;
  cutoff: number;
  timestamps: string[];
  score: number[];
  actual?: number[];
  predicted?: number[];
  value_kind?: 'forecast' | 'reconstruction';
  severity?: Array<string | null>;
  split: string[];
  is_anomaly: boolean[];
  anomalies: Array<{
    ds: string;
    score: number;
    z: number;
    severity?: string | null;
  }>;
  points: number;
  points_in_window?: number;
  points_total: number;
  window?: string;
  flags_in_split?: number;
  flags_visible?: number;
  splits_present: string[];
  split_ranges: Array<{ split: string; start: number; end: number }>;
};

export type TimesnetLocationClassSeries = {
  class_id: number;
  unique_id: string;
  label: string;
  color: string;
  actual: number[];
  predicted: number[];
};

export type TimesnetLocationAnomalyMarker = {
  index: number;
  classes: number[];
  label: string;
  mode: string;
};

export type TimesnetLocationChartResponse = {
  run_id: string;
  location_id: string;
  split?: string | null;
  timestamps: string[];
  classes: TimesnetLocationClassSeries[];
  total_actual: number[];
  total_predicted: number[];
  anomaly_markers: TimesnetLocationAnomalyMarker[];
  points: number;
  points_in_window?: number;
  points_total?: number;
  window?: string;
  split_ranges?: Array<{ split: string; start: number; end: number }>;
  flags_in_split?: number;
  flags_visible?: number;
};

export type TimesnetAnomaliesListResponse = {
  run_id: string;
  unique_id?: string | null;
  records: Array<{
    unique_id?: string;
    ds?: string;
    score?: number;
    z?: number;
    severity?: string;
    cutoff?: number;
    anomaly_kind?: 'macro' | 'micro';
    location_id?: string;
    vehicle_class?: number;
  }>;
  total: number;
};

export type TimesnetGraphNode = {
  id: string;
  station_id: string | null;
  lat: number | null;
  lon: number | null;
  macro_count: number;
  micro_count: number;
};

export type TimesnetGraphEdge = {
  source: string;
  target: string;
  weight: number;
};

export type TimesnetGraphResponse = {
  run_id: string;
  mode: AnomalyMode;
  reconstructed: boolean;
  has_adaptive: boolean;
  nodes: TimesnetGraphNode[];
  classes: number[];
  edges_static: TimesnetGraphEdge[];
  edges_adaptive: TimesnetGraphEdge[];
};

export type TimesnetGraphAnomalySeries = {
  node: string;
  is_center: boolean;
  edge_weight: number;
  cutoff: number;
  timestamps: string[];
  score: number[];
};

export type TimesnetGraphAnomalyResponse = {
  run_id: string;
  mode: AnomalyMode;
  center_node: string;
  center_ds: string;
  neighbours: Array<{ node: string; weight: number }>;
  series: TimesnetGraphAnomalySeries[];
  propagation_order: Array<{ node: string; first_cross: string }>;
};

export type MapAnomalyRecord = {
  id?: string;
  ds_utc?: string;
  summary?: string;
  severity?: string;
  location_ids?: string[];
  [key: string]: unknown;
};

export type MapStation = {
  station_id: string;
  road?: string | null;
  place?: string | null;
  segment?: string | null;
  right?: { lat: number; lon: number } | null;
  left?: { lat: number; lon: number } | null;
  lat: number | null;
  lon: number | null;
  anomaly_count: number;
  anomalies: MapAnomalyRecord[];
};

export type MapResponse = {
  dataset: string;
  source_note?: string | null;
  updated_at?: string | null;
  message?: string | null;
  stations: MapStation[];
};

export type ClassStationSelection = {
  datasetId: string;
  baseId: string;
  label: string;
  laneIds: string[];
  /** 'all' aggregates every lane; otherwise a single location_id */
  laneScope: 'all' | string;
};
