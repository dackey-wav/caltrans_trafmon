import type {
  ClassTrafficResponse,
  ForecastChartResponse,
  ForecastRunManifest,
  ForecastSeriesResponse,
  MapResponse,
  ModelsLocalResponse,
  TimesnetAnomaliesListResponse,
  TimesnetLocationChartResponse,
  TimesnetRunManifest,
  TimesnetScoresResponse,
  TimesnetSeriesListResponse,
  GtEventsResponse,
  RangePresetId,
  DatasetsResponse,
  TrafficResponse,
} from './types';

/** Same-origin /api in dev (Vite proxy); override with VITE_API_BASE if needed. */
const API_BASE = import.meta.env.VITE_API_BASE ?? '/api';

const DEFAULT_ANOMALY_MODEL = 'timesnet';

function modelRunQuery(runId: string, extra?: Record<string, string>): string {
  const qs = new URLSearchParams({ run_id: runId, ...extra });
  return qs.toString();
}

/** UTC end timestamp sent to traffic API (must match backend window rules). */
export function buildTrafficEndUtc(
  rangePreset: RangePresetId,
  endDate: string,
  endHour: number,
): string {
  if (rangePreset === '1h') {
    const h = Math.min(23, Math.max(0, endHour));
    return `${endDate} ${String(h).padStart(2, '0')}:59:59`;
  }
  return `${endDate} 23:59:59`;
}

async function apiJson<T>(
  url: string,
  label: string,
  method: 'GET' | 'POST' = 'GET',
  body?: unknown,
): Promise<T> {
  let response: Response;
  try {
    response = await fetch(url, {
      method,
      headers: body !== undefined ? { 'Content-Type': 'application/json' } : undefined,
      body: body !== undefined ? JSON.stringify(body) : undefined,
    });
  } catch (err) {
    const hint = err instanceof Error ? err.message : String(err);
    throw new Error(
      `${label}: network error (${hint}). Is uvicorn running? Try: uvicorn app.api.main:app --reload --log-level info`,
      { cause: err },
    );
  }
  if (!response.ok) {
    const text = await response.text();
    throw new Error(`${label} HTTP ${response.status}: ${text.slice(0, 400)}`);
  }
  return response.json() as Promise<T>;
}

export async function fetchDatasets(): Promise<DatasetsResponse> {
  return apiJson(`${API_BASE}/datasets`, 'datasets');
}

/** @deprecated Use fetchDatasets */
export async function fetchRegions(): Promise<DatasetsResponse> {
  return fetchDatasets();
}

export async function fetchTraffic(params: {
  dataset: string;
  stations: string[];
  rangePreset: RangePresetId;
  endDate: string;
  endHour: number;
}): Promise<TrafficResponse> {
  const qs = new URLSearchParams({
    dataset: params.dataset,
    stations: params.stations.join(','),
    range_preset: params.rangePreset,
    date_to: buildTrafficEndUtc(params.rangePreset, params.endDate, params.endHour),
  });
  return apiJson(`${API_BASE}/traffic?${qs}`, 'traffic');
}

export async function fetchClassTraffic(params: {
  dataset: string;
  stations: string[];
  rangePreset: RangePresetId;
  endDate: string;
  endHour: number;
}): Promise<ClassTrafficResponse> {
  const qs = new URLSearchParams({
    dataset: params.dataset,
    stations: params.stations.join(','),
    range_preset: params.rangePreset,
    date_to: buildTrafficEndUtc(params.rangePreset, params.endDate, params.endHour),
  });
  return apiJson(`${API_BASE}/traffic/classes?${qs}`, 'traffic/classes');
}

export async function fetchMap(dataset: string): Promise<MapResponse> {
  const qs = new URLSearchParams({ dataset });
  return apiJson(`${API_BASE}/map?${qs}`, 'map');
}

export async function fetchModelsLocal(): Promise<ModelsLocalResponse> {
  return apiJson(`${API_BASE}/models/local`, 'models/local');
}

export async function fetchTimesnetRun(
  runId: string,
  modelId: string = DEFAULT_ANOMALY_MODEL,
): Promise<TimesnetRunManifest> {
  return apiJson(
    `${API_BASE}/models/local/${modelId}/run?${modelRunQuery(runId)}`,
    'models run',
  );
}

export async function fetchTimesnetSeries(
  runId: string,
  mode: 'percentile' | 'combined' = 'combined',
  modelId: string = DEFAULT_ANOMALY_MODEL,
): Promise<TimesnetSeriesListResponse> {
  return apiJson(
    `${API_BASE}/models/local/${modelId}/series?${modelRunQuery(runId, { mode })}`,
    'models series',
  );
}

const CHART_FETCH_MAX_POINTS = 2500;

export async function fetchTimesnetLocationChart(
  runId: string,
  locationId: string,
  options?: {
    maxPoints?: number;
    split?: string;
    window?: string;
    mode?: 'percentile' | 'combined';
    modelId?: string;
  },
): Promise<TimesnetLocationChartResponse> {
  const modelId = options?.modelId ?? DEFAULT_ANOMALY_MODEL;
  const extra: Record<string, string> = {
    location_id: locationId,
    max_points: String(options?.maxPoints ?? CHART_FETCH_MAX_POINTS),
    mode: options?.mode ?? 'combined',
    window: options?.window ?? 'full',
  };
  if (options?.split) extra.split = options.split;
  return apiJson(
    `${API_BASE}/models/local/${modelId}/location-chart?${modelRunQuery(runId, extra)}`,
    'models location-chart',
  );
}

export async function fetchTimesnetScores(
  runId: string,
  uniqueId: string,
  mode: 'percentile' | 'combined' = 'combined',
  modelId: string = DEFAULT_ANOMALY_MODEL,
  window = 'full',
): Promise<TimesnetScoresResponse> {
  return apiJson(
    `${API_BASE}/models/local/${modelId}/scores?${modelRunQuery(runId, {
      unique_id: uniqueId,
      max_points: String(CHART_FETCH_MAX_POINTS),
      mode,
      window,
    })}`,
    'models scores',
  );
}

export async function fetchTimesnetAnomalies(
  runId: string,
  uniqueId?: string,
  limit = 200,
  mode: 'percentile' | 'combined' = 'combined',
  modelId: string = DEFAULT_ANOMALY_MODEL,
): Promise<TimesnetAnomaliesListResponse> {
  const extra: Record<string, string> = { limit: String(limit), mode };
  if (uniqueId) extra.unique_id = uniqueId;
  return apiJson(
    `${API_BASE}/models/local/${modelId}/anomalies?${modelRunQuery(runId, extra)}`,
    'models anomalies',
  );
}

export async function fetchGtEvents(
  modelId: string,
  options?: {
    locationPrefix?: string;
    dateFrom?: string;
    dateTo?: string;
  },
): Promise<GtEventsResponse> {
  const qs = new URLSearchParams();
  if (options?.locationPrefix) qs.set('location_prefix', options.locationPrefix);
  if (options?.dateFrom) qs.set('date_from', options.dateFrom);
  if (options?.dateTo) qs.set('date_to', options.dateTo);
  const q = qs.toString();
  return apiJson(
    `${API_BASE}/models/local/${modelId}/gt-events${q ? `?${q}` : ''}`,
    'gt-events',
  );
}

// ── Forecasting models (future forecasting) ────────────────────────────────

export async function fetchForecastRun(
  modelId: string,
  runId: string,
): Promise<ForecastRunManifest> {
  return apiJson(
    `${API_BASE}/models/local/forecast/${modelId}/run?${modelRunQuery(runId)}`,
    'forecast run',
  );
}

export async function fetchForecastSeries(
  modelId: string,
  runId: string,
  scope?: string,
): Promise<ForecastSeriesResponse> {
  const extra: Record<string, string> = {};
  if (scope) extra.scope = scope;
  return apiJson(
    `${API_BASE}/models/local/forecast/${modelId}/series?${modelRunQuery(runId, extra)}`,
    'forecast series',
  );
}

export async function fetchForecastChart(
  modelId: string,
  runId: string,
  uniqueId: string,
  scope?: string,
  window = 'full',
  maxPoints = CHART_FETCH_MAX_POINTS,
): Promise<ForecastChartResponse> {
  const extra: Record<string, string> = {
    unique_id: uniqueId,
    max_points: String(maxPoints),
    window,
  };
  if (scope) extra.scope = scope;
  return apiJson(
    `${API_BASE}/models/local/forecast/${modelId}/chart?${modelRunQuery(runId, extra)}`,
    'forecast chart',
  );
}
