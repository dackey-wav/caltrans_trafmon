/** Payloads of the MQTT topics (see app/iot/topics.py). */

export type Channel = 'flow' | 'speed' | 'occupancy';
export const CHANNELS: Channel[] = ['flow', 'speed', 'occupancy'];
export type LeadName = '15min' | '1h' | '4h' | '24h';
export const LEADS: LeadName[] = ['15min', '1h', '4h', '24h'];

export type NodeStatus = {
  state: 'online' | 'offline';
  station_id?: string;
  model?: string;
  since?: string;
  reason?: string;
  until_ms?: number;
};

export type ReplayPhase = 'idle' | 'warmup' | 'running' | 'paused' | 'finished';

export type ReplayState = {
  station_id: string;
  station: {
    latitude: number | null;
    longitude: number | null;
    freeway: string;
    direction: string;
    lane_type: string;
  };
  session: string;
  phase: ReplayPhase;
  speed: number;
  sim_time: string;
  wall_time_ms: number;
  start: string;
  end: string;
  data_from: string;
  data_to: string;
  progress: number;
  sent: number;
  dropped: number;
  drop_rate: number;
  backlog: number;
  outage: { mode: 'buffer' | 'drop'; seconds: number; until_ms: number; lost: number } | null;
};

export type Measurement = {
  station_id: string;
  ts_utc: string;
  session?: string;
  seq: number;
  sent_at_ms?: number;
  flow_veh_5min: number | null;
  speed_mph: number | null;
  occupancy_fraction: number | null;
  observed_percent: number | null;
};

/** One evaluated 15-minute bucket: [flow, speed, occupancy] vectors. */
export type Bucket = {
  t: string;
  a: number[] | null;
  w: number[] | null;
  p: Partial<Record<LeadName, number[] | null>>;
  x: boolean;
};

export type HistoryMessage = {
  session: string | null;
  buckets: Bucket[];
};

export type EvaluationMessage = {
  session: string | null;
  bucket_start: string;
  active: boolean;
  actual: Record<Channel, number> | null;
  weekly_baseline: Record<Channel, number> | null;
  predictions: Partial<Record<LeadName, Record<Channel, number>>>;
  anomaly: { flag: boolean } | null;
};

export type ForecastMessage = {
  session: string | null;
  model: string;
  issued_at: string;
  data_until: string;
  inference_ms: number;
  horizon: Array<{ ts: string } & Record<Channel, number>>;
};

type Stats = { n: number; mean: number; p50: number; p95: number; max: number } | null;

export type MetricsMessage = {
  session: string | null;
  model: string;
  last_bucket: string | null;
  messages_received: number;
  duplicates_ignored: number;
  invalid: number;
  processing_errors: number;
  forecasts_made: number;
  deviations: number;
  mqtt_latency_ms: Stats;
  inference_ms: Stats;
};

export type LogEntry = { id: number; at: number; level: 'info' | 'warn' | 'ok'; text: string };
