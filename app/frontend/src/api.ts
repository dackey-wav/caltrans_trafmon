/** REST: facts that do not change while streaming. Live data comes over MQTT. */

export type AppConfig = {
  station_id: string;
  mqtt_ws_url: string | null;
  topic_root: string;
  model: string;
};

export type StationInfo = {
  station_id: string;
  file: string;
  latitude: number | null;
  longitude: number | null;
  freeway: string;
  direction: string;
  lane_type: string;
  rows: number;
  first: string;
  last: string;
  observed_share: number;
  monthly: { month: string; rows: number; observed_share: number }[];
};

type PerChannel = Record<'flow' | 'speed' | 'occupancy', number>;

export type ModelInfo = {
  name: string;
  run_id: string;
  hparams: Record<string, string | number | null>;
  parameters: number;
  best_epoch: number;
  epochs_run: number;
  train_seconds: number;
  device: string;
  split_ds: Record<'train' | 'val' | 'test', [string, string]>;
  marks: string[];
  test: {
    methods: string[];
    n_origins: number;
    overall: Record<string, PerChannel>;
    by_lead_bucket: Record<string, Record<string, PerChannel>>;
    lead_minutes: number[];
    by_lead: Record<string, Record<'flow' | 'speed' | 'occupancy', number[]>>;
    skill_vs_weekly: PerChannel;
  };
};

async function get<T>(path: string): Promise<T> {
  const res = await fetch(path);
  if (!res.ok) {
    const body = await res.json().catch(() => ({}));
    throw new Error(body.detail ?? `${path}: HTTP ${res.status}`);
  }
  return res.json() as Promise<T>;
}

export const fetchConfig = () => get<AppConfig>('/api/config');
export const fetchStation = () => get<StationInfo>('/api/station');
export const fetchModel = () => get<ModelInfo>('/api/model');
