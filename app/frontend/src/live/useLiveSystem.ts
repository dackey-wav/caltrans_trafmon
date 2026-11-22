import { useCallback, useEffect, useReducer, useRef } from 'react';
import mqtt, { type MqttClient } from 'mqtt';
import type {
  Bucket,
  EvaluationMessage,
  ForecastMessage,
  HistoryMessage,
  LeadName,
  LogEntry,
  Measurement,
  MetricsMessage,
  NodeStatus,
  ReplayState,
} from './types';
import { CHANNELS } from './types';
import { formatSim } from '../format';

const MAX_BUCKETS = 192;
const MAX_LOG = 80;
const MAX_READINGS = 600;

/** [ms, flow, speed, occupancy, observed]: observed = 0 when PeMS imputed the row. */
export type Reading = [number, number | null, number | null, number | null, number];

export type LiveState = {
  connection: 'connecting' | 'online' | 'offline';
  sensor: NodeStatus | null;
  predictor: NodeStatus | null;
  replay: ReplayState | null;
  measurement: Measurement | null;
  measurementLatencyMs: number | null;
  receivedAt: number | null;
  buckets: Bucket[];
  /** Raw 5-minute readings as the dashboard itself receives them: [ms, flow, speed, occupancy]. */
  readings: Reading[];
  forecast: ForecastMessage | null;
  metrics: MetricsMessage | null;
  log: LogEntry[];
  rate: number;
  /** Session the buckets/forecast belong to; retained messages can arrive before the replay state. */
  dataSession: string | null;
};

type Action =
  | { type: 'connection'; value: LiveState['connection'] }
  | { type: 'message'; topic: string; payload: unknown; at: number }
  | { type: 'log'; level: LogEntry['level']; text: string }
  | { type: 'rate'; value: number };

const initial: LiveState = {
  connection: 'connecting',
  sensor: null,
  predictor: null,
  replay: null,
  measurement: null,
  measurementLatencyMs: null,
  receivedAt: null,
  buckets: [],
  readings: [],
  forecast: null,
  metrics: null,
  log: [],
  rate: 0,
  dataSession: null,
};

let logId = 0;

function withLog(state: LiveState, level: LogEntry['level'], text: string): LiveState {
  const entry = { id: ++logId, at: Date.now(), level, text };
  return { ...state, log: [entry, ...state.log].slice(0, MAX_LOG) };
}

function vec(values: Record<string, number> | null | undefined): number[] | null {
  return values ? CHANNELS.map((c) => values[c]) : null;
}

function mergeBucket(buckets: Bucket[], bucket: Bucket): Bucket[] {
  const last = buckets[buckets.length - 1];
  if (last && last.t === bucket.t) return [...buckets.slice(0, -1), bucket];
  if (last && Date.parse(bucket.t) < Date.parse(last.t)) {
    const others = buckets.filter((b) => b.t !== bucket.t);
    return [...others, bucket].sort((a, b) => Date.parse(a.t) - Date.parse(b.t)).slice(-MAX_BUCKETS);
  }
  return [...buckets, bucket].slice(-MAX_BUCKETS);
}

function statusText(name: string, status: NodeStatus): string {
  if (status.state === 'online') return `${name} online`;
  return status.reason ? `${name} offline: ${status.reason}` : `${name} offline (Last Will)`;
}

function reduceMessage(state: LiveState, topic: string, payload: unknown, at: number): LiveState {
  const leaf = topic.split('/').pop();
  const session = state.replay?.session ?? null;
  if (topic.endsWith('/predictor/status')) {
    const next = payload as NodeStatus;
    const changed = state.predictor?.state !== next.state;
    const s = { ...state, predictor: next };
    return changed ? withLog(s, next.state === 'online' ? 'ok' : 'warn', statusText('Predictor', next)) : s;
  }
  switch (leaf) {
    case 'status': {
      const next = payload as NodeStatus;
      const changed = state.sensor?.state !== next.state;
      const s = { ...state, sensor: next };
      return changed ? withLog(s, next.state === 'online' ? 'ok' : 'warn', statusText('Sensor', next)) : s;
    }
    case 'replay': {
      const next = payload as ReplayState;
      let s: LiveState = { ...state, replay: next };
      if (state.dataSession && state.dataSession !== next.session) {
        s = { ...s, buckets: [], readings: [], forecast: null, metrics: null, dataSession: next.session };
      }
      if (state.replay && next.session !== state.replay.session) {
        s = { ...s, buckets: [], readings: [], forecast: null, metrics: null, measurement: null };
        s = withLog(s, 'info', `New session from ${formatSim(next.start)} at ×${next.speed}`);
      } else if (state.replay) {
        if (state.replay.phase !== next.phase) s = withLog(s, 'info', `Replay ${next.phase}`);
        if (state.replay.speed !== next.speed) s = withLog(s, 'info', `Speed ×${next.speed}`);
        if (state.replay.drop_rate !== next.drop_rate)
          s = withLog(s, 'info', `Message loss ${(next.drop_rate * 100).toFixed(0)} %`);
        if (!state.replay.outage && next.outage)
          s = withLog(s, 'warn', `Sensor leaves the network for ${next.outage.seconds} s (${next.outage.mode})`);
        if (state.replay.backlog === 0 && next.backlog > 0)
          s = withLog(s, 'info', `Sensor buffers ${next.backlog} measurements`);
        if (state.replay.backlog > 0 && next.backlog === 0) s = withLog(s, 'ok', 'Sensor backlog delivered');
      }
      return s;
    }
    case '5min': {
      const m = payload as Measurement;
      if (session && m.session && m.session !== session) return state;
      const t = Date.parse(m.ts_utc);
      const last = state.readings[state.readings.length - 1];
      const observed = (m.observed_percent ?? 0) >= 100 ? 1 : 0;
      const reading: Reading = [t, m.flow_veh_5min, m.speed_mph, m.occupancy_fraction, observed];
      const readings = last && t <= last[0] ? state.readings : [...state.readings, reading].slice(-MAX_READINGS);
      return {
        ...state,
        readings,
        measurement: m,
        receivedAt: at,
        measurementLatencyMs: m.sent_at_ms ? Math.max(0, at - m.sent_at_ms) : null,
      };
    }
    case 'history': {
      const h = payload as HistoryMessage;
      if (session && h.session && h.session !== session) return state;
      // The retained history is authoritative; evaluations only extend it.
      const fresh = h.buckets.length || h.session !== state.dataSession ? h.buckets : state.buckets;
      return { ...state, buckets: fresh.slice(-MAX_BUCKETS), dataSession: h.session };
    }
    case 'evaluation': {
      const e = payload as EvaluationMessage;
      if (session && e.session && e.session !== session) return state;
      const p: Bucket['p'] = {};
      for (const [k, v] of Object.entries(e.predictions)) p[k as LeadName] = vec(v);
      const bucket: Bucket = {
        t: e.bucket_start,
        a: e.active ? vec(e.actual) : null,
        w: vec(e.weekly_baseline),
        p,
        x: !!e.anomaly?.flag,
      };
      return { ...state, buckets: mergeBucket(state.buckets, bucket) };
    }
    case 'forecast': {
      const f = payload as ForecastMessage;
      if (session && f.session && f.session !== session) return state;
      return { ...state, forecast: f, dataSession: f.session ?? state.dataSession };
    }
    case 'metrics': {
      const m = payload as MetricsMessage;
      if (session && m.session && m.session !== session) return state;
      return { ...state, metrics: m };
    }
    default:
      return state;
  }
}

function reducer(state: LiveState, action: Action): LiveState {
  switch (action.type) {
    case 'connection':
      if (state.connection === action.value) return state;
      return withLog(
        { ...state, connection: action.value },
        action.value === 'online' ? 'ok' : 'warn',
        action.value === 'online' ? 'Dashboard connected to the broker' : 'Dashboard lost the broker',
      );
    case 'message':
      return reduceMessage(state, action.topic, action.payload, action.at);
    case 'log':
      return withLog(state, action.level, action.text);
    case 'rate':
      return state.rate === action.value ? state : { ...state, rate: action.value };
  }
}

export function defaultMqttUrl(): string {
  const proto = window.location.protocol === 'https:' ? 'wss' : 'ws';
  return `${proto}://${window.location.hostname}:9001`;
}

/** The dashboard is the third MQTT client: it watches both nodes and sends them commands. */
export function useLiveSystem(url: string | null, stationId: string | null) {
  const [state, dispatch] = useReducer(reducer, initial);
  const clientRef = useRef<MqttClient | null>(null);
  const countRef = useRef(0);

  useEffect(() => {
    if (!url || !stationId) return;
    const root = `traffic/pems/stations/${stationId}`;
    const client = mqtt.connect(url, {
      clientId: `dashboard-${Math.random().toString(16).slice(2, 10)}`,
      clean: true,
      reconnectPeriod: 2000,
      connectTimeout: 5000,
    });
    clientRef.current = client;
    client.on('connect', () => {
      dispatch({ type: 'connection', value: 'online' });
      client.subscribe(
        ['status', 'replay', '5min', 'history', 'evaluation', 'forecast', 'metrics']
          .map((t) => `${root}/${t}`)
          .concat('traffic/pems/predictor/status'),
        { qos: 1 },
      );
    });
    client.on('offline', () => dispatch({ type: 'connection', value: 'offline' }));
    client.on('close', () => dispatch({ type: 'connection', value: 'offline' }));
    client.on('message', (topic, raw) => {
      if (topic.endsWith('/5min')) countRef.current += 1;
      try {
        dispatch({ type: 'message', topic, payload: JSON.parse(raw.toString()), at: Date.now() });
      } catch {
        /* not JSON: ignore */
      }
    });
    const timer = window.setInterval(() => {
      dispatch({ type: 'rate', value: countRef.current });
      countRef.current = 0;
    }, 1000);
    return () => {
      window.clearInterval(timer);
      client.end(true);
      clientRef.current = null;
    };
  }, [url, stationId]);

  const send = useCallback(
    (target: 'sensor' | 'predictor', command: Record<string, unknown>, note?: string) => {
      const client = clientRef.current;
      if (!client || !stationId) return;
      const topic =
        target === 'sensor' ? `traffic/pems/stations/${stationId}/control` : 'traffic/pems/predictor/control';
      client.publish(topic, JSON.stringify(command), { qos: 1 });
      dispatch({ type: 'log', level: 'info', text: note ?? `→ ${target}: ${JSON.stringify(command)}` });
    },
    [stationId],
  );

  return { state, send };
}
