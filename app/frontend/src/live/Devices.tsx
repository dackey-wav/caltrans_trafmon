import type { ReactNode } from 'react';
import { Brain, Radio, RadioTower, Unplug, WifiOff } from 'lucide-react';
import type { LiveState } from './useLiveSystem';
import { ago, formatSim } from '../format';

type Send = (target: 'sensor' | 'predictor', command: Record<string, unknown>, note?: string) => void;

const OUTAGE_SECONDS = 30;

function StatusDot({ state }: { state: 'online' | 'offline' | 'unknown' | 'warn' }) {
  return <span className={`status-dot status-${state}`} />;
}

function Countdown({ untilMs, now }: { untilMs?: number; now: number }) {
  if (!untilMs || untilMs < now) return null;
  return <span className="countdown">back in {Math.ceil((untilMs - now) / 1000)} s</span>;
}

function Row({ label, value }: { label: string; value: ReactNode }) {
  return (
    <div className="kv">
      <span>{label}</span>
      <b>{value}</b>
    </div>
  );
}

function Pipe({ active, label }: { active: boolean; label: string }) {
  return (
    <div className={`pipe ${active ? 'pipe-active' : ''}`} aria-hidden>
      <div className="pipe-line">
        <span className="pipe-pulse" />
      </div>
      <span className="pipe-label">{label}</span>
    </div>
  );
}

export function Devices({ state, send, now }: { state: LiveState; send: Send; now: number }) {
  const { sensor, predictor, replay, metrics, measurement } = state;
  const sensorOnline = sensor?.state === 'online';
  const predictorOnline = predictor?.state === 'online';
  const brokerOnline = state.connection === 'online';
  const flowing = brokerOnline && state.rate > 0;

  const simMs = replay ? Date.parse(replay.sim_time) : null;
  const lastBucketEnd = metrics?.last_bucket ? Date.parse(metrics.last_bucket) + 15 * 60_000 : null;
  const lagMin = simMs && lastBucketEnd ? Math.max(0, (simMs - lastBucketEnd) / 60_000) : null;

  const outage = (target: 'sensor' | 'predictor', mode?: 'buffer' | 'drop') =>
    send(
      target,
      { cmd: 'outage', seconds: OUTAGE_SECONDS, ...(mode ? { mode } : {}) },
      `→ ${target}: leave the network for ${OUTAGE_SECONDS} s${mode ? ` (${mode})` : ''}`,
    );

  return (
    <section className="devices">
      <article className={`device ${sensorOnline ? '' : 'device-down'}`}>
        <header>
          <Radio size={16} />
          <div>
            <h3>Device 1 · Sensor node</h3>
            <p>Replays PeMS station {replay?.station_id ?? '—'} every 5 min</p>
          </div>
          <StatusDot state={sensor ? (sensorOnline ? 'online' : 'offline') : 'unknown'} />
        </header>
        <div className="device-body">
          <Row
            label="Status"
            value={
              <>
                {sensor?.state ?? 'unknown'} <Countdown untilMs={sensor?.until_ms} now={now} />
              </>
            }
          />
          <Row label="Last reading" value={measurement ? formatSim(measurement.ts_utc) : '—'} />
          <Row
            label="Flow · speed"
            value={
              measurement
                ? `${measurement.flow_veh_5min ?? '—'} veh/5min · ${measurement.speed_mph ?? '—'} mph`
                : '—'
            }
          />
          <Row label="Sent · lost" value={replay ? `${replay.sent} · ${replay.dropped}` : '—'} />
          <Row label="Buffered" value={replay?.backlog ?? 0} />
        </div>
        <footer>
          <button className="btn-ghost" disabled={!sensorOnline} onClick={() => outage('sensor', 'buffer')}
            title="Sensor keeps measuring, stores readings and sends them after reconnecting">
            <WifiOff size={13} /> Outage · buffer
          </button>
          <button className="btn-ghost" disabled={!sensorOnline} onClick={() => outage('sensor', 'drop')}
            title="Readings taken while offline are lost: the predictor sees gaps">
            <Unplug size={13} /> Outage · lose data
          </button>
          <label className="loss" title="Random loss of single messages">
            Loss
            <select
              value={replay?.drop_rate ?? 0}
              disabled={!sensorOnline}
              onChange={(e) => send('sensor', { cmd: 'loss', value: Number(e.target.value) })}
            >
              {[0, 0.05, 0.1, 0.2].map((v) => (
                <option key={v} value={v}>
                  {v * 100} %
                </option>
              ))}
            </select>
          </label>
        </footer>
      </article>

      <Pipe active={flowing && sensorOnline} label="5min · QoS 1" />

      <article className={`device device-broker ${brokerOnline ? '' : 'device-down'}`}>
        <header>
          <RadioTower size={16} />
          <div>
            <h3>MQTT broker</h3>
            <p>Mosquitto · persistent sessions</p>
          </div>
          <StatusDot state={brokerOnline ? 'online' : state.connection === 'connecting' ? 'unknown' : 'offline'} />
        </header>
        <div className="device-body">
          <Row label="Dashboard link" value={state.connection} />
          <Row label="Messages / s" value={state.rate} />
          <Row
            label="Sensor → dashboard"
            value={state.measurementLatencyMs != null ? `${state.measurementLatencyMs} ms` : '—'}
          />
          <Row label="Last message" value={ago(state.receivedAt, now)} />
        </div>
        <footer className="hint">
          Stop it with <code>docker compose stop mqtt</code> to see both nodes buffer and recover.
        </footer>
      </article>

      <Pipe active={flowing && predictorOnline} label="forecast · retained" />

      <article className={`device ${predictorOnline ? '' : 'device-down'}`}>
        <header>
          <Brain size={16} />
          <div>
            <h3>Device 2 · Predictor node</h3>
            <p>TimesNet · next 24 h every 15 min</p>
          </div>
          <StatusDot state={predictor ? (predictorOnline ? 'online' : 'offline') : 'unknown'} />
        </header>
        <div className="device-body">
          <Row
            label="Status"
            value={
              <>
                {predictor?.state ?? 'unknown'} <Countdown untilMs={predictor?.until_ms} now={now} />
              </>
            }
          />
          <Row label="Processed up to" value={metrics?.last_bucket ? formatSim(lastBucketEnd ?? 0) : '—'} />
          <Row
            label="Behind sensor"
            value={lagMin == null ? '—' : lagMin < 16 ? 'in sync' : `${Math.round(lagMin)} min (sim)`}
          />
          <Row
            label="Forecasts · inference"
            value={
              metrics
                ? `${metrics.forecasts_made} · ${metrics.inference_ms ? `${metrics.inference_ms.p50.toFixed(0)} ms` : '—'}`
                : '—'
            }
          />
          <Row label="Duplicates dropped" value={metrics?.duplicates_ignored ?? '—'} />
        </div>
        <footer>
          <button className="btn-ghost" disabled={!predictorOnline} onClick={() => outage('predictor')}
            title="Predictor disconnects; the broker queues measurements and delivers them on return">
            <WifiOff size={13} /> Outage {OUTAGE_SECONDS} s
          </button>
        </footer>
      </article>
    </section>
  );
}
