import { useEffect, useMemo, useState } from 'react';
import ReactECharts from 'echarts-for-react';
import { Pause, Play, RotateCcw, X } from 'lucide-react';
import { Devices } from './Devices';
import { buildLiveChartOption, CHANNEL_DISPLAY, LEAD_LABEL, windowErrors } from './chart';
import type { LiveState } from './useLiveSystem';
import { CHANNELS, LEADS, type Channel, type LeadName, type ReplayState } from './types';
import { formatDate, formatTime, speedLabel, toLocalInput } from '../format';

type Send = (target: 'sensor' | 'predictor', command: Record<string, unknown>, note?: string) => void;

const SPEEDS = [1, 60, 300, 900, 1800, 3600];

/** The sensor's clock, run locally between its state messages. */
function useSimNow(replay: ReplayState | null): { sim: number | null; wall: number } {
  const [wall, setWall] = useState(() => Date.now());
  useEffect(() => {
    const id = window.setInterval(() => setWall(Date.now()), 200);
    return () => window.clearInterval(id);
  }, []);
  if (!replay) return { sim: null, wall };
  const base = Date.parse(replay.sim_time);
  if (replay.phase !== 'running') return { sim: base, wall };
  const sim = base + Math.max(0, wall - replay.wall_time_ms) * replay.speed;
  return { sim: Math.min(sim, Date.parse(replay.end)), wall };
}

function ResetPanel({ replay, send, onClose }: { replay: ReplayState; send: Send; onClose: () => void }) {
  const [start, setStart] = useState(() => toLocalInput(Date.parse(replay.start)));
  const [speed, setSpeed] = useState(replay.speed);
  const [paused, setPaused] = useState(false);
  const min = toLocalInput(Date.parse(replay.data_from));
  const max = toLocalInput(Date.parse(replay.data_to));
  return (
    <div className="reset-panel">
      <div className="reset-head">
        <b>New simulation</b>
        <button className="icon-btn" onClick={onClose} aria-label="Close">
          <X size={14} />
        </button>
      </div>
      <p className="muted">
        The sensor sends the 8 days before the start as a fast burst (the model needs 2 days of input, the
        baseline a week), then streams from the start time. The predictor clears its state.
      </p>
      <label>
        Start (station time, PT)
        <input type="datetime-local" value={start} min={min} max={max} step={300}
          onChange={(e) => setStart(e.target.value)} />
      </label>
      <label>
        Speed
        <select value={speed} onChange={(e) => setSpeed(Number(e.target.value))}>
          {SPEEDS.map((s) => (
            <option key={s} value={s}>
              ×{s} · {speedLabel(s)}
            </option>
          ))}
        </select>
      </label>
      <label className="check">
        <input type="checkbox" checked={paused} onChange={(e) => setPaused(e.target.checked)} />
        Start paused
      </label>
      <button
        className="btn-primary"
        onClick={() => {
          send('sensor', { cmd: 'reset', start, speed, paused }, `→ sensor: new session from ${start.replace('T', ' ')}`);
          onClose();
        }}
      >
        <RotateCcw size={14} /> Start session
      </button>
    </div>
  );
}

function ClockBar({ state, send, sim }: { state: LiveState; send: Send; sim: number | null }) {
  const replay = state.replay;
  const [customSpeed, setCustomSpeed] = useState('');
  const [resetOpen, setResetOpen] = useState(false);
  const running = replay?.phase === 'running';
  const canControl = state.connection === 'online' && state.sensor?.state === 'online' && !!replay;
  const progress = replay && sim
    ? Math.min(1, Math.max(0, (sim - Date.parse(replay.start)) / (Date.parse(replay.end) - Date.parse(replay.start))))
    : 0;

  return (
    <header className="clock-bar">
      <div className="sim-clock">
        <span className="sim-label">Simulated station time</span>
        <div className="sim-date">{sim ? formatDate(sim) : '—'}</div>
        <div className="sim-time">{sim ? formatTime(sim) : '--:--:--'}</div>
        <div className="sim-progress" title="Position within this session">
          <span style={{ width: `${progress * 100}%` }} />
        </div>
      </div>

      <div className="playback">
        <div className="playback-row">
          <span className={`phase phase-${replay?.phase ?? 'idle'}`}>{replay?.phase ?? 'no sensor'}</span>
          <button
            className="btn-primary"
            disabled={!canControl || !['running', 'paused'].includes(replay?.phase ?? '')}
            onClick={() => send('sensor', { cmd: running ? 'pause' : 'resume' })}
          >
            {running ? <Pause size={14} /> : <Play size={14} />} {running ? 'Pause' : 'Resume'}
          </button>
          <button className="btn-ghost" disabled={!canControl} onClick={() => setResetOpen((v) => !v)}>
            <RotateCcw size={14} /> New simulation
          </button>
        </div>
        <div className="playback-row">
          <span className="muted small">Speed</span>
          <div className="segmented">
            {SPEEDS.map((s) => (
              <button
                key={s}
                className={replay?.speed === s ? 'active' : ''}
                disabled={!canControl}
                onClick={() => send('sensor', { cmd: 'speed', value: s })}
                title={speedLabel(s)}
              >
                ×{s}
              </button>
            ))}
          </div>
          <form
            className="custom-speed"
            onSubmit={(e) => {
              e.preventDefault();
              const v = Number(customSpeed);
              if (v >= 1) send('sensor', { cmd: 'speed', value: v });
              setCustomSpeed('');
            }}
          >
            <input
              inputMode="numeric"
              placeholder={replay ? `×${replay.speed}` : '×'}
              value={customSpeed}
              disabled={!canControl}
              onChange={(e) => setCustomSpeed(e.target.value.replace(/[^0-9.]/g, ''))}
            />
          </form>
          <span className="muted small">{replay ? `1 s = ${speedLabel(replay.speed).replace('/s', '')}` : ''}</span>
        </div>
      </div>
      {resetOpen && replay && <ResetPanel replay={replay} send={send} onClose={() => setResetOpen(false)} />}
    </header>
  );
}

function Kpi({ label, value, sub, tone }: { label: string; value: string; sub?: string; tone?: 'good' | 'bad' }) {
  return (
    <div className="kpi">
      <span>{label}</span>
      <b className={tone ? `tone-${tone}` : ''}>{value}</b>
      {sub && <small>{sub}</small>}
    </div>
  );
}

export function LivePage({ state, send }: { state: LiveState; send: Send }) {
  const { sim, wall } = useSimNow(state.replay);
  const [channel, setChannel] = useState<Channel>('flow');
  const [lead, setLead] = useState<LeadName>('1h');
  const [pastHours, setPastHours] = useState(24);
  const [showWeekly, setShowWeekly] = useState(true);
  const [showDeviations, setShowDeviations] = useState(false);

  // Redraw on data, and on the clock only once per simulated 5 minutes.
  const simTick = sim ? Math.floor(sim / 300_000) : 0;
  const option = useMemo(
    () =>
      buildLiveChartOption({
        buckets: state.buckets,
        readings: state.readings,
        forecast: state.forecast,
        channel,
        lead,
        nowMs: simTick ? simTick * 300_000 : null,
        pastHours,
        showWeekly,
        showDeviations,
      }),
    [state.buckets, state.readings, state.forecast, channel, lead, simTick, pastHours, showWeekly, showDeviations],
  );

  const errors = useMemo(() => {
    const last = state.buckets[state.buckets.length - 1];
    if (!last) return null;
    return windowErrors(state.buckets, channel, lead, Date.parse(last.t) - 24 * 3600_000);
  }, [state.buckets, channel, lead]);
  const d = CHANNEL_DISPLAY[channel];
  const warming = state.replay?.phase === 'warmup' || (state.buckets.length > 0 && !state.forecast);

  return (
    <div className="live-page">
      <ClockBar state={state} send={send} sim={sim} />
      <Devices state={state} send={send} now={wall} />

      <section className="panel chart-panel">
        <div className="panel-toolbar">
          <div className="segmented">
            {CHANNELS.map((c) => (
              <button key={c} className={c === channel ? 'active' : ''} onClick={() => setChannel(c)}>
                {CHANNEL_DISPLAY[c].label}
              </button>
            ))}
          </div>
          <label className="inline">
            Compare with forecast made
            <select value={lead} onChange={(e) => setLead(e.target.value as LeadName)}>
              {LEADS.map((l) => (
                <option key={l} value={l}>
                  {LEAD_LABEL[l]}
                </option>
              ))}
            </select>
          </label>
          <div className="segmented">
            {[24, 48].map((h) => (
              <button key={h} className={h === pastHours ? 'active' : ''} onClick={() => setPastHours(h)}>
                {h} h back
              </button>
            ))}
          </div>
          <label className="check">
            <input type="checkbox" checked={showWeekly} onChange={(e) => setShowWeekly(e.target.checked)} />
            Last week
          </label>
          <label className="check" title="15-min-ahead miss above 6× the usual error. Unverified: PeMS has no incident ground truth here.">
            <input type="checkbox" checked={showDeviations} onChange={(e) => setShowDeviations(e.target.checked)} />
            Large deviations
          </label>
        </div>
        <div className="chart-wrap">
          {state.buckets.length === 0 && state.readings.length === 0 && (
            <div className="chart-empty">
              {state.connection !== 'online'
                ? 'Connecting to the MQTT broker…'
                : warming
                  ? 'Predictor is warming up…'
                  : 'Waiting for the first 15-minute bucket…'}
            </div>
          )}
          <ReactECharts option={option} notMerge style={{ height: '100%', width: '100%' }} />
        </div>
      </section>

      <section className="bottom-row">
        <div className="panel kpis">
          <h4>
            {d.label}, forecast {LEAD_LABEL[lead]} · last 24 h
          </h4>
          <div className="kpi-grid">
            <Kpi label="Model MAE" value={errors ? `${errors.model.toFixed(d.digits)} ${d.unit}` : '—'} />
            <Kpi
              label="Last-week MAE"
              value={errors ? `${errors.weekly.toFixed(d.digits)} ${d.unit}` : '—'}
              sub="naive: same time 7 days ago"
            />
            <Kpi
              label="Skill vs last week"
              value={errors?.skill != null ? `${(errors.skill * 100).toFixed(0)} %` : '—'}
              tone={errors?.skill != null ? (errors.skill > 0 ? 'good' : 'bad') : undefined}
              sub="share of error removed"
            />
            <Kpi label="Buckets scored" value={errors ? String(errors.n) : '0'} sub="observed 15-min steps" />
          </div>
        </div>
        <div className="panel event-log">
          <h4>Events</h4>
          <ul>
            {state.log.map((e) => (
              <li key={e.id} className={`log-${e.level}`}>
                <time>{new Date(e.at).toLocaleTimeString('en-GB')}</time>
                {e.text}
              </li>
            ))}
            {state.log.length === 0 && <li className="muted">Nothing yet.</li>}
          </ul>
        </div>
      </section>
    </div>
  );
}
