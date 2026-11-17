import type { Bucket, Channel, ForecastMessage, LeadName } from './types';
import type { Reading } from './useLiveSystem';
import { CHANNELS } from './types';
import { STATION_TZ } from '../format';

export const COLORS = {
  actual: '#5794f2',
  raw: 'rgba(138, 184, 255, 0.45)',
  imputed: 'rgba(199, 208, 217, 0.35)',
  predicted: '#ff9830',
  weekly: '#8a8f98',
  deviation: '#f2495c',
  now: '#73bf69',
};

/** Display units: flow per hour reads better than per 15-minute bucket. */
export const CHANNEL_DISPLAY: Record<Channel, { label: string; unit: string; factor: number; digits: number }> = {
  flow: { label: 'Flow', unit: 'veh/h', factor: 4, digits: 0 },
  speed: { label: 'Speed', unit: 'mph', factor: 1, digits: 1 },
  occupancy: { label: 'Occupancy', unit: '%', factor: 100, digits: 1 },
};

export const LEAD_LABEL: Record<LeadName, string> = {
  '15min': '15 min ahead',
  '1h': '1 h ahead',
  '4h': '4 h ahead',
  '24h': '24 h ahead',
};

const axisTime = new Intl.DateTimeFormat('en-GB', {
  timeZone: STATION_TZ,
  hour: '2-digit',
  minute: '2-digit',
  hour12: false,
});
const axisDay = new Intl.DateTimeFormat('en-GB', { timeZone: STATION_TZ, weekday: 'short', day: '2-digit', month: 'short' });
const tipTime = new Intl.DateTimeFormat('en-GB', {
  timeZone: STATION_TZ,
  weekday: 'short',
  day: '2-digit',
  month: 'short',
  hour: '2-digit',
  minute: '2-digit',
  hour12: false,
});

type Point = [number, number | null];

export function scaled(values: number[] | null | undefined, channel: Channel): number | null {
  if (!values) return null;
  const v = values[CHANNELS.indexOf(channel)];
  return v == null || !Number.isFinite(v) ? null : v * CHANNEL_DISPLAY[channel].factor;
}

export type LiveChartInput = {
  buckets: Bucket[];
  readings: Reading[];
  forecast: ForecastMessage | null;
  channel: Channel;
  lead: LeadName;
  nowMs: number | null;
  pastHours: number;
  showWeekly: boolean;
  showDeviations: boolean;
};

export function buildLiveChartOption(input: LiveChartInput): Record<string, unknown> {
  const { buckets, readings, forecast, channel, lead, nowMs, pastHours, showWeekly, showDeviations } = input;
  const d = CHANNEL_DISPLAY[channel];
  // 5-minute flow is per 5 min: x12 for veh/h (the 15-minute factor is x4).
  const rawFactor = channel === 'flow' ? 12 : d.factor;
  const ci = CHANNELS.indexOf(channel) + 1;
  const rawValue = (r: Reading, observed: boolean): Point => [
    r[0],
    r[ci] == null || (r[4] === 1) !== observed ? null : (r[ci] as number) * rawFactor,
  ];
  const raw: Point[] = readings.map((r) => rawValue(r, true));
  const imputed: Point[] = readings.map((r) => rawValue(r, false));
  const anyImputed = imputed.some((p) => p[1] != null);
  const actual: Point[] = buckets.map((b) => [Date.parse(b.t), scaled(b.a, channel)]);
  const predicted: Point[] = buckets.map((b) => [Date.parse(b.t), scaled(b.p[lead], channel)]);
  const weekly: Point[] = buckets.map((b) => [Date.parse(b.t), scaled(b.w, channel)]);
  const future: Point[] = (forecast?.horizon ?? []).map((h) => [
    Date.parse(h.ts),
    h[channel] * d.factor,
  ]);
  const deviations = buckets
    .filter((b) => b.x && b.a)
    .map((b) => [Date.parse(b.t), scaled(b.a, channel)]);

  const now = nowMs ?? (actual.length ? actual[actual.length - 1][0] : Date.now());
  const min = now - pastHours * 3600_000;
  const max = now + 24 * 3600_000;

  const line = (name: string, data: Point[], color: string, extra: Record<string, unknown> = {}) => ({
    name,
    type: 'line',
    data,
    showSymbol: false,
    smooth: 0.15,
    connectNulls: false,
    lineStyle: { width: 1.8, color },
    itemStyle: { color },
    ...extra,
  });

  return {
    backgroundColor: 'transparent',
    animation: false,
    color: [COLORS.raw, COLORS.imputed, COLORS.actual, COLORS.predicted, COLORS.predicted, COLORS.weekly, COLORS.deviation],
    tooltip: {
      trigger: 'axis',
      backgroundColor: '#111217',
      borderColor: '#2c3235',
      textStyle: { color: '#d8d9da', fontSize: 12 },
      axisPointer: { type: 'line', lineStyle: { color: '#3d4148' } },
      formatter: (params: Array<{ value: Point; seriesName: string; color: string }>) => {
        if (!params.length) return '';
        const head = tipTime.format(params[0].value[0]);
        const rows = params
          .filter((p) => p.value[1] != null)
          .map(
            (p) =>
              `<div style="display:flex;justify-content:space-between;gap:16px"><span><span style="color:${p.color}">●</span> ${p.seriesName}</span><b>${Number(p.value[1]).toFixed(d.digits)}</b></div>`,
          )
          .join('');
        return `<div style="margin-bottom:4px;color:#9aa0a6">${head} PT</div>${rows}`;
      },
    },
    legend: {
      bottom: 0,
      textStyle: { color: '#9aa0a6', fontSize: 11 },
      itemWidth: 18,
      data: [
        'Sensor 5-min readings',
        ...(anyImputed ? ['Imputed by PeMS (not scored)'] : []),
        'Actual (15 min, scored)',
        `Predicted ${LEAD_LABEL[lead]}`,
        'Current 24 h forecast',
        ...(showWeekly ? ['Same time last week'] : []),
        ...(showDeviations ? ['Large deviation'] : []),
      ],
    },
    grid: { left: 12, right: 20, top: 28, bottom: 44, containLabel: true },
    xAxis: {
      type: 'time',
      min,
      max,
      axisLine: { lineStyle: { color: '#2c3235' } },
      splitLine: { show: false },
      axisLabel: {
        color: '#9aa0a6',
        fontSize: 10,
        hideOverlap: true,
        formatter: (value: number) => {
          const t = axisTime.format(value);
          return t === '00:00' ? `{day|${axisDay.format(value)}}` : t;
        },
        rich: { day: { color: '#e6e8ea', fontWeight: 600, fontSize: 10 } },
      },
    },
    yAxis: {
      type: 'value',
      name: d.unit,
      nameTextStyle: { color: '#6e7681', fontSize: 10, align: 'left' },
      scale: channel !== 'flow',
      splitLine: { lineStyle: { color: '#1f2328' } },
      axisLabel: { color: '#9aa0a6', fontSize: 10 },
    },
    series: [
      line('Sensor 5-min readings', raw, COLORS.raw, {
        lineStyle: { width: 1, color: COLORS.raw },
        smooth: 0,
        z: 1,
      }),
      ...(anyImputed
        ? [
            line('Imputed by PeMS (not scored)', imputed, COLORS.imputed, {
              lineStyle: { width: 1, color: COLORS.imputed, type: 'dashed' },
              smooth: 0,
              z: 1,
            }),
          ]
        : []),
      line('Actual (15 min, scored)', actual, COLORS.actual, {
        lineStyle: { width: 2.2, color: COLORS.actual },
        markLine: {
          silent: true,
          symbol: 'none',
          label: { formatter: 'now', color: COLORS.now, fontSize: 10, position: 'insideEndTop' },
          lineStyle: { color: COLORS.now, type: 'solid', width: 1 },
          data: [{ xAxis: now }],
        },
        markArea: {
          silent: true,
          itemStyle: { color: 'rgba(255, 152, 48, 0.04)' },
          data: [[{ xAxis: now }, { xAxis: max }]],
        },
      }),
      line(`Predicted ${LEAD_LABEL[lead]}`, predicted, COLORS.predicted, {
        lineStyle: { width: 1.6, color: COLORS.predicted },
      }),
      line('Current 24 h forecast', future, COLORS.predicted, {
        lineStyle: { width: 1.6, color: COLORS.predicted, type: 'dashed' },
      }),
      ...(showWeekly
        ? [
            line('Same time last week', weekly, COLORS.weekly, {
              lineStyle: { width: 1.2, color: COLORS.weekly, type: 'dotted' },
            }),
          ]
        : []),
      ...(showDeviations
        ? [
            {
              name: 'Large deviation',
              type: 'scatter',
              data: deviations,
              symbolSize: 9,
              itemStyle: { color: COLORS.deviation },
            },
          ]
        : []),
    ],
  };
}

/** MAE of the model and of last week's values over the buckets that have all three. */
export function windowErrors(buckets: Bucket[], channel: Channel, lead: LeadName, sinceMs: number) {
  let n = 0;
  let model = 0;
  let weekly = 0;
  for (const b of buckets) {
    if (Date.parse(b.t) < sinceMs) continue;
    const a = scaled(b.a, channel);
    const p = scaled(b.p[lead], channel);
    const w = scaled(b.w, channel);
    if (a == null || p == null || w == null) continue;
    n += 1;
    model += Math.abs(p - a);
    weekly += Math.abs(w - a);
  }
  if (!n) return null;
  return { n, model: model / n, weekly: weekly / n, skill: weekly > 0 ? 1 - model / weekly : null };
}
