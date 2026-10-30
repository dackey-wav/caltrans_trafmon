import type {
  GtEvent,
  TimesnetLocationChartResponse,
  TimesnetScoresResponse,
} from '../types';

export function seriesDisplayName(locationId: string): string {
  const base = seriesLocationId(locationId);
  const parts = base.split('-');
  if (parts.length >= 2 && /^l\d+$/i.test(parts[parts.length - 1])) {
    const lane = parts.pop();
    const classId = seriesClassId(locationId);
    const classSuffix = classId != null ? ` · cls${classId}` : '';
    return `${parts.join('-')} · ${lane}${classSuffix}`;
  }
  return locationId;
}

export function seriesLocationId(uniqueId: string): string {
  const index = uniqueId.indexOf('__cls');
  return index >= 0 ? uniqueId.slice(0, index) : uniqueId;
}

export function seriesClassId(uniqueId: string): number | null {
  const match = uniqueId.match(/__cls(\d+)$/);
  return match ? parseInt(match[1], 10) : null;
}

export type TimesnetChartLayers = {
  gtBands: boolean;
  modelBands: boolean;
  modelPoints: boolean;
  splitBands: boolean;
};

export const DEFAULT_TIMESNET_CHART_LAYERS: TimesnetChartLayers = {
  gtBands: true,
  modelBands: true,
  modelPoints: true,
  splitBands: true,
};

export type ChartWindowPreset = 'full' | 'test' | 'tail7d' | 'tail30d';
export type ChartSplitRange = { split: string; start: number; end: number };
export type ChartZoom = { start: number; end: number };

export const CHART_WINDOW_LABELS: Record<ChartWindowPreset, string> = {
  full: 'Full timeline',
  test: 'Test split',
  tail7d: 'Last 7 days',
  tail30d: 'Last 30 days',
};

export const MODEL_CHART_WINDOW_PRESETS: ChartWindowPreset[] = [
  'full',
  'test',
  'tail7d',
  'tail30d',
];

type ChartPointCounts = {
  points: number;
  points_in_window?: number;
  points_total?: number;
};

export function chartPointSubtitle(
  windowPreset: ChartWindowPreset,
  counts: ChartPointCounts,
): string {
  const label = CHART_WINDOW_LABELS[windowPreset];
  const { points, points_in_window: inWindow, points_total: total } = counts;
  const windowPoints = inWindow ?? points;
  if (points < windowPoints) return `${label} · ${points} of ${windowPoints} points`;
  if (total != null && total > windowPoints) {
    return `${label} · ${windowPoints} points · ${total} in series`;
  }
  return `${label} · ${points} points`;
}

export const GT_BAND_COLOR = 'rgba(87, 148, 242, 0.22)';
export const MODEL_BAND_COLOR = 'rgba(242, 73, 92, 0.20)';

function indexToZoomPct(start: number, end: number, pointCount: number): ChartZoom {
  if (pointCount <= 1) return { start: 0, end: 100 };
  const denominator = pointCount - 1;
  return {
    start: (start / denominator) * 100,
    end: ((end + 1) / denominator) * 100,
  };
}

export function chartTimestampMs(label: string): number {
  const normalized = label.includes('T') ? label : `${label.replace(' ', 'T')}:00Z`;
  return Date.parse(normalized);
}

export function resolveChartZoom(
  preset: ChartWindowPreset,
  timestamps: string[],
  splitRanges?: ChartSplitRange[],
): ChartZoom {
  const pointCount = timestamps.length;
  if (pointCount <= 1 || preset === 'full') return { start: 0, end: 100 };
  if (preset === 'test') {
    const range = splitRanges?.find((item) => item.split === 'test');
    return range
      ? indexToZoomPct(range.start, range.end, pointCount)
      : { start: 0, end: 100 };
  }
  const tailHours = preset === 'tail7d' ? 7 * 24 : preset === 'tail30d' ? 30 * 24 : null;
  if (tailHours == null) return { start: 0, end: 100 };
  const cutoffMs = chartTimestampMs(timestamps[pointCount - 1]) - tailHours * 60 * 60 * 1000;
  let startIndex = 0;
  for (let index = 0; index < pointCount; index += 1) {
    if (chartTimestampMs(timestamps[index]) >= cutoffMs) {
      startIndex = index;
      break;
    }
  }
  return indexToZoomPct(startIndex, pointCount - 1, pointCount);
}

const SPLIT_BAND_COLORS: Record<string, string> = {
  train: 'rgba(110, 118, 129, 0.10)',
  val: 'rgba(184, 119, 217, 0.12)',
  test: 'rgba(87, 148, 242, 0.10)',
};

export function buildSplitMarkAreas(
  splitRanges?: ChartSplitRange[],
): Array<[{ xAxis: number; itemStyle?: { color: string } }, { xAxis: number }]> {
  if (!splitRanges?.length) return [];
  return splitRanges
    .filter((range) => range.split in SPLIT_BAND_COLORS || range.split === 'test')
    .map((range) => [
      {
        xAxis: range.start,
        itemStyle: {
          color: SPLIT_BAND_COLORS[range.split] ?? 'rgba(110, 118, 129, 0.08)',
        },
      },
      { xAxis: range.end },
    ]);
}

export function chartDataZoom(start: number, end: number): Record<string, unknown>[] {
  return [
    { type: 'inside', start, end },
    {
      type: 'slider',
      height: 18,
      bottom: 8,
      start,
      end,
      borderColor: '#2c3235',
      fillerColor: 'rgba(87, 148, 242, 0.15)',
      handleStyle: { color: '#5794f2' },
      textStyle: { color: '#6e7681', fontSize: 9 },
    },
  ];
}

export function buildModelAnomalyMarkAreas(
  timestamps: string[],
  isAnomaly: boolean[],
): Array<[{ xAxis: number }, { xAxis: number }]> {
  if (!timestamps.length) return [];
  const areas: Array<[{ xAxis: number }, { xAxis: number }]> = [];
  let start = -1;
  const flush = (endIndex: number) => {
    if (start < 0) return;
    areas.push([
      { xAxis: Math.max(0, start - 0.5) },
      { xAxis: Math.min(timestamps.length - 1, endIndex + 0.5) },
    ]);
    start = -1;
  };
  isAnomaly.forEach((flag, index) => {
    if (flag && start < 0) start = index;
    else if (!flag && start >= 0) flush(index - 1);
  });
  if (start >= 0) flush(isAnomaly.length - 1);
  return areas;
}

export function buildGtMarkAreas(
  timestamps: string[],
  events: GtEvent[],
): Array<[{ xAxis: number }, { xAxis: number }]> {
  if (!timestamps.length || !events.length) return [];
  const areas: Array<[{ xAxis: number }, { xAxis: number }]> = [];
  for (const event of events) {
    const startMs = Date.parse(event.time_start_utc);
    const endMs = Date.parse(event.time_end_utc);
    if (Number.isNaN(startMs) || Number.isNaN(endMs)) continue;
    let startIndex = -1;
    let endIndex = -1;
    timestamps.forEach((timestamp, index) => {
      const timestampMs = chartTimestampMs(timestamp);
      if (timestampMs >= startMs && startIndex < 0) startIndex = index;
      if (timestampMs <= endMs) endIndex = index;
    });
    if (startIndex < 0 || endIndex < startIndex) continue;
    areas.push([
      { xAxis: Math.max(0, startIndex - 0.5) },
      { xAxis: Math.min(timestamps.length - 1, endIndex + 0.5) },
    ]);
  }
  return areas;
}

function layeredMarkAreas(
  timestamps: string[],
  isAnomaly: boolean[],
  splitRanges: ChartSplitRange[] | undefined,
  gtEvents: GtEvent[],
  layers: TimesnetChartLayers,
) {
  const gtAreas = layers.gtBands ? buildGtMarkAreas(timestamps, gtEvents) : [];
  const modelAreas = layers.modelBands
    ? buildModelAnomalyMarkAreas(timestamps, isAnomaly)
    : [];
  const splitAreas = layers.splitBands ? buildSplitMarkAreas(splitRanges) : [];
  return [
    ...splitAreas,
    ...gtAreas.map((segment) => [
      { ...segment[0], itemStyle: { color: GT_BAND_COLOR } },
      segment[1],
    ]),
    ...modelAreas.map((segment) => [
      { ...segment[0], itemStyle: { color: MODEL_BAND_COLOR } },
      segment[1],
    ]),
  ];
}

export function buildTimesnetScoreChartOption(
  data: TimesnetScoresResponse,
  displayName?: string,
  gtEvents: GtEvent[] = [],
  layers: TimesnetChartLayers = DEFAULT_TIMESNET_CHART_LAYERS,
  windowPreset: ChartWindowPreset = 'test',
): Record<string, unknown> {
  const useSampling = data.timestamps.length > 80;
  const anomalyData: Array<[number, number]> = [];
  data.is_anomaly.forEach((flag, index) => {
    if (flag) anomalyData.push([index, data.score[index]]);
  });
  const zoom = resolveChartZoom(windowPreset, data.timestamps, data.split_ranges);
  const cutoff = Number(data.cutoff ?? 0);
  const markAreaData = layeredMarkAreas(
    data.timestamps,
    data.is_anomaly,
    data.split_ranges,
    gtEvents,
    layers,
  );

  return {
    backgroundColor: 'transparent',
    title: {
      text: displayName ?? data.unique_id,
      subtext: chartPointSubtitle(windowPreset, data),
      left: 0,
      top: 0,
      textStyle: { color: '#9aa0a6', fontSize: 11, fontWeight: 500 },
      subtextStyle: { color: '#6e7681', fontSize: 10 },
    },
    tooltip: {
      trigger: 'axis',
      backgroundColor: '#111217',
      borderColor: '#2c3235',
      textStyle: { color: '#d8d9da', fontSize: 12 },
    },
    grid: { left: 48, right: 24, bottom: 56, top: 48, containLabel: true },
    dataZoom: chartDataZoom(zoom.start, zoom.end),
    xAxis: {
      type: 'category',
      boundaryGap: false,
      data: data.timestamps,
      axisLine: { lineStyle: { color: '#2c3235' } },
      axisLabel: { color: '#9aa0a6', fontSize: 10 },
    },
    yAxis: {
      type: 'value',
      splitLine: { lineStyle: { color: '#1f2328' } },
      axisLabel: { color: '#9aa0a6', fontSize: 10 },
    },
    series: [
      {
        name: 'Score',
        type: 'line',
        smooth: 0.2,
        showSymbol: false,
        sampling: useSampling ? 'lttb' : undefined,
        large: useSampling,
        lineStyle: { width: 1.8, color: '#5794f2' },
        itemStyle: { color: '#5794f2' },
        data: data.score,
        ...(cutoff > 0
          ? {
              markLine: {
                silent: true,
                symbol: 'none',
                lineStyle: { color: '#ff9830', type: 'dashed', width: 1 },
                label: {
                  formatter: `cutoff ${cutoff.toFixed(3)}`,
                  color: '#ff9830',
                  fontSize: 10,
                },
                data: [{ yAxis: cutoff }],
              },
            }
          : {}),
        ...(markAreaData.length
          ? { markArea: { silent: true, data: markAreaData } }
          : {}),
      },
      ...(layers.modelPoints
        ? [
            {
              name: 'Anomaly',
              type: 'scatter',
              symbolSize: 7,
              itemStyle: { color: '#f2495c' },
              data: anomalyData,
              z: 10,
            },
          ]
        : []),
    ],
    animation: false,
  };
}

function markerAreas(
  markers: TimesnetLocationChartResponse['anomaly_markers'],
): Array<[{ xAxis: number }, { xAxis: number }]> {
  return markers.map((marker) => [
    { xAxis: Math.max(0, marker.index - 0.45) },
    { xAxis: marker.index + 0.45 },
  ]);
}

function markerPoints(
  total: number[],
  markers: TimesnetLocationChartResponse['anomaly_markers'],
) {
  return markers.map((marker) => ({
    coord: [marker.index, total[marker.index] ?? 0] as [number, number],
    name: marker.label,
    itemStyle: { color: '#f2495c' },
  }));
}

export function buildTimesnetLocationStackOption(
  data: TimesnetLocationChartResponse,
  mode: 'actual' | 'predicted',
  gtEvents: GtEvent[] = [],
  layers: TimesnetChartLayers = DEFAULT_TIMESNET_CHART_LAYERS,
  windowPreset: ChartWindowPreset = 'test',
): Record<string, unknown> {
  const total = mode === 'actual' ? data.total_actual : data.total_predicted;
  const title =
    mode === 'actual'
      ? `${data.location_id} — actual (all classes)`
      : `${data.location_id} — predicted (all classes)`;
  const barSeries = data.classes.map((item) => ({
    name: item.label,
    type: 'bar',
    stack: mode,
    emphasis: { focus: 'series' },
    itemStyle: { color: item.color },
    data: mode === 'actual' ? item.actual : item.predicted,
  }));
  const zoom = resolveChartZoom(windowPreset, data.timestamps, data.split_ranges);
  const gtAreas = layers.gtBands ? buildGtMarkAreas(data.timestamps, gtEvents) : [];
  const modelAreas = layers.modelBands ? markerAreas(data.anomaly_markers) : [];
  const splitAreas = layers.splitBands ? buildSplitMarkAreas(data.split_ranges) : [];
  const markAreaData = [
    ...splitAreas,
    ...gtAreas.map((segment) => [
      { ...segment[0], itemStyle: { color: GT_BAND_COLOR } },
      segment[1],
    ]),
    ...modelAreas.map((segment) => [
      { ...segment[0], itemStyle: { color: MODEL_BAND_COLOR } },
      segment[1],
    ]),
  ];

  return {
    backgroundColor: 'transparent',
    title: {
      text: title,
      subtext: chartPointSubtitle(windowPreset, data),
      left: 0,
      top: 0,
      textStyle: { color: '#9aa0a6', fontSize: 11, fontWeight: 500 },
      subtextStyle: { color: '#6e7681', fontSize: 10 },
    },
    tooltip: {
      trigger: 'axis',
      axisPointer: { type: 'shadow' },
      backgroundColor: '#111217',
      borderColor: '#2c3235',
      textStyle: { color: '#d8d9da', fontSize: 12 },
    },
    legend: {
      type: 'scroll',
      data: [...data.classes.map((item) => item.label), 'total'],
      textStyle: { color: '#9aa0a6', fontSize: 10 },
      bottom: 28,
    },
    grid: { left: 48, right: 24, bottom: 56, top: 48, containLabel: true },
    dataZoom: chartDataZoom(zoom.start, zoom.end),
    xAxis: {
      type: 'category',
      data: data.timestamps,
      axisLine: { lineStyle: { color: '#2c3235' } },
      axisLabel: { color: '#9aa0a6', fontSize: 9, hideOverlap: true },
    },
    yAxis: {
      type: 'value',
      splitLine: { lineStyle: { color: '#1f2328' } },
      axisLabel: { color: '#9aa0a6', fontSize: 10 },
    },
    series: [
      ...barSeries,
      {
        name: 'total',
        type: 'line',
        data: total,
        showSymbol: false,
        lineStyle: { width: 1.4, color: '#e8eaed' },
        z: 5,
        markArea: markAreaData.length
          ? { silent: true, data: markAreaData }
          : undefined,
        markPoint: layers.modelPoints
          ? {
              symbol: 'circle',
              symbolSize: 8,
              data: markerPoints(total, data.anomaly_markers),
              label: { show: false },
            }
          : undefined,
      },
    ],
    animation: false,
  };
}

export function buildTimesnetForecastChartOption(
  data: TimesnetScoresResponse,
  displayName?: string,
  gtEvents: GtEvent[] = [],
  layers: TimesnetChartLayers = DEFAULT_TIMESNET_CHART_LAYERS,
  windowPreset: ChartWindowPreset = 'test',
): Record<string, unknown> | null {
  const actual = data.actual;
  const predicted = data.predicted;
  if (!actual?.length && !predicted?.length) return null;
  const predictedLabel = data.value_kind === 'reconstruction' ? 'reconstructed' : 'predicted';
  const warnPoints: Array<[number, number]> = [];
  const alertPoints: Array<[number, number]> = [];
  const markerValues = actual?.length ? actual : predicted ?? [];
  data.is_anomaly.forEach((flag, index) => {
    if (!flag) return;
    const point: [number, number] = [index, markerValues[index] ?? 0];
    if ((data.severity?.[index] ?? '').toLowerCase() === 'alert') alertPoints.push(point);
    else warnPoints.push(point);
  });
  const zoom = resolveChartZoom(windowPreset, data.timestamps, data.split_ranges);
  const markAreaData = layeredMarkAreas(
    data.timestamps,
    data.is_anomaly,
    data.split_ranges,
    gtEvents,
    layers,
  );
  const legend = ['actual', predictedLabel, 'anomaly (warn)', 'anomaly (alert)'].filter(
    (name) =>
      (name !== 'actual' || !!actual?.length) &&
      (name !== predictedLabel || !!predicted?.length) &&
      (name !== 'anomaly (warn)' || warnPoints.length > 0) &&
      (name !== 'anomaly (alert)' || alertPoints.length > 0),
  );

  return {
    backgroundColor: 'transparent',
    title: {
      text: displayName ?? seriesDisplayName(data.unique_id),
      subtext: `Actual vs ${predictedLabel} · ${chartPointSubtitle(windowPreset, data)}`,
      left: 0,
      top: 0,
      textStyle: { color: '#9aa0a6', fontSize: 11, fontWeight: 500 },
      subtextStyle: { color: '#6e7681', fontSize: 10 },
    },
    tooltip: {
      trigger: 'axis',
      backgroundColor: '#111217',
      borderColor: '#2c3235',
      textStyle: { color: '#d8d9da', fontSize: 12 },
    },
    legend: { data: legend, textStyle: { color: '#9aa0a6', fontSize: 10 }, bottom: 28 },
    grid: { left: 48, right: 24, bottom: 56, top: 48, containLabel: true },
    dataZoom: chartDataZoom(zoom.start, zoom.end),
    xAxis: {
      type: 'category',
      boundaryGap: false,
      data: data.timestamps,
      axisLine: { lineStyle: { color: '#2c3235' } },
      axisLabel: { color: '#9aa0a6', fontSize: 10 },
    },
    yAxis: {
      type: 'value',
      splitLine: { lineStyle: { color: '#1f2328' } },
      axisLabel: { color: '#9aa0a6', fontSize: 10 },
    },
    series: [
      ...(actual?.length
        ? [
            {
              name: 'actual',
              type: 'line',
              smooth: 0.15,
              showSymbol: false,
              lineStyle: { width: 1.8, color: '#5794f2' },
              data: actual,
              ...(markAreaData.length
                ? { markArea: { silent: true, data: markAreaData } }
                : {}),
            },
          ]
        : []),
      ...(predicted?.length
        ? [
            {
              name: predictedLabel,
              type: 'line',
              smooth: 0.15,
              showSymbol: false,
              lineStyle: { width: 1.6, color: '#ff9830', type: 'dashed' },
              data: predicted,
            },
          ]
        : []),
      ...(layers.modelPoints && warnPoints.length
        ? [
            {
              name: 'anomaly (warn)',
              type: 'scatter',
              symbolSize: 9,
              itemStyle: { color: '#ff9830', borderColor: '#ff9830' },
              data: warnPoints,
              z: 10,
            },
          ]
        : []),
      ...(layers.modelPoints && alertPoints.length
        ? [
            {
              name: 'anomaly (alert)',
              type: 'scatter',
              symbolSize: 11,
              itemStyle: { color: '#f2495c' },
              data: alertPoints,
              z: 11,
            },
          ]
        : []),
    ],
    animation: false,
  };
}
