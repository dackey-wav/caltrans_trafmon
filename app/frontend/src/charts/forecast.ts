import type { ForecastChartResponse, GtEvent } from '../types';
import {
  buildGtMarkAreas,
  buildModelAnomalyMarkAreas,
  buildSplitMarkAreas,
  chartDataZoom,
  chartPointSubtitle,
  DEFAULT_TIMESNET_CHART_LAYERS,
  GT_BAND_COLOR,
  MODEL_BAND_COLOR,
  resolveChartZoom,
  seriesDisplayName,
  type ChartWindowPreset,
  type TimesnetChartLayers,
} from './anomaly';

const BASELINE_COLOR = '#8a8f98';

function anomalyPoints(data: ForecastChartResponse) {
  const warning: Array<[number, number]> = [];
  const alert: Array<[number, number]> = [];
  const values = data.actual?.length ? data.actual : data.predicted ?? [];
  data.is_anomaly.forEach((flag, index) => {
    if (!flag) return;
    const point: [number, number] = [index, values[index] ?? 0];
    if ((data.severity?.[index] ?? '').toLowerCase() === 'alert') alert.push(point);
    else warning.push(point);
  });
  return { warning, alert };
}

export function buildForecastChartOption(
  data: ForecastChartResponse,
  displayName?: string,
  gtEvents: GtEvent[] = [],
  layers: TimesnetChartLayers = DEFAULT_TIMESNET_CHART_LAYERS,
  windowPreset: ChartWindowPreset = 'test',
): Record<string, unknown> | null {
  const { actual, predicted, baseline } = data;
  if (!actual?.length && !predicted?.length) return null;
  const points = anomalyPoints(data);
  const zoom = resolveChartZoom(windowPreset, data.timestamps, data.split_ranges);
  const gtAreas = layers.gtBands ? buildGtMarkAreas(data.timestamps, gtEvents) : [];
  const modelAreas = layers.modelBands
    ? buildModelAnomalyMarkAreas(data.timestamps, data.is_anomaly)
    : [];
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
  const hasBaseline = !!baseline?.length;
  const legend = ['actual', 'predicted', 'baseline', 'anomaly (warn)', 'anomaly (alert)'].filter(
    (name) =>
      (name !== 'actual' || !!actual?.length) &&
      (name !== 'predicted' || !!predicted?.length) &&
      (name !== 'baseline' || hasBaseline) &&
      (name !== 'anomaly (warn)' || points.warning.length > 0) &&
      (name !== 'anomaly (alert)' || points.alert.length > 0),
  );

  return {
    backgroundColor: 'transparent',
    title: {
      text: displayName ?? seriesDisplayName(data.unique_id),
      subtext: `Actual vs forecast (${data.scope}) · ${chartPointSubtitle(windowPreset, data)}`,
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
              name: 'predicted',
              type: 'line',
              smooth: 0.15,
              showSymbol: false,
              lineStyle: { width: 1.6, color: '#ff9830', type: 'dashed' },
              data: predicted,
            },
          ]
        : []),
      ...(hasBaseline
        ? [
            {
              name: 'baseline',
              type: 'line',
              smooth: 0.15,
              showSymbol: false,
              lineStyle: { width: 1.2, color: BASELINE_COLOR, type: 'dotted' },
              data: baseline,
            },
          ]
        : []),
      ...(layers.modelPoints && points.warning.length
        ? [
            {
              name: 'anomaly (warn)',
              type: 'scatter',
              symbolSize: 9,
              itemStyle: { color: '#ff9830' },
              data: points.warning,
              z: 10,
            },
          ]
        : []),
      ...(layers.modelPoints && points.alert.length
        ? [
            {
              name: 'anomaly (alert)',
              type: 'scatter',
              symbolSize: 11,
              itemStyle: { color: '#f2495c' },
              data: points.alert,
              z: 11,
            },
          ]
        : []),
    ],
    animation: false,
  };
}

export function buildForecastResidualOption(
  data: ForecastChartResponse,
  displayName?: string,
  gtEvents: GtEvent[] = [],
  layers: TimesnetChartLayers = DEFAULT_TIMESNET_CHART_LAYERS,
  windowPreset: ChartWindowPreset = 'test',
): Record<string, unknown> | null {
  const residual = data.residual;
  if (!residual?.length) return null;
  const anomalyData: Array<[number, number]> = [];
  data.is_anomaly.forEach((flag, index) => {
    if (flag) anomalyData.push([index, residual[index] ?? 0]);
  });
  const zoom = resolveChartZoom(windowPreset, data.timestamps, data.split_ranges);
  const markAreaData = (layers.gtBands ? buildGtMarkAreas(data.timestamps, gtEvents) : []).map(
    (segment) => [{ ...segment[0], itemStyle: { color: GT_BAND_COLOR } }, segment[1]],
  );

  return {
    backgroundColor: 'transparent',
    title: {
      text: displayName ?? seriesDisplayName(data.unique_id),
      subtext: `Residual (actual − forecast) · ${chartPointSubtitle(windowPreset, data)}`,
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
        name: 'residual',
        type: 'line',
        smooth: 0.1,
        showSymbol: false,
        lineStyle: { width: 1.4, color: '#73bf69' },
        data: residual,
        markLine: {
          silent: true,
          symbol: 'none',
          lineStyle: { color: '#6e7681', type: 'dashed', width: 1 },
          data: [{ yAxis: 0 }],
        },
        ...(markAreaData.length
          ? { markArea: { silent: true, data: markAreaData } }
          : {}),
      },
      ...(layers.modelPoints && anomalyData.length
        ? [
            {
              name: 'anomaly',
              type: 'scatter',
              symbolSize: 8,
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
