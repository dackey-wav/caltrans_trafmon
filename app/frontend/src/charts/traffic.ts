import type { ClassTrafficResponse, TrafficResponse } from '../types';
import { seriesDisplayName } from './anomaly';

const LINE_COLORS = [
  '#5794f2',
  '#ff9830',
  '#73bf69',
  '#f2495c',
  '#b877d9',
  '#fade2a',
  '#56d0e0',
  '#ff85c0',
  '#c7d0d9',
  '#8ab8ff',
];

export function buildCountChartOption(data: TrafficResponse): Record<string, unknown> {
  const names = Object.keys(data.series);
  const useSampling = data.timestamps.length > 80;
  const series = names.map((locationId) => ({
    name: seriesDisplayName(locationId),
    type: 'line',
    smooth: 0.25,
    showSymbol: false,
    sampling: useSampling ? 'lttb' : undefined,
    large: useSampling,
    lineStyle: { width: 1.8 },
    emphasis: { focus: 'series' },
    data: data.series[locationId],
  }));

  return {
    backgroundColor: 'transparent',
    tooltip: {
      trigger: 'axis',
      backgroundColor: '#111217',
      borderColor: '#2c3235',
      textStyle: { color: '#d8d9da', fontSize: 12 },
    },
    legend: {
      type: 'scroll',
      data: names.map(seriesDisplayName),
      textStyle: { color: '#9aa0a6', fontSize: 11 },
      bottom: 0,
      pageIconColor: '#5794f2',
      pageTextStyle: { color: '#9aa0a6' },
    },
    grid: { left: 48, right: 24, bottom: 56, top: 24, containLabel: true },
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
    series,
    color: LINE_COLORS,
    animation: false,
  };
}

export function buildClassChartOption(data: ClassTrafficResponse): Record<string, unknown> {
  const classIds = data.classes.length > 0 ? data.classes : Object.keys(data.series).map(Number);
  const labels = data.class_labels ?? {};
  const colors = data.class_colors ?? {};
  const series = classIds.map((classId) => {
    const key = String(classId);
    return {
      name: labels[key] ?? `cls${classId}`,
      type: 'bar',
      stack: 'vehicles',
      emphasis: { focus: 'series' },
      itemStyle: { color: colors[key] ?? '#AAAAAA' },
      data: data.series[key] ?? [],
    };
  });

  return {
    backgroundColor: 'transparent',
    tooltip: {
      trigger: 'axis',
      axisPointer: { type: 'shadow' },
      backgroundColor: '#111217',
      borderColor: '#2c3235',
      textStyle: { color: '#d8d9da', fontSize: 12 },
    },
    legend: {
      type: 'scroll',
      data: classIds.map((id) => labels[String(id)] ?? `cls${id}`),
      textStyle: { color: '#9aa0a6', fontSize: 11 },
      bottom: 0,
      pageIconColor: '#5794f2',
      pageTextStyle: { color: '#9aa0a6' },
    },
    grid: { left: 48, right: 24, bottom: 56, top: 24, containLabel: true },
    xAxis: {
      type: 'category',
      data: data.timestamps,
      axisLine: { lineStyle: { color: '#2c3235' } },
      axisLabel: { color: '#9aa0a6', fontSize: 10, hideOverlap: true },
    },
    yAxis: {
      type: 'value',
      splitLine: { lineStyle: { color: '#1f2328' } },
      axisLabel: { color: '#9aa0a6', fontSize: 10 },
    },
    series,
    animation: false,
  };
}
