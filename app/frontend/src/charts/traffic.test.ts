import { describe, expect, it } from 'vitest';
import { buildClassChartOption, buildCountChartOption } from './traffic';
import type { ClassTrafficResponse, TrafficResponse } from '../types';

const trafficFixture: TrafficResponse = {
  timestamps: ['2024-12-02 10:00', '2024-12-02 11:00'],
  series: {
    'loc-a': [1, 3],
    'loc-b': [2, 0],
  },
};

describe('buildCountChartOption', () => {
  it('builds one line series per location', () => {
    const option = buildCountChartOption(trafficFixture);
    const series = option.series as Array<{ name: string; data: number[] }>;
    expect(series).toHaveLength(2);
    expect(series[0]?.data).toEqual([1, 3]);
    expect(option.xAxis).toMatchObject({ data: trafficFixture.timestamps });
  });
});

const classFixture: ClassTrafficResponse = {
  timestamps: ['2024-12-02 10:00'],
  classes: [1, 7],
  series: { '1': [4], '7': [9] },
  class_labels: { '1': 'cars', '7': 'trucks' },
};

describe('buildClassChartOption', () => {
  it('stacks bars by vehicle class', () => {
    const option = buildClassChartOption(classFixture);
    const series = option.series as Array<{ name: string; stack: string }>;
    expect(series).toHaveLength(2);
    expect(series.every((s) => s.stack === 'vehicles')).toBe(true);
    expect(series[1]?.name).toBe('trucks');
  });
});
