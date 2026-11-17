import { describe, expect, it } from 'vitest';
import { buildLiveChartOption, scaled, windowErrors } from './chart';
import type { Bucket } from './types';

const t0 = Date.parse('2025-06-02T07:00:00Z');
const bucket = (i: number, a: number | null, p: number, w: number): Bucket => ({
  t: new Date(t0 + i * 900_000).toISOString(),
  a: a == null ? null : [a, 60, 0.05],
  w: [w, 60, 0.05],
  p: { '1h': [p, 60, 0.05] },
  x: false,
});

describe('live chart helpers', () => {
  it('shows flow per hour and occupancy in percent', () => {
    expect(scaled([100, 60, 0.05], 'flow')).toBe(400);
    expect(scaled([100, 60, 0.05], 'occupancy')).toBeCloseTo(5);
    expect(scaled(null, 'speed')).toBeNull();
  });

  it('scores only buckets with actual, prediction and last week', () => {
    const buckets = [bucket(0, 100, 110, 130), bucket(1, null, 90, 90), bucket(2, 100, 90, 100)];
    const e = windowErrors(buckets, 'flow', '1h', t0)!;
    expect(e.n).toBe(2);
    expect(e.model).toBe(40); // |440-400| and |360-400|, veh/h
    expect(e.weekly).toBe(60);
    expect(e.skill).toBeCloseTo(1 / 3);
  });

  it('draws the now line and leaves gaps where data is missing', () => {
    const option = buildLiveChartOption({
      buckets: [bucket(0, 100, 110, 130), bucket(1, null, 90, 90)],
      readings: [[t0, 30, 60, 0.05, 1], [t0 + 300_000, 31, 60, 0.05, 0]],
      forecast: null,
      channel: 'flow',
      lead: '1h',
      nowMs: t0 + 1_800_000,
      pastHours: 24,
      showWeekly: false,
      showDeviations: false,
    }) as { series: Array<{ name: string; data: Array<[number, number | null]>; markLine?: unknown }> };
    const actual = option.series.find((s) => s.name.startsWith('Actual'))!;
    expect(actual.data[1][1]).toBeNull();
    expect(actual.markLine).toBeTruthy();
    const imputed = option.series.find((s) => s.name.startsWith('Imputed'))!;
    expect(imputed.data.map((p) => p[1])).toEqual([null, 372]);
  });
});
