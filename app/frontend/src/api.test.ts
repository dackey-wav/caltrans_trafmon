import { describe, expect, it } from 'vitest';
import { buildTrafficEndUtc, fetchDatasets } from './api';

describe('buildTrafficEndUtc', () => {
  it('uses hour boundary for 1h preset', () => {
    expect(buildTrafficEndUtc('1h', '2024-12-02', 9)).toBe('2024-12-02 09:59:59');
  });

  it('clamps hour to 0–23', () => {
    expect(buildTrafficEndUtc('1h', '2024-12-02', 99)).toBe('2024-12-02 23:59:59');
  });

  it('uses end of day for multi-day presets', () => {
    expect(buildTrafficEndUtc('7d', '2024-12-02', 5)).toBe('2024-12-02 23:59:59');
  });
});

describe('fetchDatasets query', () => {
  it('loads datasets from /api/datasets', async () => {
    const body = await fetchDatasets();
    expect(body.datasets[0]?.id).toBe('dataset-a');
    expect(body.max_points_hint).toBeGreaterThan(0);
  });
});
