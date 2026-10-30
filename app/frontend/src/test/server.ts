import { http, HttpResponse } from 'msw';
import { setupServer } from 'msw/node';
import type { DatasetsResponse } from '../types';

const sampleDatasets: DatasetsResponse = {
  datasets: [
    {
      id: 'dataset-a',
      label: 'Dataset A',
      table: 'vehicletrace_feature_dataset_a',
      station_count: 0,
      location_count: 0,
      stations: [],
    },
  ],
  range_presets: {
    '15m': { label: '15 minutes', resolution: '15min', hours: 0.25 },
    '1h': { label: '1 hour', resolution: '15min', hours: 1 },
    '1d': { label: '1 day', resolution: '1h', hours: 24 },
    '7d': { label: '7 days', resolution: '6h', hours: 168 },
    '30d': { label: '30 days', resolution: '1d', hours: 720 },
  },
  max_points_hint: 2500,
};

export const server = setupServer(
  http.get('/api/datasets', () => HttpResponse.json(sampleDatasets)),
);

export { sampleDatasets };
