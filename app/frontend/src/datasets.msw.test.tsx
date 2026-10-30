import { useEffect, useState } from 'react';
import { render, screen, waitFor } from '@testing-library/react';
import { describe, expect, it } from 'vitest';
import { fetchDatasets } from './api';

function DatasetsProbe() {
  const [label, setLabel] = useState('pending');
  useEffect(() => {
    fetchDatasets().then((body) => setLabel(body.datasets[0]?.label ?? 'missing'));
  }, []);
  return <div data-testid="dataset-label">{label}</div>;
}

describe('datasets MSW smoke', () => {
  it('serves /api/datasets via MSW', async () => {
    render(<DatasetsProbe />);
    await waitFor(() => {
      expect(screen.getByTestId('dataset-label')).toHaveTextContent('Dataset A');
    });
  });
});
