import { render, screen } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { describe, expect, it, vi } from 'vitest';
import { TrafficSidebar } from './traffic/TrafficSidebar';
import type { DatasetInfo } from './types';

const dataset: DatasetInfo = {
  id: 'dataset-a',
  label: 'Dataset A',
  table: 'vehicletrace_feature_dataset_a',
  station_count: 1,
  location_count: 1,
  stations: [
    {
      base_id: '10001-seg',
      label: 'Site 10001',
      lanes: [{ location_id: '10001-seg-l1', lane: 'l1' }],
    },
  ],
};

describe('TrafficSidebar', () => {
  it('opens chart mode menu from the root panel', async () => {
    const user = userEvent.setup();
    const onOpenMode = vi.fn();
    render(
      <TrafficSidebar
        chartMode={null}
        panel="menu"
        datasets={[dataset]}
        expandedDatasets={{}}
        expandedBases={{}}
        selectedLocations={new Set()}
        classSelection={null}
        onOpenMode={onOpenMode}
        onOpenModels={() => {}}
        onBack={() => {}}
        onToggleDataset={() => {}}
        onToggleBase={() => {}}
        onSetBaseSelection={() => {}}
        onToggleLane={() => {}}
        onSelectClassStation={() => {}}
        onSetClassLaneScope={() => {}}
      />,
    );
    await user.click(screen.getByRole('button', { name: /vehicles by count/i }));
    expect(onOpenMode).toHaveBeenCalledWith('count');
  });
});
