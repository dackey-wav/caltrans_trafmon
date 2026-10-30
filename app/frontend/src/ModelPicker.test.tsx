import { render, screen } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { describe, expect, it, vi } from 'vitest';
import { ModelPicker } from './ModelPicker';
import type { LocalModelInfo } from './types';

const models: LocalModelInfo[] = [
  {
    id: 'timesnet',
    label: 'TimesNet',
    family: 'anomaly',
    artifacts_root: 'artifacts/timesnet',
    runs: [{ run_id: 'r1' }],
    run_count: 1,
    tagline: 'Periodicity-aware anomaly',
    focus: 'locations',
  },
];

describe('ModelPicker', () => {
  it('shows loading state', () => {
    render(<ModelPicker models={[]} loading onSelect={() => {}} />);
    expect(screen.getByText(/loading models/i)).toBeInTheDocument();
  });

  it('calls onSelect when a model tile is clicked', async () => {
    const user = userEvent.setup();
    const onSelect = vi.fn();
    render(<ModelPicker models={models} loading={false} onSelect={onSelect} />);
    await user.click(screen.getByRole('button', { name: /timesnet/i }));
    expect(onSelect).toHaveBeenCalledWith('timesnet');
  });
});
