import { render, screen } from '@testing-library/react';
import { describe, expect, it } from 'vitest';
import { KpiCard } from './KpiCard';

describe('KpiCard', () => {
  it('renders label, value, and optional subtext', () => {
    render(<KpiCard label="Runs" value="12" sub="last 7d" />);
    expect(screen.getByText('Runs')).toBeInTheDocument();
    expect(screen.getByText('12')).toBeInTheDocument();
    expect(screen.getByText('last 7d')).toBeInTheDocument();
  });
});
