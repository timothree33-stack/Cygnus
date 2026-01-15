import React from 'react';
import { render, screen } from '@testing-library/react';
import Dashboard from '../../components/dashboard/Dashboard';

describe('Dashboard', () => {
  it('renders metric cards and server status', async () => {
    const metrics = { cpu_percent: 12.3, cpu_count: 4, ram_percent: 44.4, ram_used: 1024, ram_total: 2048 };
    const llm = { servers: { katz: { port: 8082, status: 'online' }, dogz: { port: 8083, status: 'online' }, cygnus: { port: 8081, status: 'online' } }, all_online: true };
    const vision = { running: false, chunks: 0, memory: 0 };
    const tailscale = { connected: false };
    const debate = { total_debates: 5, helix_active: false, katz_wins: 2, dogz_wins: 3 };

    global.fetch = vi.fn((url) => {
      if (url.endsWith('/metrics')) return Promise.resolve({ json: () => Promise.resolve(metrics) });
      if (url.endsWith('/llm/status')) return Promise.resolve({ json: () => Promise.resolve(llm) });
      if (url.endsWith('/vision/status')) return Promise.resolve({ json: () => Promise.resolve(vision) });
      if (url.endsWith('/tailscale/status')) return Promise.resolve({ json: () => Promise.resolve(tailscale) });
      if (url.endsWith('/debate/stats')) return Promise.resolve({ json: () => Promise.resolve(debate) });
      return Promise.resolve({ json: () => Promise.resolve({}) });
    });

    render(<Dashboard />);

    expect(await screen.findByText(/🦅 Cygnus Pyramid Dashboard/)).toBeInTheDocument();
    expect(await screen.findByText('CPU')).toBeInTheDocument();
    expect(await screen.findByText('RAM')).toBeInTheDocument();
    expect(await screen.findByText('🦅 Falcon Servers')).toBeInTheDocument();
  });
});
