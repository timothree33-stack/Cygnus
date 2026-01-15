import React from 'react';
import { render, screen } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import App from '../../App';

describe('Debate flow', () => {
  it('starts a debate when Start is clicked', async () => {
    global.fetch = vi.fn(() => Promise.resolve({ json: () => Promise.resolve({ status: 'debate_started', topic: 'Testing' }) }));

    render(<App />);

    // Enter a topic then click Start
    const input = await screen.findByPlaceholderText('Enter a debate topic...');
    await userEvent.type(input, 'Testing');
    const startButton = await screen.findByText('Start');
    await userEvent.click(startButton);

    expect(global.fetch).toHaveBeenCalled();
    expect(global.fetch).toHaveBeenCalledWith(expect.stringContaining('/api/debate/start?'), expect.any(Object));
  });
});
