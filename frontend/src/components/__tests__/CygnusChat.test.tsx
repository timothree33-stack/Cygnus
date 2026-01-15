import React from 'react';
import { render, screen } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import CygnusChat from '../CygnusChat';

describe('CygnusChat', () => {
  it('sends a message and displays a response', async () => {
    global.fetch = vi.fn(() => Promise.resolve({ ok: true, json: () => Promise.resolve({ response: 'Hello from Cygnus' }) }));

    render(<CygnusChat />);

    const input = screen.getByPlaceholderText('Ask Cygnus...');
    await userEvent.type(input, 'Test question');

    const send = screen.getByText('Send');
    await userEvent.click(send);

    expect(global.fetch).toHaveBeenCalled();
    expect(await screen.findByText('Hello from Cygnus')).toBeInTheDocument();
  });
});
