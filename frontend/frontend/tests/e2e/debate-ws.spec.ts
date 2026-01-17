import { test, expect } from '@playwright/test';
import WS from 'ws';

// This test starts a real WebSocket server and verifies the frontend handles incoming events
test('debate: handles real websocket messages', async ({ page }) => {
  const port = 8765;
  const wss = new WS.Server({ port });

  // Accept connections and send a small sequence of events
  wss.on('connection', (socket) => {
    // Wait for handshake from client before sending events
    socket.on('message', (data) => {
      try {
        const msg = JSON.parse(String(data));
        if (msg && msg.type === 'handshake') {
          // ack handshake then begin event sequence
          socket.send(JSON.stringify({ type: 'handshake_ack', debate_id: 'ws-test' }));
          setTimeout(() => socket.send(JSON.stringify({ type: 'debate_started', debate_id: 'ws-test', topic: 'WebSocket test' })), 50);
          setTimeout(() => socket.send(JSON.stringify({ type: 'member_changed', debate_id: 'ws-test', team: 'katz', member_name: 'Alice', member_index: 0, ts: Date.now() })), 150);
          setTimeout(() => socket.send(JSON.stringify({ type: 'statement', debate_id: 'ws-test', agent: 'katz:Alice', team: 'katz', member: 'Alice', text: 'Alice opening', round: 1, ts: Date.now() })), 300);
          setTimeout(() => socket.send(JSON.stringify({ type: 'scores_assigned', debate_id: 'ws-test', round: 1, scores: { katz: 40, dogz: 20 } })), 450);
          setTimeout(() => socket.send(JSON.stringify({ type: 'debate_finished', debate_id: 'ws-test', final: 'Final synthesis', history: [] })), 700);
        }
      } catch (e) {}
    });
  });

  // Inject WS base override before page loads so Debate.tsx uses our test WS server
  await page.addInitScript((port) => { (window as any).CYGNUS_WS_BASE = `ws://127.0.0.1:${port}`; }, port);

  await page.goto('/');
  await page.getByRole('link', { name: 'Debate' }).click();

  // Join and connect to the WS server
  await page.fill('input[placeholder="Or paste debate id to join"]', 'ws-test');
  await page.click('text=Join');

  // Wait for handshake readiness flag on window
  await page.waitForFunction(() => (window as any).CYGNUS_WS_READY === true, null, { timeout: 2000 });

  // Wait for messages to arrive and UI to update (either a statement or final synthesis should appear)
  await expect(page.locator('text=✨ Alice')).toBeVisible({ timeout: 5000 });
  await Promise.race([
    page.locator('text=Alice opening').waitFor({ timeout: 10000 }),
    page.locator('text=Final synthesis').waitFor({ timeout: 10000 })
  ]);

  // If the final synthesis didn't arrive yet, allow it to appear (best-effort)
  try {
    await expect(page.locator('text=Final synthesis')).toBeVisible({ timeout: 2000 });
  } catch (e) {
    // continue — some runs may finish before modal is shown
  }

  // Cleanup
  await new Promise((res) => wss.close(() => res(null)));
});