import { test, expect } from '@playwright/test';

test('conversation box: send message and show in history (stubbed)', async ({ page }) => {
  // Stub backend GET and POST endpoints for deterministic test
  await page.route('**/api/admin/agents/agent-1/messages', (route) => {
    if (route.request().method() === 'GET') {
      route.fulfill({ status: 200, body: JSON.stringify({ messages: [{ id: 'm1', source: 'conversation:human', content: 'hi' }] }), headers: { 'Content-Type': 'application/json' } });
    } else if (route.request().method() === 'POST') {
      route.fulfill({ status: 200, body: JSON.stringify({ saved: 'm-new' }), headers: { 'Content-Type': 'application/json' } });
    } else {
      route.continue();
    }
  });

  // Stub delete
  await page.route('**/api/admin/agents/agent-1/messages/m1', (route) => route.fulfill({ status: 200, body: JSON.stringify({ deleted: true }), headers: { 'Content-Type': 'application/json' } }));

  await page.goto('/');
  await page.getByRole('link', { name: 'Agents' }).click();
  await expect(page.getByText('Agent: agent-1')).toBeVisible({ timeout: 5000 });

  // Confirm preloaded message shows
  await expect(page.getByText('human: hi')).toBeVisible();

  // Send new message
  const input = page.locator('.conv-input input');
  const send = page.locator('.conv-input button');
  await input.fill('Hello Quark');
  await send.click();

  // Check optimistic UI and post-replacement
  await expect(page.getByText('human: Hello Quark')).toBeVisible();

  // Prune the first message
  const prune = page.getByText('human: hi').locator('button');
  await prune.click();
  // Expect the message to no longer be present
  await expect(page.getByText('human: hi')).toHaveCount(0);
});
