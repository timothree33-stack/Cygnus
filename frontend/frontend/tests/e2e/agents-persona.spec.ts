import { test, expect } from '@playwright/test';

test('agents: add persona entry and show in persona list', async ({ page }) => {
  // Stub memories GET to include existing persona
  await page.route('**/api/admin/memories?agent_id=agent-1', (route) => {
    route.fulfill({ status: 200, body: JSON.stringify({ memories: [{ id: 'p1', agent_id: 'agent-1', content: 'Existing persona', source: 'persona' }] }), headers: { 'Content-Type': 'application/json' } });
  });

  // Stub persona POST
  await page.route('**/api/admin/agents/agent-1/persona', (route) => {
    route.fulfill({ status: 200, body: JSON.stringify({ saved: 'p2' }), headers: { 'Content-Type': 'application/json' } });
  });

  await page.goto('/');
  await page.getByRole('link', { name: 'Agents' }).click();
  await expect(page.getByText('Persona')).toBeVisible();

  // Add persona text
  const textarea = page.locator('.persona-input textarea');
  await textarea.fill('New persona text for testing');
  await page.getByRole('button', { name: 'Add to persona' }).click();

  // New persona should appear in list
  await expect(page.getByText('New persona text for testing')).toBeVisible();
});