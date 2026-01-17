import { test, expect } from '@playwright/test';

test('persona: list and add entry', async ({ page }) => {
  await page.goto('/');
  await page.getByRole('link', { name: 'Persona' }).click();

  // Stub list and add endpoints
  await page.route('**/api/admin/agents/*/persona', (route) => {
    if (route.request().method() === 'GET') {
      route.fulfill({ status: 200, body: JSON.stringify({ persona: [{ id: 'p1', content: 'Existing' }] }), headers: { 'Content-Type': 'application/json' } });
    } else if (route.request().method() === 'POST') {
      route.fulfill({ status: 200, body: JSON.stringify({ saved: 'p2' }), headers: { 'Content-Type': 'application/json' } });
    } else {
      route.continue();
    }
  });

  await page.fill('input[placeholder="Agent: "]', 'cygnus');
  await page.click('text=Load');

  await expect(page.getByText('Existing')).toBeVisible();

  await page.fill('textarea[placeholder="Add persona entry..."]', 'New persona entry');
  await page.click('text=Add');
  await expect(page.getByText('New persona entry')).toBeVisible();
});