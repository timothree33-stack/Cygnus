import { test, expect } from '@playwright/test';

test('debate: enable court-fool toggle', async ({ page }) => {
  await page.goto('/');
  await page.getByRole('link', { name: 'Debate' }).click();
  const checkbox = page.locator('input[type="checkbox"]');
  await expect(checkbox).toBeVisible();
  await checkbox.check();
  await expect(page.getByText('Court-Fool enabled')).toBeVisible();
});

// Full debate smoke test (fast pause) — starts a debate via backend then joins in the UI and waits for rounds
test('debate: full 5-round flow (smoke)', async ({ page, request }) => {
  // Start a fast debate directly via API with pause_sec=1 to speed up test
  const base = 'http://127.0.0.1:8001';
  const r = await request.post(`${base}/api/debate/start?pause_sec=1&rounds=5`);
  expect(r.ok()).toBeTruthy();
  const body = await r.json();
  const debateId = body.debate_id;
  await page.goto('/');
  await page.getByRole('link', { name: 'Debate' }).click();
  // Join by id
  await page.fill('input[placeholder="Or paste debate id to join"]', debateId);
  await page.click('text=Join');
  // Wait for round 5 to appear (or until timeout)
  await page.waitForSelector('text=Round 5', { timeout: 30000 });
  await expect(page.getByText('Round 5', { exact: true })).toBeVisible();

  // Verify scoreboard updated and visible
  await expect(page.getByText('Scoreboard')).toBeVisible();
  // Wait for at least one inning cell to be populated (score >= 0 shown)
  await page.waitForSelector('table tbody tr td', { timeout: 30000 });
  const totalCats = await page.locator('text=Cats (Katz)').locator('xpath=../td[last()]').innerText();
  const totalDogs = await page.locator('text=Dogs (Dogz)').locator('xpath=../td[last()]').innerText();
  expect(parseInt(totalCats || '0')).toBeGreaterThanOrEqual(0);
  expect(parseInt(totalDogs || '0')).toBeGreaterThanOrEqual(0);
});
