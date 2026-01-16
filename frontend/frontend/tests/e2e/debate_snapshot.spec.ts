import { test, expect } from '@playwright/test';

// Snapshot modal smoke test
test('debate: snapshot modal opens', async ({ page, request }) => {
  const base = 'http://127.0.0.1:8001';
  const r = await request.post(`${base}/api/debate/start?pause_sec=1&rounds=2`);
  expect(r.ok()).toBeTruthy();
  const body = await r.json();
  const debateId = body.debate_id;

  await page.goto('/');
  await page.getByRole('link', { name: 'Debate' }).click();
  // Join by id
  await page.fill('input[placeholder="Or paste debate id to join"]', debateId);
  await page.click('text=Join');

  // Trigger a camera capture and then open the resulting snapshot
  await page.click('button:has-text("Capture from camera")');
  await page.waitForSelector('button:has-text("View snapshot")', { timeout: 20000 });
  await page.click('button:has-text("View snapshot")');
  await page.waitForSelector('text=Snapshot details', { timeout: 5000 });
  await expect(page.getByText('Snapshot details')).toBeVisible();
});