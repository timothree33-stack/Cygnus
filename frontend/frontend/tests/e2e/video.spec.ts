import { test, expect } from '@playwright/test';

test('video: capture camera snapshot shows response', async ({ page }) => {
  await page.goto('/');
  await page.getByRole('link', { name: 'Video' }).click();

  // Stub camera capture endpoint
  await page.route('**/api/debate/*/camera-capture', (route) => route.fulfill({ status: 200, body: JSON.stringify({ image_saved: 'img1', memory_id: 'mem1' }), headers: { 'Content-Type': 'application/json' } }));

  // Enter debate id & capture
  await page.fill('input[placeholder="Enter debate id"]', 'fake-debate');
  await page.click('text=Capture');

  // Expect the returned snapshot JSON to be shown
  await expect(page.getByText('image_saved')).toBeVisible();
  await expect(page.getByText('mem1')).toBeVisible();
});