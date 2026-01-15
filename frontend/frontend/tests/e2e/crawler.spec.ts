import { test, expect } from '@playwright/test';

test('crawler page has title and controls', async ({ page }) => {
  await page.goto('/');
  // Click the nav link (disambiguates link vs page heading)
  await page.getByRole('link', { name: 'Crawler' }).click();
  // Ensure the page heading is visible
  await expect(page.getByRole('heading', { name: 'Crawler' })).toBeVisible();
});
