import { test, expect } from '@playwright/test';

test('crawler page has title and controls', async ({ page }) => {
  await page.goto('/');
  await page.getByText('Crawler').click();
  await expect(page.getByText('Crawler')).toBeVisible();
});
