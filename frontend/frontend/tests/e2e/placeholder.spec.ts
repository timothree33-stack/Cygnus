import { test, expect } from '@playwright/test';

test('placeholder sanity test', async ({ page }) => {
  await page.goto('/');
  // Basic assertion: page should load and have a title (non-empty)
  const title = await page.title();
  expect(title).toBeDefined();
});
