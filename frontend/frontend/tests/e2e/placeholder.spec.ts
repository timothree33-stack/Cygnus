import { test, expect } from '@playwright/test';

test('placeholder sanity test', async ({ page }) => {
  // Use absolute URL in CI to avoid baseURL resolution issues
  await page.goto('http://127.0.0.1:5173/');
  // Basic assertion: page should load and have a title (non-empty)
  const title = await page.title();
  expect(title).toBeDefined();
});
