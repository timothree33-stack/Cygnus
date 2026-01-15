import { test, expect } from '@playwright/test';

const PAGES = ['/','/dashboard','/agents','/crawler','/debate'];

for (const path of PAGES) {
  test(`smoke: ${path}`, async ({ page }) => {
    // Use relative path so Playwright's baseURL (from config) is applied.
    const resp = await page.goto(path, { waitUntil: 'domcontentloaded' });
    // Ensure navigation succeeded and page is not an error page
    expect(resp).toBeTruthy();
    const status = resp?.status() ?? 0;
    expect(status === 200 || status === 304 || status === 0).toBeTruthy();

    // Basic assertion: page should have a title (non-empty) or a visible body
    const title = await page.title();
    if (title && title.length > 0) {
      expect(title).toBeDefined();
    } else {
      const bodyVisible = await page.locator('body').isVisible();
      expect(bodyVisible).toBeTruthy();
    }
  });
}
