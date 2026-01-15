import { test, expect } from '@playwright/test';

test('dashboard loads and shows conversation box', async ({ page }) => {
  await page.goto('/');
  // Click the nav link to navigate to dashboard (disambiguate from the page heading)
  await page.getByRole('link', { name: 'Dashboard' }).click();
  // Ensure the Dashboard page heading is visible
  await expect(page.getByRole('heading', { name: 'Dashboard' })).toBeVisible();
  await expect(page.getByText('Agent: dashboard')).toBeVisible();
});
