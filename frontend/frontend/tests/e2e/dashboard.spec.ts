import { test, expect } from '@playwright/test';

test('dashboard loads and shows conversation box', async ({ page }) => {
  await page.goto('/');
  await page.getByText('Dashboard').click();
  // Ensure the Dashboard page heading is visible (disambiguate from the nav link)
  await expect(page.getByRole('heading', { name: 'Dashboard' })).toBeVisible();
  await expect(page.getByText('Agent: dashboard')).toBeVisible();
});
