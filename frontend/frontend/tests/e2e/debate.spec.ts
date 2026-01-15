import { test, expect } from '@playwright/test';

test('debate: enable court-fool toggle', async ({ page }) => {
  await page.goto('/');
  await page.getByRole('link', { name: 'Debate' }).click();
  const checkbox = page.locator('input[type="checkbox"]');
  await expect(checkbox).toBeVisible();
  await checkbox.check();
  await expect(page.getByText('Court-Fool enabled')).toBeVisible();
});
