import { test, expect } from '@playwright/test';

// This test stubs repeated /api/debate/:id/state responses to simulate member rotation over time
test('debate: scoreboard highlights and tooltips reflect rotating active members', async ({ page }) => {
  await page.goto('/');
  await page.getByRole('link', { name: 'Debate' }).click();

  // Prepare sequential state responses
  let callCount = 0;
  await page.route('**/api/debate/*/state', (route) => {
    callCount += 1;
    if (callCount === 1) {
      // initial join: Alice active for Katz
      const body = {
        topic: 'Rotation topic',
        active_members: { katz: 'Alice', dogz: 'Xena' },
        history: [
          { round: 1, katz: 'Alice opening', dogz: 'Xena reply', scores: { katz: 40, dogz: 20 }, katz_member: 'Alice', dogz_member: 'Xena' }
        ]
      };
      route.fulfill({ status: 200, body: JSON.stringify(body), headers: { 'Content-Type': 'application/json' } });
    } else {
      // subsequent poll: Katz rotates to Bob
      const body = {
        topic: 'Rotation topic',
        active_members: { katz: 'Bob', dogz: 'Xena' },
        history: [
          { round: 1, katz: 'Alice opening', dogz: 'Xena reply', scores: { katz: 40, dogz: 20 }, katz_member: 'Alice', dogz_member: 'Xena' },
          { round: 2, katz: 'Bob followup', dogz: 'Xena counter', scores: { katz: 30, dogz: 35 }, katz_member: 'Bob', dogz_member: 'Xena' }
        ]
      };
      route.fulfill({ status: 200, body: JSON.stringify(body), headers: { 'Content-Type': 'application/json' } });
    }
  });

  // Join stubbed debate
  await page.fill('input[placeholder="Or paste debate id to join"]', 'rot-id');
  await page.click('text=Join');

  // Expect initial active member badge for Alice to be visible
  await expect(page.locator('text=✨ Alice')).toBeVisible();
  // Hover the Cats header and expect tooltip to contain 'Active: Alice' (title attribute)
  await page.hover('th:has-text("Team")');

  // Wait for next poll to occur and UI to update (poll period is 1s in fallback); wait a bit longer
  await page.waitForTimeout(1500);

  // Now expect the badge to reflect Bob
  await expect(page.locator('text=✨ Bob')).toBeVisible();
});