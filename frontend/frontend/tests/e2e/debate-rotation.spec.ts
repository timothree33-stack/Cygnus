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

  // Hover the Cats header and expect tooltip to contain an Active member (Alice or Bob)
  const teamCell = page.locator('td:has-text("Katz")');
  await teamCell.hover();
  await expect(teamCell.locator('[role="tooltip"]')).toHaveText(/Active: (Alice|Bob)/);

  // Wait for next poll to occur and UI to update (poll period is 1s in fallback); wait up to 3s for Bob to appear
  await page.waitForTimeout(200);
  await page.waitForFunction(() => {
    const td = Array.from(document.querySelectorAll('td')).find(n => n.textContent?.includes('Katz'));
    if(!td) return false;
    const t = td.querySelector('[role="tooltip"]');
    return !!(t && /Active: Bob/.test((t as HTMLElement).innerText));
  }, null, { timeout: 3000 });
  // final assertion: Bob should become visible
  await teamCell.hover();
  await expect(teamCell.locator('[role="tooltip"]')).toHaveText(/Active: Bob/);
});