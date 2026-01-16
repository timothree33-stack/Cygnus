import { test, expect } from '@playwright/test';

// This test stubs WebSocket messages to simulate a debate history with member breakdowns
test('debate: member panel opens and shows per-member details', async ({ page }) => {
  // Navigate to Debate page
  await page.goto('/');
  await page.getByRole('link', { name: 'Debate' }).click();

  // Stub WebSocket by intercepting the WS URL and sending a sample message after connection
  await page.route('**/ws/debates/**', (route) => {
    const ws = route.request();
    // Not all runtimes allow us to control raw ws; instead, stub fetch used by fallback polling
    route.continue();
  });

  // Instead we will inject a simulated history via fetch interception when the page calls /api/debate/:id/state
  const fakeHistory = {
    topic: 'Test topic',
    history: [
      { round: 1, cats_members: [{name:'C1', text:'C1 says', score:30},{name:'C2', text:'C2', score:10}], dogs_members: [{name:'D1', text:'D1 says', score:20}], scores: {cats: 40, dogs: 20} }
    ]
  };

  await page.route('**/api/debate/*/state', (route) => route.fulfill({ status: 200, body: JSON.stringify(fakeHistory), headers: { 'Content-Type': 'application/json' } }));

  // Start by entering the debate id and joining (any id will work as it's stubbed)
  await page.fill('input[placeholder="Or paste debate id to join"]', 'fake-id');
  await page.click('text=Join');

  // Open member panel; the page has a button 'Open Member Panel (latest round)'
  await page.click('text=Open Member Panel (latest round)');

  // Expect to see Cats and Dogs member details
  await expect(page.getByText('C1 says')).toBeVisible();
  await expect(page.getByText('D1 says')).toBeVisible();
});