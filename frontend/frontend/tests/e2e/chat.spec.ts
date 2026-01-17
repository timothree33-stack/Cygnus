import { test, expect } from '@playwright/test';

test('chat: chat window for cygnus loads and can send', async ({ page, request }) => {
  await page.goto('/');
  await page.getByRole('link', { name: 'Chat' }).click();
  await expect(page.getByText('Chat with Cygnus')).toBeVisible();
  // Type a message and send
  await page.fill('input[placeholder="Say something..."]', 'Hello Cygnus');
  await page.click('text=Send');
  // Message should appear in message list
  await expect(page.getByText('human: Hello Cygnus')).toBeVisible();
});
