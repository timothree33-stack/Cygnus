import { defineConfig, devices } from '@playwright/test';

export default defineConfig({
  testDir: './tests/e2e',
  timeout: 30_000,
  expect: { timeout: 5000 },
  use: {
    baseURL: 'http://localhost:5173',
    headless: true,
    viewport: { width: 1280, height: 800 },
  },
const projects = [
  { name: 'chromium', use: { ...devices['Desktop Chrome'] } },
  { name: 'firefox', use: { ...devices['Desktop Firefox'] } },
];

// If PLAYWRIGHT_SKIP_WEBKIT is set to '1', omit WebKit (useful on runners without WebKit OS deps)
if (!process.env.PLAYWRIGHT_SKIP_WEBKIT || process.env.PLAYWRIGHT_SKIP_WEBKIT === '0') {
  projects.push({ name: 'webkit', use: { ...devices['Desktop Safari'] } });
}
  webServer: {
    // Use dev server if PLAYWRIGHT_USE_DEV_SERVER is set; otherwise build+preview for CI stability
    command: `sh -c "if [ \"$PLAYWRIGHT_USE_DEV_SERVER\" = \"1\" ]; then npm run dev; else npm run build && npm run preview -- --port 5173; fi"`,
    cwd: __dirname,
    url: 'http://localhost:5173',
    reuseExistingServer: true,
    timeout: 180_000,
  },
});
