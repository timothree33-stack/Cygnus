import { defineConfig, devices } from '@playwright/test';

const projects = [
  { name: 'chromium', use: { ...devices['Desktop Chrome'] } },
  { name: 'firefox', use: { ...devices['Desktop Firefox'] } },
];

// If PLAYWRIGHT_SKIP_WEBKIT is set to '1', omit WebKit (useful on runners without WebKit OS deps)
if (!process.env.PLAYWRIGHT_SKIP_WEBKIT || process.env.PLAYWRIGHT_SKIP_WEBKIT === '0') {
  projects.push({ name: 'webkit', use: { ...devices['Desktop Safari'] } });
}

export default defineConfig({
  // Tests live in nested `frontend/tests/e2e` when FRONTEND_DIR is `frontend`
  testDir: './frontend/tests/e2e',
  timeout: 120000,
  expect: { timeout: 10000 },
  grep: process.env.FULL_STACK ? /@fullstack/ : undefined,
  fullyParallel: true,
  forbidOnly: !!process.env.CI,
  retries: process.env.CI ? 1 : 0,
  reporter: 'list',
  use: {
    baseURL: 'http://127.0.0.1:5173',
    headless: true,
    viewport: { width: 1280, height: 720 },
    actionTimeout: 5000,
    trace: 'on-first-retry'
  },
  webServer: {
    // Use dev server if PLAYWRIGHT_USE_DEV_SERVER is set; otherwise build+preview for CI stability
    // Run commands in nested frontend folder when present and sanitize the input to avoid shell errors
    command: `bash -lc "cd frontend 2>/dev/null || true; v=$(printf '%s' \"$PLAYWRIGHT_USE_DEV_SERVER\" | tr -d '\\n' | cut -d' ' -f1); if [ \"x$v\" = \"x1\" ]; then npm run dev; else npm run build && npm run preview -- --port 5173; fi"`,
    port: 5173,
    cwd: __dirname, // run dev in the config folder (frontend)
    reuseExistingServer: !process.env.CI,
    timeout: 180_000,
  },
  projects
});