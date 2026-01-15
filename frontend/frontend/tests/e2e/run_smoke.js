const { chromium } = require('playwright');

(async () => {
  const browser = await chromium.launch({ headless: true });
  const page = await browser.newPage();
  const BASE = 'http://127.0.0.1:5175';
  const PAGES = ['/', '/dashboard', '/agents', '/crawler', '/debate'];

  let failed = false;

  for (const path of PAGES) {
    try {
      const url = BASE + path;
      const resp = await page.goto(url, { waitUntil: 'domcontentloaded', timeout: 15000 });
      const status = resp ? resp.status() : 0;
      const title = await page.title();
      console.log(`${url} -> status=${status}, title="${title}"`);
      if (!(status === 200 || status === 304 || status === 0) ) {
        console.error(`Unexpected status for ${url}: ${status}`);
        failed = true;
      }
    } catch (err) {
      console.error(`Error visiting ${path}:`, err.message);
      failed = true;
    }
  }

  await browser.close();
  if (failed) process.exit(2);
  console.log('Smoke checks completed');
})();
