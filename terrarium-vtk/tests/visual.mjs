/**
 * Visual test — takes a screenshot of the terrarium and checks for errors.
 * Run: node tests/visual.mjs
 */
import { chromium } from 'playwright';

async function test() {
  const browser = await chromium.launch({
    headless: false,
    args: ['--use-gl=angle', '--use-angle=swiftshader'],
  });
  const page = await browser.newPage({ viewport: { width: 1280, height: 800 } });

  const errors = [];
  page.on('console', msg => {
    if (msg.type() === 'error') errors.push(msg.text());
  });
  page.on('pageerror', err => errors.push(err.message));

  console.log('Loading page...');
  await page.goto('http://localhost:5173', { waitUntil: 'networkidle', timeout: 15000 });

  // Wait for WebSocket to connect and frames to render
  await page.waitForTimeout(8000);

  // Check if connected
  const statusText = await page.textContent('#status');
  console.log(`Connection status: ${statusText}`);

  // Take screenshot
  const path = 'tests/screenshot.png';
  await page.screenshot({ path });
  console.log(`Screenshot saved: ${path}`);

  // Check status text
  const status = await page.textContent('#status');
  console.log(`Status: ${status}`);

  // Check for WebGL errors
  const hasCanvas = await page.$('canvas');
  console.log(`Canvas present: ${!!hasCanvas}`);

  // Report errors
  if (errors.length > 0) {
    console.log('\n=== CONSOLE ERRORS ===');
    for (const e of errors) console.log(`  ${e}`);
  } else {
    console.log('No console errors');
  }

  await browser.close();
}

test().catch(e => { console.error(e); process.exit(1); });
