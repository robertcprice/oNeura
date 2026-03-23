import { webkit } from 'playwright';
async function test() {
  const browser = await webkit.launch({ headless: true });
  const page = await browser.newPage({ viewport: { width: 1400, height: 900 } });
  page.on('pageerror', err => console.log(`[ERR] ${err.message.substring(0,150)}`));
  await page.goto('http://localhost:8420/?preset=terrarium', { waitUntil: 'load', timeout: 30000 });
  for (let i = 0; i < 15; i++) {
    await page.waitForTimeout(1000);
    const s = await page.evaluate(() => !document.body.innerText.includes('CONNECTING'));
    if (s) { console.log(`Connected at ${i}s`); break; }
  }
  const play = await page.$('#btnPlay');
  if (play) await play.click();
  await page.waitForTimeout(8000);
  await page.screenshot({ path: '/tmp/terrarium_water_fix.png' });
  console.log('Done');
  await browser.close();
}
test().catch(e => { console.error(e.message); process.exit(1); });
