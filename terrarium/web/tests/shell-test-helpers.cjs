const fs = require('fs/promises');
const os = require('os');
const path = require('path');
const { expect } = require('@playwright/test');

async function saveJsonText(raw, prefix) {
  const targetPath = path.join(
    os.tmpdir(),
    `${prefix}-${Date.now()}-${Math.random().toString(36).slice(2, 8)}.json`
  );
  await fs.writeFile(targetPath, raw, 'utf8');
  return {
    filePath: targetPath,
    fileName: path.basename(targetPath),
    json: JSON.parse(raw),
  };
}

async function exportJson(page, triggerLocator, endpointPath, prefix) {
  const [response] = await Promise.all([
    page.waitForResponse(resp => {
      const url = new URL(resp.url());
      return resp.ok() && url.pathname === endpointPath;
    }),
    triggerLocator.click(),
  ]);
  return saveJsonText(await response.text(), prefix);
}

async function waitForLiveShell(page) {
  await page.goto('/?renderer=canvas', { waitUntil: 'domcontentloaded' });
  await expect(page.locator('#grpTransfers')).toContainText('Save / Restore');
  await expect(page.locator('#sceneModeChip')).toContainText(/live/i, { timeout: 90_000 });
  await page.locator('#btnPause').click();
  await expect(page.locator('#grpTransfers')).toContainText(/Transfer idle|Transfer busy/);
}

async function waitForTransferIdle(page) {
  await expect(page.locator('#grpTransfers')).toContainText('Transfer idle', { timeout: 90_000 });
}

async function setRangeValue(page, selector, value) {
  await page.locator(selector).evaluate((element, nextValue) => {
    element.value = String(nextValue);
    element.dispatchEvent(new Event('input', { bubbles: true }));
    element.dispatchEvent(new Event('change', { bubbles: true }));
  }, value);
}

module.exports = {
  exportJson,
  saveJsonText,
  setRangeValue,
  waitForLiveShell,
  waitForTransferIdle,
};
