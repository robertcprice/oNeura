const path = require('path');
const { test, expect } = require('@playwright/test');
const {
  exportJson,
  waitForLiveShell,
  waitForTransferIdle,
} = require('./shell-test-helpers.cjs');

test.describe.configure({ mode: 'serial' });

test('demo shell import/export flow stays operator-ready', async ({ page }) => {
  const transfers = page.locator('#grpTransfers');
  const chooser = page.locator('#bundleChoiceOverlay');
  const chooserCard = chooser.locator('.bundle-choice-card');
  const confirmOverlay = page.locator('#confirmOverlay');

  await waitForLiveShell(page);

  const bundle = await test.step('export full bundle from the transfer console', async () => {
    const exported = await exportJson(page, page.locator('[data-transfer-export-bundle]'), '/api/export/bundle', 'oneura-bundle');
    expect(exported.json?.simulation?.checkpoint).toBeTruthy();
    expect(exported.json?.simulation?.archive).toBeTruthy();
    expect(
      Array.isArray(exported.json?.simulation?.snapshot_history)
      || !!exported.json?.simulation?.snapshot
    ).toBeTruthy();
    await expect(transfers).toContainText('Full bundle exported');
    await expect(transfers).toContainText('Live backend export');
    await expect(transfers).toContainText('Payloads');
    await waitForTransferIdle(page);
    return exported;
  });

  const archive = await test.step('export archive json from the transfer console', async () => {
    const exported = await exportJson(page, page.locator('[data-transfer-export-archive]'), '/api/archive', 'oneura-archive');
    expect(exported.json?.snapshot).toBeTruthy();
    expect(exported.json?.organism_registry).toBeTruthy();
    await expect(transfers).toContainText('Archive exported');
    await waitForTransferIdle(page);
    return exported;
  });

  const checkpoint = await test.step('export restart-grade checkpoint json', async () => {
    const exported = await exportJson(page, page.locator('[data-transfer-export-checkpoint]'), '/api/checkpoint', 'oneura-checkpoint');
    expect(exported.json?.limitations?.fidelity).toBeTruthy();
    expect(exported.json?.snapshot).toBeTruthy();
    await expect(transfers).toContainText('Checkpoint exported');
    await waitForTransferIdle(page);
    return exported;
  });

  await test.step('load an exported archive into read-only inspection mode', async () => {
    await page.setInputFiles('#archiveImportInput', archive.filePath);
    await expect(transfers).toContainText('Archive inspect active', { timeout: 90_000 });
    await expect(page.locator('#grpWorld')).toContainText('Return Live');
    await expect(transfers).toContainText(path.basename(archive.filePath));
    await page.locator('[data-world-return-live]').click();
    await expect(transfers).toContainText('Live backend ready', { timeout: 30_000 });
    await waitForTransferIdle(page);
  });

  await test.step('route a full bundle into replay mode through the chooser', async () => {
    await page.setInputFiles('#bundleImportInput', bundle.filePath);
    await expect(chooser).toHaveAttribute('aria-hidden', 'false', { timeout: 30_000 });
    await expect(chooserCard).toContainText('Replay picker');
    await expect(chooserCard).toContainText('Payloads');
    await chooserCard.getByRole('button', { name: 'Load Replay', exact: true }).click();
    await expect(page.locator('#replaySourceBadge')).toContainText('Imported', { timeout: 30_000 });
    await expect(transfers).toContainText(path.basename(bundle.filePath));
    await waitForTransferIdle(page);
  });

  await test.step('route a full bundle into live restore through the checkpoint picker chooser', async () => {
    await page.setInputFiles('#checkpointImportInput', bundle.filePath);
    await expect(chooser).toHaveAttribute('aria-hidden', 'false', { timeout: 30_000 });
    await expect(chooserCard).toContainText('Checkpoint picker');
    await expect(chooserCard).toContainText('Payloads');
    await chooserCard.getByRole('button', { name: 'Restore Live World', exact: true }).click();
    await expect(confirmOverlay).toHaveAttribute('aria-hidden', 'false', { timeout: 30_000 });
    await expect(confirmOverlay).toContainText('Replace the live world from a saved checkpoint?');
    const restoreResponse = await Promise.all([
      page.waitForResponse(resp => {
        const url = new URL(resp.url());
        return url.pathname === '/api/import/checkpoint'
          && resp.request().method() === 'POST';
      }),
      confirmOverlay.locator('[data-confirm-choice="confirm"]').click(),
    ]).then(values => values[0]);
    expect(restoreResponse.ok()).toBeTruthy();
    const restored = await restoreResponse.json();
    expect(restored?.status).toBe('ok');
    await expect(page.locator('#replaySourceBadge')).toContainText(/backend/i, { timeout: 90_000 });
    await expect(page.locator('#sceneModeChip')).toContainText(/live/i, { timeout: 90_000 });
    await expect(transfers).toContainText(path.basename(bundle.filePath));
    await waitForTransferIdle(page);
  });
});
