const { test, expect } = require('@playwright/test');
const {
  exportJson,
  setRangeValue,
  waitForLiveShell,
  waitForTransferIdle,
} = require('./shell-test-helpers.cjs');

test.describe.configure({ mode: 'serial' });

test('demo shell preserves operator state and protects world replacement flows', async ({ page, context }) => {
  const confirmOverlay = page.locator('#confirmOverlay');
  const shortcutOverlay = page.locator('#shortcutOverlay');
  const commandOverlay = page.locator('#commandOverlay');
  const chooserOverlay = page.locator('#bundleChoiceOverlay');
  const chooserCard = chooserOverlay.locator('.bundle-choice-card');

  await waitForLiveShell(page);

  await test.step('persist layout state into local storage', async () => {
    await page.locator('#surfaceModeBtns button[data-surface-mode="chemistry"]').click();
    await page.keyboard.press('Digit5');
    await setRangeValue(page, '#fpsSlider', 7);
    await setRangeValue(page, '#replaySpeedSlider', 2);
    await page.locator('.tabs button[data-tab="learn"]').click();

    await expect(page.locator('#surfaceModeBtns button[data-surface-mode="chemistry"]')).toHaveClass(/active/);
    await expect(page.locator('body')).toHaveAttribute('data-surface-mode', 'chemistry');
    await expect(page.locator('.tabs button[data-tab="learn"]')).toHaveClass(/active/);
    await expect(page.locator('#viewBtns button[data-view="chemistry"]')).toHaveClass(/active/);
    await expect(page.locator('#toolbar button[data-tool="fruit"]')).toHaveClass(/active/);
    await expect(page.locator('#fpsDisplay')).toHaveText('7');
    await expect(page.locator('#replaySpeedLabel')).toHaveText('2.0x');

    const stored = await page.evaluate(() => JSON.parse(localStorage.getItem('oneura:terrarium-ui-prefs:v1') || '{}'));
    expect(stored).toMatchObject({
      surfaceMode: 'chemistry',
      mainTab: 'learn',
      statsMode: 'chemistry',
      view: 'chemistry',
      tool: 'fruit',
      fps: 7,
      replayRate: 2,
    });
  });

  await test.step('keyboard overlays open and close cleanly', async () => {
    await page.keyboard.press('H');
    await expect(shortcutOverlay).toHaveAttribute('aria-hidden', 'false');
    await page.keyboard.press('Escape');
    await expect(shortcutOverlay).toHaveAttribute('aria-hidden', 'true');

    await page.keyboard.press('/');
    await expect(commandOverlay).toHaveAttribute('aria-hidden', 'false');
    await page.locator('#commandInput').fill('share');
    await expect(page.locator('#commandResults')).toContainText('Copy Share Link');
    await page.keyboard.press('Escape');
    await expect(commandOverlay).toHaveAttribute('aria-hidden', 'true');
  });

  await test.step('a new page in the same browser context restores the saved layout', async () => {
    const restoredPage = await context.newPage();
    await waitForLiveShell(restoredPage);
    await expect(restoredPage.locator('body')).toHaveAttribute('data-surface-mode', 'chemistry');
    await expect(restoredPage.locator('.tabs button[data-tab="learn"]')).toHaveClass(/active/);
    await expect(restoredPage.locator('#viewBtns button[data-view="chemistry"]')).toHaveClass(/active/);
    await expect(restoredPage.locator('#toolbar button[data-tool="fruit"]')).toHaveClass(/active/);
    await expect(restoredPage.locator('#fpsDisplay')).toHaveText('7');
    await expect(restoredPage.locator('#replaySpeedLabel')).toHaveText('2.0x');
    await restoredPage.locator('.tabs button[data-tab="stats"]').click();
    await expect(restoredPage.locator('#tabStats')).toHaveAttribute('data-stats-mode', 'chemistry');
    await restoredPage.close();
  });

  await page.locator('.tabs button[data-tab="stats"]').click();
  await page.getByRole('button', { name: 'Overview' }).click();

  const bundle = await test.step('export a full bundle for restore and archive-routing tests', async () => {
    const exported = await exportJson(
      page,
      page.locator('[data-transfer-export-bundle]'),
      '/api/export/bundle',
      'oneura-bundle-ux'
    );
    expect(exported.json?.simulation?.checkpoint).toBeTruthy();
    expect(exported.json?.simulation?.archive).toBeTruthy();
    await expect(page.locator('#grpTransfers')).toContainText('Full bundle exported');
    await waitForTransferIdle(page);
    return exported;
  });

  await test.step('bundle restores require chooser selection and an explicit confirmation', async () => {
    await page.setInputFiles('#checkpointImportInput', bundle.filePath);
    await expect(chooserOverlay).toHaveAttribute('aria-hidden', 'false', { timeout: 30_000 });
    await expect(chooserCard).toContainText('Checkpoint picker');
    await chooserCard.getByRole('button', { name: 'Restore Live World', exact: true }).click();
    await expect(confirmOverlay).toHaveAttribute('aria-hidden', 'false', { timeout: 30_000 });
    await expect(confirmOverlay).toContainText('Replace the live world from a saved checkpoint?');
    await expect(confirmOverlay).toContainText('Incoming Preset');
    await confirmOverlay.getByRole('button', { name: 'Cancel restore', exact: true }).click();
    await expect(confirmOverlay).toHaveAttribute('aria-hidden', 'true', { timeout: 30_000 });
    await waitForTransferIdle(page);
  });

  await test.step('the same bundle can route into read-only archive inspection mode', async () => {
    await page.setInputFiles('#bundleImportInput', bundle.filePath);
    await expect(chooserOverlay).toHaveAttribute('aria-hidden', 'false', { timeout: 30_000 });
    await expect(chooserCard).toContainText('Replay picker');
    await chooserCard.getByRole('button', { name: 'Inspect Archive', exact: true }).click();
    await expect(page.locator('#grpTransfers')).toContainText('Archive inspect active', { timeout: 90_000 });
    await expect(page.locator('#grpWorld')).toContainText('Return Live');
  });

  await test.step('archive inspect mode confirms world resets and records recents/share state', async () => {
    await page.locator('[data-demo-reset="aquarium:42"]').click();
    await expect(confirmOverlay).toHaveAttribute('aria-hidden', 'false', { timeout: 30_000 });
    await expect(confirmOverlay).toContainText('Reset the current simulation state?');
    await expect(confirmOverlay).toContainText('Next Preset');
    await confirmOverlay.locator('[data-confirm-choice="confirm"]').click();
    await expect(confirmOverlay).toHaveAttribute('aria-hidden', 'true', { timeout: 30_000 });
    await expect(page.locator('#sceneModeChip')).toContainText(/live/i, { timeout: 90_000 });
    await expect(page.locator('#grpGuide')).toContainText('aquarium · seed 42', { timeout: 30_000 });
    await waitForTransferIdle(page);

    const recentWorlds = await page.evaluate(() => JSON.parse(localStorage.getItem('oneura:terrarium-recent-worlds:v1') || '[]'));
    expect(recentWorlds[0]).toMatchObject({
      preset: 'aquarium',
      seed: 42,
    });

    const shareUrl = await page.evaluate(() => buildShareUrl());
    expect(shareUrl).toContain('preset=aquarium');
    expect(shareUrl).toContain('seed=42');
    expect(shareUrl).toContain('surface=chemistry');
    expect(shareUrl).toContain('view=chemistry');
    expect(shareUrl).toContain('tool=fruit');
    expect(shareUrl).toContain('stats=overview');
    expect(shareUrl).toContain('fps=7');
    expect(shareUrl).toContain('replayRate=2');
  });
});
