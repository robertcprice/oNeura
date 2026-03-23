const path = require('path');
const { defineConfig } = require('@playwright/test');

const port = Number(process.env.ONEURA_WEB_SMOKE_PORT || 8447);
const repoRoot = path.resolve(__dirname, '../..');

module.exports = defineConfig({
  testDir: path.join(__dirname, 'tests'),
  timeout: 300_000,
  expect: {
    timeout: 30_000,
  },
  fullyParallel: false,
  workers: 1,
  reporter: 'list',
  use: {
    baseURL: `http://127.0.0.1:${port}`,
    browserName: 'webkit',
    headless: true,
    viewport: { width: 1440, height: 960 },
    acceptDownloads: true,
    trace: 'retain-on-failure',
    screenshot: 'only-on-failure',
    video: 'off',
  },
  webServer: process.env.ONEURA_WEB_SMOKE_SKIP_SERVER
    ? undefined
    : {
        command: `cargo run -p oneura-cli --bin terrarium_web --features web -- --port ${port}`,
        url: `http://127.0.0.1:${port}/`,
        cwd: repoRoot,
        reuseExistingServer: true,
        timeout: 300_000,
      },
  projects: [
    {
      name: 'webkit',
      use: {
        browserName: 'webkit',
      },
    },
  ],
});
