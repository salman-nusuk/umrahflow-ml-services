import puppeteer from 'puppeteer';
import { mkdir } from 'fs/promises';
import { join, dirname } from 'path';
import { fileURLToPath } from 'url';

const __dirname = dirname(fileURLToPath(import.meta.url));
const SCREENSHOT_DIR = join(__dirname, '..', 'docs', 'screenshots');

const PAGES = [
  // L1 views
  { name: '01-l1-dashboard-light', path: '/', role: 'L1', theme: 'light', label: 'L1 Dashboard (Light)' },
  { name: '02-l1-dashboard-dark', path: '/', role: 'L1', theme: 'dark', label: 'L1 Dashboard (Dark)' },
  { name: '03-passport-queue', path: '/queue', role: 'L1', theme: 'light', label: 'Passport Queue' },
  { name: '04-ocr-workspace', path: '/ocr', role: 'L1', theme: 'light', label: 'OCR Workspace' },
  { name: '05-screening', path: '/screening', role: 'L1', theme: 'light', label: 'Screening' },
  { name: '06-voucher-list', path: '/vouchers', role: 'L1', theme: 'light', label: 'Voucher List' },
  { name: '07-voucher-detail', path: '/vouchers/v1', role: 'L1', theme: 'light', label: 'Voucher Detail' },
  { name: '08-nusuk-monitor', path: '/nusuk', role: 'L1', theme: 'light', label: 'Nusuk Monitor' },
  { name: '09-walkins', path: '/walkins', role: 'L1', theme: 'light', label: 'Walk-in Directory' },

  // L2 views
  { name: '10-l2-dashboard', path: '/', role: 'L2', theme: 'light', label: 'L2 Manager Dashboard' },
  { name: '11-ledger', path: '/ledger', role: 'L2', theme: 'light', label: 'Voucher Ledger' },
  { name: '12-team', path: '/team', role: 'L2', theme: 'light', label: 'Team Management' },
  { name: '13-ocr-accuracy', path: '/ocr-accuracy', role: 'L2', theme: 'light', label: 'OCR Accuracy' },
  { name: '14-blacklist', path: '/blacklist', role: 'L2', theme: 'light', label: 'Blacklist Management' },
  { name: '15-audit', path: '/audit', role: 'L2', theme: 'light', label: 'Audit Trail' },

  // L3 views
  { name: '16-l3-command-center', path: '/', role: 'L3', theme: 'light', label: 'L3 Executive Command Center' },
  { name: '17-l3-command-center-dark', path: '/', role: 'L3', theme: 'dark', label: 'L3 Command Center (Dark)' },
  { name: '18-revenue', path: '/revenue', role: 'L3', theme: 'light', label: 'Revenue Analytics' },
  { name: '19-compliance', path: '/compliance', role: 'L3', theme: 'light', label: 'Compliance Scorecard' },
  { name: '20-agents', path: '/agents', role: 'L3', theme: 'light', label: 'Agent Directory' },
  { name: '21-agent-detail', path: '/agents/a1', role: 'L3', theme: 'light', label: 'Agent Detail' },

  // Shared
  { name: '22-reports', path: '/reports', role: 'L2', theme: 'light', label: 'Reports' },
  { name: '23-settings', path: '/settings', role: 'L1', theme: 'light', label: 'Settings' },

  // Auth
  { name: '24-login', path: '/login', role: null, theme: 'light', label: 'Login' },
  { name: '25-forgot-password', path: '/forgot-password', role: null, theme: 'light', label: 'Forgot Password' },
];

async function captureScreenshots() {
  await mkdir(SCREENSHOT_DIR, { recursive: true });

  const browser = await puppeteer.launch({
    headless: true,
    defaultViewport: { width: 1440, height: 900 },
    args: ['--no-sandbox', '--disable-setuid-sandbox'],
  });

  const page = await browser.newPage();

  for (const config of PAGES) {
    console.log(`Capturing: ${config.label}...`);

    try {
      // Navigate to the page
      await page.goto(`http://localhost:3000${config.path}`, {
        waitUntil: 'networkidle2',
        timeout: 15000,
      });

      // Set theme if needed
      if (config.theme === 'dark') {
        await page.evaluate(() => {
          document.documentElement.classList.add('dark');
          localStorage.setItem('umrahflow-theme', 'dark');
        });
        await new Promise(r => setTimeout(r, 500));
      } else {
        await page.evaluate(() => {
          document.documentElement.classList.remove('dark');
          localStorage.setItem('umrahflow-theme', 'light');
        });
        await new Promise(r => setTimeout(r, 300));
      }

      // Set role if applicable (click the role button in the topbar)
      if (config.role) {
        try {
          const roleButtons = await page.$$('header button');
          for (const btn of roleButtons) {
            const text = await btn.evaluate(el => el.textContent?.trim());
            if (text === config.role) {
              await btn.click();
              await new Promise(r => setTimeout(r, 800));
              break;
            }
          }
        } catch (e) {
          // Role buttons may not exist on auth pages
        }
      }

      // Wait for content to settle
      await new Promise(r => setTimeout(r, 500));

      // Take screenshot
      await page.screenshot({
        path: join(SCREENSHOT_DIR, `${config.name}.png`),
        fullPage: false,
      });

      console.log(`  Done: ${config.name}.png`);
    } catch (err) {
      console.error(`  FAILED: ${config.name} - ${err.message}`);
    }
  }

  await browser.close();
  console.log(`\nAll screenshots saved to ${SCREENSHOT_DIR}`);
}

captureScreenshots().catch(console.error);
