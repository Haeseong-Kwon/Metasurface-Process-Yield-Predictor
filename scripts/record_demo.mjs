import { chromium } from 'playwright';

(async () => {
    const browser = await chromium.launch({ headless: true });
    const context = await browser.newContext({
        recordVideo: {
            dir: './recordings',
            size: { width: 1280, height: 720 }
        }
    });
    const page = await context.newPage();

    console.log('Navigating to simulator...');
    await page.goto('http://localhost:3000/simulator');
    await page.waitForLoadState('networkidle');

    // Adjust E-beam Dose
    console.log('Adjusting E-beam Dose...');
    const sliders = await page.locator('input[type="range"]').all();
    await sliders[0].fill('600');
    await page.waitForTimeout(1000);

    // Adjust Etching Time
    console.log('Adjusting Etching Time...');
    await sliders[1].fill('220');
    await page.waitForTimeout(1000);

    // Click Predict Yield
    console.log('Clicking Predict Yield...');
    await page.click('button:has-text("Predict Yield")');

    // Wait for animation
    console.log('Waiting for animation to complete...');
    await page.waitForTimeout(8000);

    await context.close();
    await browser.close();
    console.log('Recording finished.');
})();
