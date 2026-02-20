const { chromium } = require('playwright');
const path = require('path');
const fs = require('fs');
const { execSync } = require('child_process');

async function record() {
    console.log('Starting browser with forced CPU rendering...');
    const browser = await chromium.launch({
        args: [
            '--disable-gpu',
            '--disable-software-rasterizer',
            '--blink-settings=imagesEnabled=true',
            '--no-sandbox',
            '--disable-setuid-sandbox'
        ]
    });

    const context = await browser.newContext({
        viewport: { width: 1280, height: 800 },
        deviceScaleFactor: 2, // High resolution as requested
        recordVideo: {
            dir: './recordings',
            size: { width: 1280, height: 800 }
        }
    });

    const page = await context.newPage();

    console.log('Navigating to http://localhost:3000/simulator...');
    await page.goto('http://localhost:3000/simulator', { waitUntil: 'networkidle' });

    console.log('Wait for initial load...');
    await page.waitForTimeout(2000);

    // Interaction loop - 15 seconds approx
    console.log('Starting interaction simulation...');

    // 1. Move E-beam Dose (Parameter 1)
    console.log('Adjusting Parameter 1: E-beam Dose...');
    for (let i = 0; i <= 15; i++) {
        const val = 450 + (i * 10);
        await page.evaluate((v) => {
            const sliders = document.querySelectorAll('input[type="range"]');
            if (sliders[0]) {
                sliders[0].value = v;
                sliders[0].dispatchEvent(new Event('input', { bubbles: true }));
            }
        }, val);
        await page.waitForTimeout(100);
    }

    // 2. Adjust Etching Time (Parameter 2)
    console.log('Adjusting Parameter 2: Etching Time...');
    for (let i = 0; i <= 20; i++) {
        const val = 120 + (i * 2);
        await page.evaluate((v) => {
            const sliders = document.querySelectorAll('input[type="range"]');
            if (sliders[1]) {
                sliders[1].value = v;
                sliders[1].dispatchEvent(new Event('input', { bubbles: true }));
            }
        }, val);
        await page.waitForTimeout(80);
    }

    // 3. Adjust Gas Flow (Parameter 3)
    console.log('Adjusting Parameter 3: Gas Flow...');
    for (let i = 0; i <= 10; i++) {
        const val = 100 + (i * 5);
        await page.evaluate((v) => {
            const sliders = document.querySelectorAll('input[type="range"]');
            if (sliders[2]) {
                sliders[2].value = v;
                sliders[2].dispatchEvent(new Event('input', { bubbles: true }));
            }
        }, val);
        await page.waitForTimeout(100);
    }

    await page.waitForTimeout(500);

    // 4. Click Predict Yield
    console.log('Clicking "Predict Yield" button...');
    await page.click('button:has-text("Predict Yield")');

    // Wait for "Calculation" animation
    await page.waitForTimeout(1500);

    // 5. Final pause to show results
    console.log('Final pause...');
    await page.waitForTimeout(3000);

    await context.close();
    const videoPath = await page.video().path();
    console.log('Video recorded to:', videoPath);

    await browser.close();

    // Convert to GIF using ffmpeg (24 FPS, Optimized)
    const outputGif = path.join(__dirname, '..', 'docs', 'yield_predictor_demo.gif');
    console.log('Converting to GIF (24 FPS):', outputGif);

    try {
        // High quality GIF conversion with 24fps
        execSync(`ffmpeg -y -i ${videoPath} -vf "fps=24,scale=1000:-1:flags=lanczos,split[s0][s1];[s0]palettegen=max_colors=128[p];[s1][p]paletteuse=dither=bayer:bayer_scale=5" ${outputGif}`);
        console.log('Successfully generated GIF!');
    } catch (err) {
        console.error('Failed to convert video to GIF:', err.message);
    }
}

record().catch(console.error);
