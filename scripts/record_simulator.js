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

    // Interaction loop
    console.log('Starting interaction simulation...');

    // 1. Move E-beam Dose slider
    console.log('Adjusting E-beam Dose...');
    await page.evaluate(() => {
        const slider = document.querySelector('input[type="range"]'); // First slider
        if (slider) {
            slider.scrollIntoView();
        }
    });

    // Simulate gradual change
    for (let i = 0; i <= 20; i++) {
        const val = 450 + (i * 10);
        await page.evaluate((v) => {
            const sliders = document.querySelectorAll('input[type="range"]');
            if (sliders[0]) {
                sliders[0].value = v;
                sliders[0].dispatchEvent(new Event('change', { bubbles: true }));
                sliders[0].dispatchEvent(new Event('input', { bubbles: true }));
            }
        }, val);
        await page.waitForTimeout(50);
    }

    await page.waitForTimeout(1000);

    // 2. Adjust Etching Time
    console.log('Adjusting Etching Time...');
    for (let i = 0; i <= 30; i++) {
        const val = 120 + i;
        await page.evaluate((v) => {
            const sliders = document.querySelectorAll('input[type="range"]');
            if (sliders[1]) {
                sliders[1].value = v;
                sliders[1].dispatchEvent(new Event('change', { bubbles: true }));
                sliders[1].dispatchEvent(new Event('input', { bubbles: true }));
            }
        }, val);
        await page.waitForTimeout(30);
    }

    await page.waitForTimeout(2000);

    // 3. Hover over risk indicators or something?
    // Just wait a bit to show the result
    console.log('Final pause...');
    await page.waitForTimeout(2000);

    await context.close();
    const videoPath = await page.video().path();
    console.log('Video recorded to:', videoPath);

    await browser.close();

    // Convert to GIF using ffmpeg
    const outputGif = path.join(__dirname, '..', 'docs', 'yield_predictor_demo.gif');
    console.log('Converting to GIF:', outputGif);

    try {
        // High quality GIF conversion
        execSync(`ffmpeg -y -i ${videoPath} -vf "fps=20,scale=1000:-1:flags=lanczos,split[s0][s1];[s0]palettegen[p];[s1][p]paletteuse" ${outputGif}`);
        console.log('Successfully generated GIF!');
    } catch (err) {
        console.error('Failed to convert video to GIF:', err.message);
    }
}

record().catch(console.error);
