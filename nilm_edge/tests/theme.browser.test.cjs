// Requires Playwright and an installed Chromium browser. Run with:
// node --test nilm_edge/tests/theme.browser.test.cjs
// Set NILM_BROWSER_CHANNEL=msedge to use the installed Windows Edge browser.
const { test } = require('node:test');
const assert = require('node:assert/strict');
const fs = require('node:fs');
const http = require('node:http');
const path = require('node:path');
const { chromium } = require('playwright');
const base = path.resolve(__dirname, '../www');

test('HA rendered theme switches an ingress iframe without exposing hass', async () => {
    const server = http.createServer((req, res) => {
        const url = new URL(req.url, 'http://localhost');
        res.setHeader('Content-Type', 'text/html');
        if (url.pathname === '/') {
            res.end(`<html style="--primary-background-color:#fafafa;background:var(--primary-background-color)">
                <head><meta name="color-scheme" content="light"></head>
                <body><home-assistant></home-assistant>
                <script>document.querySelector('home-assistant').attachShadow({mode:'open'}).innerHTML='<iframe src="/ingress/fixture"></iframe>'</script></body></html>`);
        } else if (url.pathname === '/ingress/fixture') {
            const component = fs.readFileSync(path.join(base, 'components/dashboard.html'), 'utf8').replace(/<script\b[^>]*>[\s\S]*?<\/script>/g, '');
            const training = fs.readFileSync(path.join(base, 'components/training.html'), 'utf8');
            const trainingStyles = training.match(/<style>[\s\S]*?<\/style>/)[0];
            // Exercise the production renderer, including dynamically created cards.
            const renderer = training.match(/function renderTrainingOverviewItem\([^]*?\n        }/)[0];
            const renderCard = new Function(`${renderer}; return renderTrainingOverviewItem;`)();
            const overview = ['Training Server', 'Appliance Name', 'Mains Power Sensor', 'Mains Range', 'Supervision', 'Labels']
                .map(label => renderCard(label, 'Selected value', 'Configuration details.')).join('');
            res.end(`<html><head><link rel="stylesheet" href="/theme.css"><script src="/theme.js"></script><script src="/chart.js"></script></head>
                <body>${component}${trainingStyles}<div id="trainingPrepareOverview">${overview}</div><canvas id="testCanvas"></canvas><script>
                document.addEventListener('DOMContentLoaded',()=>{window.chart=new Chart(document.querySelector('#testCanvas'),{type:'line',data:{labels:['A','B'],datasets:[{data:[1,2]}]},options:{animation:false}})})
                </script></body></html>`);
        } else {
            const files = { '/theme.js': ['js/theme.js', 'application/javascript'],
                '/theme.css': ['css/theme.css', 'text/css'], '/chart.js': ['vendor/js/chart.min.js', 'application/javascript'] };
            const file = files[url.pathname];
            if (!file) { res.writeHead(404); res.end(); return; }
            res.setHeader('Content-Type', file[1]);
            res.end(fs.readFileSync(path.join(base, file[0])));
        }
    });
    await new Promise(resolve => server.listen(0, '127.0.0.1', resolve));
    let browser;
    try {
        browser = await chromium.launch({ headless: true, ...(process.env.NILM_BROWSER_CHANNEL ? { channel: process.env.NILM_BROWSER_CHANNEL } : {}) });
        const page = await browser.newPage({ colorScheme: 'light' });
        page.setDefaultTimeout(5000);
        const errors = [];
        page.on('pageerror', error => errors.push(error.message));
        await page.goto(`http://127.0.0.1:${server.address().port}/`);
        const frame = page.frames().find(frame => frame.url().endsWith('/ingress/fixture'));
        await frame.waitForFunction(() => window.chart && document.documentElement.dataset.theme === 'light');
        const overviewCards = frame.locator('.training-overview-card');
        assert.equal(await overviewCards.count(), 6);
        assert.equal(await overviewCards.first().evaluate(el => getComputedStyle(el).backgroundColor), 'rgba(255, 255, 255, 0.9)');
        assert.equal(await frame.locator('.training-overview-value').first().evaluate(el => getComputedStyle(el).color), 'rgb(30, 41, 59)');
        await page.evaluate(() => {
            document.documentElement.style.setProperty('--primary-background-color', '#111111');
            document.querySelector('meta[name="color-scheme"]').content = 'dark';
        });
        await frame.waitForFunction(() => document.documentElement.dataset.theme === 'dark');
        assert.equal(await frame.locator('.field-input').first().evaluate(el => getComputedStyle(el).backgroundColor), 'rgb(30, 41, 59)');
        for (const card of await overviewCards.all()) {
            assert.equal(await card.evaluate(el => getComputedStyle(el).backgroundColor), 'rgba(30, 41, 59, 0.9)');
            assert.equal(await card.locator('.training-overview-value').evaluate(el => getComputedStyle(el).color), 'rgb(241, 245, 249)');
            assert.equal(await card.locator('.training-overview-label').evaluate(el => getComputedStyle(el).color), 'rgb(168, 181, 199)');
            assert.equal(await card.locator('.training-overview-help').evaluate(el => getComputedStyle(el).color), 'rgb(168, 181, 199)');
        }
        assert.equal(await frame.evaluate(() => window.chart.options.scales.x.ticks.color), '#cbd5e1');
        // Older HA layouts: no color-scheme metadata and no hass object.
        await page.evaluate(() => {
            document.querySelector('meta[name="color-scheme"]').remove();
            document.documentElement.style.setProperty('--primary-background-color', '#fafafa');
        });
        await frame.waitForFunction(() => document.documentElement.dataset.theme === 'light');
        assert.equal(await overviewCards.first().evaluate(el => getComputedStyle(el).backgroundColor), 'rgba(255, 255, 255, 0.9)');
        assert.equal(await frame.evaluate(() => window.chart.options.scales.x.ticks.color), '#666666');
        await page.evaluate(() => document.documentElement.style.setProperty('--primary-background-color', '#111111'));
        await frame.waitForFunction(() => document.documentElement.dataset.theme === 'dark');
        assert.deepEqual(errors, []);
    } finally {
        if (browser) await browser.close();
        await new Promise(resolve => server.close(resolve));
    }
});
