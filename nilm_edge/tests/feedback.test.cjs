const { test } = require('node:test');
const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const vm = require('node:vm');

const html = fs.readFileSync(path.join(__dirname, '../www/components/dashboard.html'), 'utf8');
const source = html.match(/<script>([\s\S]*?)<\/script>/)[1];

function setup() {
    const card = { hidden: true };
    const context = vm.createContext({
        window: { nilmFeedbackLastPromptAt: Date.now() }, console,
        localStorage: {
            getItem() { throw new Error('Feedback should not read storage'); },
            setItem() { throw new Error('Feedback should not write storage'); },
        },
    });
    vm.runInContext(source, context);
    context.card = card;
    vm.runInContext(`
        dashboardFeedback = card;
        signalPredictionPanel = { classList: { add() {}, remove() {} } };
        signalPredictionMeta = { style: {} };
        signalPredictionList = {};
        renderInlinePredictionShareChart = () => {};
    `, context);
    return {
        card,
        render: (predictions = []) => {
            context.predictions = predictions;
            vm.runInContext('activeInlinePredictions = predictions; renderInlinePredictionUi();', context);
        },
    };
}

const results = [{ modelKey: 'test', modelName: 'Test', powerSeries: [{ x: 1, y: 0 }] }];

test('feedback appears only with nonempty results, including zero power', () => {
    const app = setup();
    app.render();
    assert.equal(app.card.hidden, true);
    app.render([{ powerSeries: [], stateSeries: [] }]);
    assert.equal(app.card.hidden, true);
    app.render(results);
    assert.equal(app.card.hidden, false);
    app.render(results);
    assert.equal(app.card.hidden, false);
});

test('feedback appears every time results are displayed and on later Dashboard visits', () => {
    const app = setup();
    app.render(results);
    assert.equal(app.card.hidden, false);
    app.render();
    assert.equal(app.card.hidden, true);
    app.render(results);
    assert.equal(app.card.hidden, false);
    const nextVisit = setup();
    nextVisit.render(results);
    assert.equal(nextVisit.card.hidden, false);
});

test('state-only results also show feedback', () => {
    const app = setup();
    app.render([{ stateSeries: [{ x: 1, y: 0 }] }]);
    assert.equal(app.card.hidden, false);
});

test('feedback keeps the requested copy and safe new-tab link without dismissal', () => {
    assert.match(html, /Help improve HA-NILM<\/h2>/);
    assert.match(html, /Does the disaggregation match what actually happened in your home\? Your feedback helps improve HA-NILM\. It takes just 1 minute\./);
    assert.match(html, /href="https:\/\/ha-nilm\.bigwicho\.com\/feedback\?source=ha" target="_blank" rel="noopener noreferrer">Share feedback<\/a>/);
    assert.doesNotMatch(html, /dashboardFeedbackDismiss|dismissDashboardFeedback|nilm_feedback_last_prompt_at_v1/);
});
