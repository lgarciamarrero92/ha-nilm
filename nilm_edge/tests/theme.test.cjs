const { test } = require('node:test');
const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const vm = require('node:vm');

const source = fs.readFileSync(path.join(__dirname, '../www/js/theme.js'), 'utf8');

function setup({ systemDark = false, haDark, crossOrigin = false } = {}) {
    const root = { dataset: {} };
    const events = {};
    const media = { matches: systemDark, addEventListener: (_, fn) => { events.system = fn; } };
    const host = { hass: { themes: { darkMode: haDark } } };
    const parent = { document: { querySelector: () => host } };
    parent.parent = parent;
    if (crossOrigin) Object.defineProperty(parent, 'document', { get() { throw new Error('SecurityError'); } });
    const chart = { update: (mode) => { events.redraw = mode; } };
    const Chart = { instances: { chart }, register: (plugin) => { events.plugin = plugin; } };
    const window = {
        parent: haDark === undefined && !crossOrigin ? null : parent,
        matchMedia: () => media,
        setInterval: (fn) => { events.poll = fn; return 1; },
        clearInterval: () => {},
        addEventListener: (name, fn) => { events[name] = fn; },
        Chart,
    };
    window.parent ||= window;
    const document = { documentElement: root, addEventListener: (name, fn) => { events[name] = fn; } };
    vm.runInNewContext(source, { window, document, Chart });
    return { root, host, media, events };
}

test('HA preference takes priority over the system in both directions', () => {
    assert.equal(setup({ systemDark: false, haDark: true }).root.dataset.theme, 'dark');
    assert.equal(setup({ systemDark: true, haDark: false }).root.dataset.theme, 'light');
});

test('HA changes update open charts without a page reload', () => {
    const { root, host, events } = setup({ haDark: false });
    host.hass = { themes: { darkMode: true } };
    events.poll();
    assert.equal(root.dataset.theme, 'dark');
    assert.equal(events.redraw, 'none');
    delete events.redraw;
    events.poll();
    assert.equal(events.redraw, undefined);
    host.hass.themes.darkMode = false;
    events.poll();
    assert.equal(root.dataset.theme, 'light');
});

test('standalone and cross-origin views follow system changes', () => {
    for (const crossOrigin of [false, true]) {
        const { root, media, events } = setup({ crossOrigin });
        assert.equal(root.dataset.theme, 'light');
        media.matches = true;
        events.system();
        assert.equal(root.dataset.theme, 'dark');
    }
});

test('HA becoming available after iframe initialization overrides the fallback', () => {
    const { root, host, events } = setup({ haDark: null, systemDark: true });
    assert.equal(root.dataset.theme, 'dark');
    host.hass.themes.darkMode = false;
    events.poll();
    assert.equal(root.dataset.theme, 'light');
});

test('charts created after a theme change receive readable axes and legend colors', () => {
    const { root, events } = setup({ haDark: true });
    events.DOMContentLoaded();
    const chart = { options: {
        scales: { x: { ticks: {}, title: {}, grid: {} } },
        plugins: { legend: { labels: {} } },
    } };
    events.plugin.beforeUpdate(chart);
    assert.equal(chart.options.scales.x.ticks.color, '#cbd5e1');
    assert.equal(chart.options.plugins.legend.labels.color, '#cbd5e1');
    root.dataset.theme = 'light';
    events.plugin.beforeUpdate(chart);
    assert.equal(chart.options.scales.x.ticks.color, '#666666');
});
