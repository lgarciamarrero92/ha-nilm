const { test } = require('node:test');
const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const vm = require('node:vm');

const source = fs.readFileSync(path.join(__dirname, '../www/js/theme.js'), 'utf8');

function setup({ systemDark = false, haDark, crossOrigin = false, embedded = haDark !== undefined || crossOrigin,
    exposeHass = true, scheme, background = '', renderedBackground = 'rgba(0, 0, 0, 0)' } = {}) {
    const root = { dataset: {} };
    const events = {};
    const media = { matches: systemDark, addEventListener: (_, fn) => { events.system = fn; } };
    const host = { hass: { themes: { darkMode: haDark } } };
    const parentRoot = { scheme, background, renderedBackground };
    const parent = { document: {
        documentElement: parentRoot,
        querySelector: (selector) => {
            if (selector === 'home-assistant') return exposeHass ? host : null;
            if (selector === 'meta[name="color-scheme"]') return { getAttribute: () => parentRoot.scheme };
            return null;
        },
    }, getComputedStyle: () => ({
        getPropertyValue: () => parentRoot.background,
        backgroundColor: parentRoot.renderedBackground,
    }) };
    parent.parent = parent;
    if (crossOrigin) Object.defineProperty(parent, 'document', { get() { throw new Error('SecurityError'); } });
    const chart = { update: (mode) => { events.redraw = mode; } };
    const Chart = { instances: { chart }, register: (plugin) => { events.plugin = plugin; } };
    const window = {
        parent: embedded ? parent : null,
        matchMedia: () => media,
        setInterval: (fn) => { events.poll = fn; return 1; },
        clearInterval: () => {},
        addEventListener: (name, fn) => { events[name] = fn; },
        Chart,
    };
    window.parent ||= window;
    const document = { documentElement: root, addEventListener: (name, fn) => { events[name] = fn; } };
    vm.runInNewContext(source, { window, document, Chart });
    return { root, host, media, events, parentRoot, parent, window };
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

test('rendered HA metadata follows profile changes even when hass is unavailable', () => {
    const { root, events, parentRoot } = setup({ embedded: true, exposeHass: false,
        scheme: 'light', background: '#fafafa', renderedBackground: 'rgb(250, 250, 250)' });
    assert.equal(root.dataset.theme, 'light');
    parentRoot.scheme = 'dark';
    parentRoot.background = '#111111';
    parentRoot.renderedBackground = 'rgb(17, 17, 17)';
    events.poll();
    assert.equal(root.dataset.theme, 'dark');
    parentRoot.scheme = 'light';
    events.poll();
    assert.equal(root.dataset.theme, 'light');
});

test('older HA layouts without scheme metadata follow the rendered background', () => {
    const { root, events, parentRoot } = setup({ embedded: true, exposeHass: false,
        background: '#fafafa', renderedBackground: 'rgb(250, 250, 250)' });
    assert.equal(root.dataset.theme, 'light');
    parentRoot.background = '#111111';
    parentRoot.renderedBackground = 'rgb(17, 17, 17)';
    events.poll();
    assert.equal(root.dataset.theme, 'dark');
    parentRoot.background = '#fafafa';
    parentRoot.renderedBackground = 'rgb(250, 250, 250)';
    events.poll();
    assert.equal(root.dataset.theme, 'light');
});

test('rendered metadata takes priority over stale HA state', () => {
    assert.equal(setup({ haDark: false, scheme: 'dark' }).root.dataset.theme, 'dark');
    assert.equal(setup({ haDark: true, scheme: 'light' }).root.dataset.theme, 'light');
});

test('HA background variables work when older layouts leave the root transparent', () => {
    assert.equal(setup({ embedded: true, exposeHass: false, background: '#111' }).root.dataset.theme, 'dark');
    assert.equal(setup({ embedded: true, exposeHass: false, background: '#fafafa' }).root.dataset.theme, 'light');
    assert.equal(setup({ embedded: true, exposeHass: false, background: 'rgb(17, 17, 17)' }).root.dataset.theme, 'dark');
});

test('custom theme metadata supporting both schemes uses the effective HA mode', () => {
    assert.equal(setup({ haDark: true, scheme: 'dark light' }).root.dataset.theme, 'dark');
    assert.equal(setup({ haDark: false, scheme: 'dark light' }).root.dataset.theme, 'light');
});

test('nested ingress frames still find the containing HA frontend', () => {
    const { root, events, parent, window } = setup({ haDark: true });
    window.parent = { parent, document: { documentElement: {}, querySelector: () => null },
        getComputedStyle: () => ({ getPropertyValue: () => '' }) };
    root.dataset.theme = 'light';
    events.poll();
    assert.equal(root.dataset.theme, 'dark');
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
