/* Home Assistant ingress shares its origin with the containing frontend.
 * Read the effective user theme, including HA's automatic mode. No API token
 * is needed, and a separately opened UI falls back to the system preference.
 */
(() => {
    const root = document.documentElement;
    const systemTheme = window.matchMedia('(prefers-color-scheme: dark)');

    function homeAssistantDarkMode() {
        let frame = window;
        while (frame.parent !== frame) {
            try {
                frame = frame.parent;
                const hass = frame.document.querySelector('home-assistant')?.hass;
                if (typeof hass?.themes?.darkMode === 'boolean') {
                    return hass.themes.darkMode;
                }
            } catch (_) {
                // Cross-origin embeddings cannot expose their user settings.
                break;
            }
        }
        return systemTheme.matches;
    }

    function syncTheme() {
        const theme = homeAssistantDarkMode() ? 'dark' : 'light';
        if (root.dataset.theme === theme) return;
        root.dataset.theme = theme;
        // Chart.js canvases do not inherit CSS colors; redraw existing charts.
        if (window.Chart) {
            Object.values(Chart.instances).forEach((chart) => chart.update('none'));
        }
    }

    syncTheme();
    systemTheme.addEventListener('change', syncTheme);
    // HA replaces hass as settings change, without a DOM mutation or a theme
    // property in the ingress postMessage protocol. Poll only the small flag.
    let timer = window.setInterval(syncTheme, 500);
    window.addEventListener('pagehide', () => window.clearInterval(timer));
    window.addEventListener('pageshow', (event) => {
        if (event.persisted) {
            syncTheme();
            timer = window.setInterval(syncTheme, 500);
        }
    });

    document.addEventListener('DOMContentLoaded', () => {
        if (!window.Chart) return;
        Chart.register({
            id: 'nilmTheme',
            beforeUpdate(chart) {
                const dark = root.dataset.theme === 'dark';
                const text = dark ? '#cbd5e1' : '#666666';
                const grid = dark ? 'rgba(148, 163, 184, 0.18)' : 'rgba(0, 0, 0, 0.1)';
                chart.options.color = text;
                for (const scale of Object.values(chart.options.scales || {})) {
                    scale.ticks.color = text;
                    scale.title.color = text;
                    scale.grid.color = grid;
                    scale.grid.borderColor = grid;
                }
                const legend = chart.options.plugins.legend;
                if (legend) legend.labels.color = text;
            },
        });
    });
})();
