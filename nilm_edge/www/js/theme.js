/* Read the containing Home Assistant frontend's rendered theme. Ingress
 * shares its origin, so this needs no credentials or backend theme settings.
 * A separately opened UI follows the browser's color scheme.
 */
(() => {
    const root = document.documentElement;
    const systemTheme = window.matchMedia('(prefers-color-scheme: dark)');

    function renderedHomeAssistantDarkMode(frame) {
        const parentDocument = frame.document;
        const host = parentDocument.querySelector('home-assistant');
        const styles = frame.getComputedStyle(parentDocument.documentElement);
        const background = styles.getPropertyValue('--primary-background-color').trim();
        // Ignore ordinary containing pages; these signals belong to HA.
        if (!host && !background) return undefined;

        const scheme = parentDocument.querySelector('meta[name="color-scheme"]')
            ?.getAttribute('content')?.trim();
        // HA updates this metadata with the effective default-theme mode.
        if (scheme === 'dark') return true;
        if (scheme === 'light') return false;

        const darkMode = host?.hass?.themes?.darkMode;
        if (typeof darkMode === 'boolean') return darkMode;

        // HA also applies its theme to the document root. Read that rendered
        // background when hass is absent (e.g. a different frontend layout).
        // Older frontends may paint the body instead of the document root.
        // Their resolved background variable still identifies the theme.
        let rgb = (styles.backgroundColor || '').match(/^rgba?\(\s*(\d+)[, ]+\s*(\d+)[, ]+\s*(\d+)(?:\s*[,/]\s*([\d.]+))?\s*\)$/);
        if (!rgb || Number(rgb[4]) === 0) {
            const hex = background.match(/^#([\da-f]{3}|[\da-f]{6})$/i);
            if (hex) {
                const value = hex[1].length === 3 ? [...hex[1]].map(digit => digit + digit).join('') : hex[1];
                rgb = ['', ...value.match(/../g).map(channel => parseInt(channel, 16))];
            } else {
                rgb = background.match(/^rgb\(\s*(\d+)[, ]+\s*(\d+)[, ]+\s*(\d+)\s*\)$/);
            }
        }
        if (background && rgb && (rgb[4] === undefined || Number(rgb[4]) > 0)) {
            const brightness = 0.2126 * Number(rgb[1]) + 0.7152 * Number(rgb[2]) + 0.0722 * Number(rgb[3]);
            return brightness < 128;
        }
        return undefined;
    }

    function homeAssistantDarkMode() {
        let frame = window;
        while (frame.parent !== frame) {
            try {
                frame = frame.parent;
                const darkMode = renderedHomeAssistantDarkMode(frame);
                if (typeof darkMode === 'boolean') return darkMode;
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
    // The ingress properties protocol does not include theme changes. Poll
    // the effective frontend theme, including after HA replaces its state.
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
