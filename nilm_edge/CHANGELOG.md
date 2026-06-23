# Changelog

## 1.1.6

- Publish persistent per-appliance cumulative energy-consumed sensors in `kWh`, ready for the Home Assistant Energy dashboard.
- Preserve cumulative energy across add-on restarts and avoid integrating long stale-data gaps.
- Show each live model's native energy entity in its dashboard card.

## 1.1.5

- Added support for mains and appliance power sensors reported in `kW` by normalizing them to `W` across training and live disaggregation.
- Improved compatibility for existing configurations by resolving the mains sensor unit automatically when needed.

## 1.1.4

- Added support for external training servers, so training can run on another machine using a saved URL such as `http://<host>:<port>/train`.
- Added a `Custom External Server` option in the Training page alongside the autodetected internal Home Assistant training app.
- Improved training server selection so the internal app is selected by default when available, while custom mode only appears when chosen explicitly.
- Tightened training server validation and readiness checks to reject incomplete selections and invalid endpoints.
- Improved status messaging in the Training page so the active training server is shown clearly as internal or custom.
