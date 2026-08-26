# Changelog

## 1.1.9

- Added support for configuring multiple mains power sensors and assigning appliance models to the correct mains source.
- Preserved compatibility with existing single-mains installations while automatically migrating legacy models to the primary mains sensor.
- Added unassigned-model handling so models remain available after their mains sensor is deleted and can be reassigned without retraining.
- Updated live disaggregation to run each appliance model only against its assigned mains power sensor.
- Updated offline preview and aggregated mains preview to group predictions by mains sensor and calculate base-load energy per mains before summing totals.
- Updated training so new appliance models are saved with the selected mains power sensor assignment.
- Improved configuration normalization so stale model assignments to removed mains sensors are saved as unassigned.
- Improved mains management in the dashboard with add, edit, and delete modals, required display names, clearer validation errors, and refreshed sensor selections after changes.
- Refined the appliance model dashboard layout, model counts, unassigned-model presentation, and disaggregation controls for clearer multi-mains workflows.

## 1.1.8

- Added a configurable `sensor_max_gap_s` add-on option for mains sensors with slower update cadences, defaulting to 60 seconds.
- Applied the configured sensor gap consistently across live disaggregation, training preparation, sensor-derived training intervals, and offline prediction previews.

## 1.1.7

- Added an optional ON/OFF helper for interval supervision, allowing users to select a binary sensor and automatically prefill editable ON intervals from its history.
- Added a binary-sensor discovery endpoint and filtered out HA-NILM virtual binary sensors from helper choices.
- Improved interval supervision controls with a clear-all action in the Selected Windows Preview section and clearer success/status modals after helper intervals are created.
- Fixed published Home Assistant entity IDs for appliance names with non-English letters by transliterating names into ASCII-safe slugs while preserving the original friendly name.

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
