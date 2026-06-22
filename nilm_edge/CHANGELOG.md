# Changelog

## 1.1.5.5

- Publish native NILM energy states to six decimal `kWh` places while retaining full accumulator precision.

## 1.1.5.4

- Align native NILM energy entities with Home Assistant's integral helper (`state_class: total`) so Home Assistant displays their translated values with its normal energy precision.

## 1.1.5.3

- Suggest two decimal places for native NILM energy values in Home Assistant cards.

## 1.1.5.2

- Show each live model's native energy-consumed entity in its dashboard card.

## 1.1.5.1

- Publish persistent per-appliance cumulative energy sensors for use in the Home Assistant Energy dashboard.

## 1.1.5

- Added support for mains and appliance power sensors reported in `kW` by normalizing them to `W` across training and live disaggregation.
- Improved compatibility for existing configurations by resolving the mains sensor unit automatically when needed.

## 1.1.4

- Added support for external training servers, so training can run on another machine using a saved URL such as `http://<host>:<port>/train`.
- Added a `Custom External Server` option in the Training page alongside the autodetected internal Home Assistant training app.
- Improved training server selection so the internal app is selected by default when available, while custom mode only appears when chosen explicitly.
- Tightened training server validation and readiness checks to reject incomplete selections and invalid endpoints.
- Improved status messaging in the Training page so the active training server is shown clearly as internal or custom.
