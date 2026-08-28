# Home Assistant NILM App

![NILM for Home Assistant logo](https://github.com/lgarciamarrero92/ha-nilm/raw/main/nilm_edge/logo.png)

This app runs real-time NILM inference inside Home Assistant.

It monitors one or more mains power sensors, applies trained appliance models, and publishes live appliance power, cumulative energy-consumed, and on/off entities back to Home Assistant.

NILM provides estimation from aggregate mains data, not direct per-appliance measurement.

## Main Features

- Live disaggregation from one aggregate mains sensor.
- Built-in UI for setup, model management, and preview.
- Training job preparation and handoff to the separate `NILM Training Server` app (or compatible external training server).
- Training server selection supports either the local Home Assistant training app or a remote `nilm_trainer` URL on another machine.
- Per-model live publish toggle.

## Quick Start

1. Install both apps: `NILM` and `NILM Training Server`.
2. Start `NILM Training Server` first, then start `NILM`.
3. Open the `NILM` UI and save your mains power sensor.
4. Open the Training page and confirm the training server is ready.
5. If the trainer runs on another machine, choose `Custom External Server` and save its URL.

## Core Workflow

1. Select and save the mains sensor in `NILM`.
2. In Training, choose manual interval labeling or sensor-based labeling.
3. Prepare training data from Home Assistant history.
4. Send the job to the training server.
5. When training finishes, validate predictions in the NILM Dashboard.
6. Enable live publishing for selected models.
7. Use generated entities in dashboards and automations.

Notes:
- Training range is limited to the previous 7 days.
- Better training quality comes from complete labeling of the chosen interval.
- Live entities are updated approximately every 8 seconds.

## Multiple Phase

HA-NILM supports one or more mains power sensors, making it suitable for split-phase and multi-phase electrical supplies. Train each appliance model using the mains sensor that measures its circuit. In the NILM Dashboard, you can view each configured mains sensor separately or view their aggregate, calculated as the sum of all declared mains sensors.

Each appliance model name must be unique within its model bundle, including models assigned to different mains sensors. Use names such as `fridge_kitchen` and `fridge_garage`; names that differ only by capitalization, spaces, or punctuation can resolve to the same internal name and replace an existing model.

## Published Entities

For each model enabled for live publishing:

- `sensor.nilm_<appliance>_power`
- `sensor.nilm_<appliance>_energy_consumed` (cumulative `kWh`, compatible with the Home Assistant Energy dashboard)
- `binary_sensor.nilm_<appliance>_on`

## Requirements

- Home Assistant app environment.
- A mains power sensor already available in Home Assistant.
- Recorder history for the selected training period.
- Either both NILM apps installed on Home Assistant, or `NILM` plus a reachable remote `nilm_trainer` server URL.
- Around 4 GB RAM available for Home Assistant + NILM apps.
- Mains updates in the order of seconds (1s, 3s, 5s, 10s typically work well).

## Full Documentation

This README only covers the essentials.

For complete setup, training, troubleshooting, and advanced guidance, visit:

- https://ha-nilm.bigwicho.com/
