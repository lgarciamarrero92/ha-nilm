<p align="center">
  <img src="nilm_edge/logo.png" alt="NILM for Home Assistant logo" width="240">
</p>

<h1 align="center">Non-Intrusive Load Monitoring (NILM) For Home Assistant</h1>

<h2 align="center">🔌 Turn your whole-home power sensor into appliance activity and energy insight</h2>

HA-NILM learns your appliance patterns during a short guided training phase, then uses them to estimate real-time appliance activity and energy usage from your whole-home power meter, locally in Home Assistant.

<p align="center">
  <img src="nilm_edge/www/ha-nilm.png" alt="NILM for Home Assistant interface preview" width="640">
</p>

This repository contains two NILM services that work together:

- `NILM` (inference service)
- `NILM Training Server` (training service)

NILM estimates appliance behavior from one aggregate mains power sensor. It provides useful estimation, not direct per-appliance metering.

You can run the training server either:

- as the `NILM Training Server` app inside Home Assistant
- or as the `ha-nilm-trainer` Docker container reachable from `NILM`

## Why HA-NILM

- Estimate appliance activity and energy usage from one mains sensor instead of metering every device.
- Learn appliance patterns from your Home Assistant history.
- Review estimated appliance activity, consumption, and energy share over time.
- Publish live estimated entities for dashboards, automations, and energy workflows.

## Before You Begin

- A working Home Assistant OS or Home Assistant Container installation.
- A mains power sensor already available in Home Assistant.
- Recorder history for the time range you want to train.
- Either both services installed as Home Assistant OS apps, the `NILM` Home Assistant OS app plus a reachable trainer container, or both services running with Docker Compose for Home Assistant Container.
- At least 4 GB RAM for Home Assistant and the NILM services.

## What Each Service Does

### NILM

- Monitors one mains power sensor in Home Assistant.
- Runs live NILM inference.
- Publishes estimated appliance entities (`power`, cumulative `energy consumed`, and `on/off`) for dashboards, automations, and the Energy dashboard.
- Provides UI for setup, preview, and training job preparation.

### NILM Training Server

- Receives prepared training jobs from NILM.
- Runs model training in the background.
- Returns trained embeddings and learned thresholds back to NILM.

This training service can run:

- as the Home Assistant `NILM Training Server` app
- or as a Docker container reachable over your network

## Quick Start: Home Assistant OS

1. Open Home Assistant.
2. Go to `Settings` > `Apps` > `Install App`.
3. Add this repository: `https://github.com/lgarciamarrero92/ha-nilm`.
4. Install `NILM`.
5. Choose one training server option:
   - install `NILM Training Server` in Home Assistant
   - or run the `ha-nilm-trainer` Docker container on a reachable Docker host
6. Start the training server first.
7. Start `NILM`.
8. Open the `NILM` interface.
9. Select and save your mains sensor.
10. Open the Training interface.
11. In the first Training step:
   - use the autodetected internal training server if you installed the Home Assistant app
   - or choose `Custom External Server` and save the external trainer URL in the form `http://<host-or-ip>:<port>/train`
12. Confirm the training server is ready.
13. Train appliance models, validate in dashboard preview, then enable live publishing.

## Quick Start: Home Assistant Container

Run both NILM services with Docker Compose and connect `NILM` to Home Assistant through the normal Home Assistant REST and WebSocket APIs.

Create a long-lived access token in Home Assistant:

1. Open your Home Assistant user profile.
2. Go to `Security`.
3. Under `Long-lived access tokens`, create a token for NILM.
4. Copy it immediately. Home Assistant only shows the token once.

Create a folder for the Compose stack and add this `.env` file:

```env
HA_TOKEN=replace_with_your_home_assistant_long_lived_access_token
HA_REST_API_URL=http://YOUR_HOME_ASSISTANT_HOST_IP:8123/api
HA_WS_URL=ws://YOUR_HOME_ASSISTANT_HOST_IP:8123/api/websocket
```

Use the host machine LAN IP or hostname that other devices can use to reach Home Assistant.

Add this `docker-compose.yml` file in the same folder:

```yaml
services:
  nilm-trainer:
    image: ghcr.io/lgarciamarrero92/ha-nilm-trainer:latest
    container_name: nilm-trainer
    restart: unless-stopped
    ports:
      - "8024:8080"
    networks:
      - nilm

  nilm-edge:
    image: ghcr.io/lgarciamarrero92/ha-nilm-edge:latest
    container_name: nilm-edge
    restart: unless-stopped
    ports:
      - "8099:8099"
    volumes:
      - nilm_data:/data
    environment:
      SUPERVISOR_TOKEN: "${HA_TOKEN}"
      HA_REST_API_URL: "${HA_REST_API_URL}"
      HA_WS_URL: "${HA_WS_URL}"
    networks:
      - nilm

volumes:
  nilm_data:

networks:
  nilm:
    driver: bridge
```

`SUPERVISOR_TOKEN` is set from `HA_TOKEN` because the NILM container uses the same environment variable name for Home Assistant API authentication.

Start the stack:

```bash
docker compose up -d
```

Open the NILM interface:

```text
http://<docker-host>:8099/
```

Then:

1. Select and save your mains power sensor in the NILM dashboard.
2. Open the Training interface.
3. Choose `Custom External Server`.
4. Save this training server URL:

```text
http://nilm-trainer:8080/train
```

## Why Two Services

Training and live inference are split by design:

- `NILM` stays lightweight and responsive for continuous runtime.
- `NILM Training Server` handles heavier ML training workloads.
- `NILM Training Server` is only needed when you want to train or retrain models. After your models are trained and enabled in `NILM`, you can stop the training service and keep only `NILM` running for live inference.
- When the training server runs as a Docker container, `NILM` connects to it through the URL you save in the Training page.

## Why NILM Requests Supervisor Manager Role On Home Assistant OS

When installed as a Home Assistant OS app, `NILM` requests the Home Assistant Supervisor `manager` role so it can query the Supervisor API and autodetect the `NILM Training Server` add-on for you.

This is used to:

- Enumerate installed add-ons.
- Detect whether `NILM Training Server` is installed and started.
- Read its internal add-on hostname.
- Build the internal training server URL automatically so you do not have to enter it manually in the common case.

Home Assistant Container installs use the long-lived access token shown in the container quick start instead.

## Main Workflow

1. In `NILM` Training, choose manual interval labeling or sensor-based labeling.
2. Select the internal Home Assistant OS training server or save a custom external training server URL in the first Training step.
3. Select a mains range that you can label completely.
4. Prepare and send the training job to `NILM Training Server`.
5. Wait for job completion in the Training Jobs table.
6. Validate predictions in the NILM Dashboard.
7. Enable live publishing for selected models.

Notes:
- Training range is limited to the previous 7 days.
- Live entities update approximately every 8 seconds.

## Documentation

Main end-user documentation:

- https://ha-nilm.bigwicho.com/

## License

This project is licensed under the Apache License 2.0. See [LICENSE](LICENSE) for details.
