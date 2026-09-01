# llama.cpp for Home Assistant

A Home Assistant custom integration that turns a running [llama.cpp](https://github.com/ggml-org/llama.cpp) server into first-class HA entities. It polls the server's Prometheus `/metrics` endpoint (plus `/slots`, `/props`, and `/health`) and exposes every metric the server reports, and — when configured — a separate GPU metrics exporter as its own device.

- **One device per llama.cpp address**, model name and build info pulled from `/props`.
- **Every metric mapped.** All known llama.cpp metrics get the right unit, state class, and device class. Anything new the server starts emitting still gets a sensible fallback sensor, so nothing is silently dropped.
- **Labeled series** (e.g. `..._accepted_tokens_per_pos_total{position="0"}`) become one entity per label set.
- **Slots as subdevices.** Each active slot becomes a subdevice with its own context/token sensors and activity binary sensors.
- **Optional GPU device.** Point it at a GPU metrics exporter (JSON at the base URL) to get utilization, memory, temperature, power, clocks, and fan on a separate device.
- **Live availability.** Entities go `unavailable` the moment a poll fails and recover automatically.
- **No extra Python dependencies.** Uses only HA's built-in `aiohttp`.

## Requirements

A reachable llama.cpp server exposing:

| Endpoint    | Used for                                   |
| ----------- | ------------------------------------------ |
| `/metrics`  | Prometheus metrics (required)              |
| `/slots`    | Per-slot state (subdevices)                |
| `/props`    | Model name, build info, sleeping state     |
| `/health`   | Server health                              |

Optionally, a GPU metrics exporter serving a JSON object at its base URL. A
ready-to-run, headless podman container for the exporter lives in
[`gpu-exporter/`](gpu-exporter/).

## Installation

### HACS

1. In HACS, add this repository as a custom repository.
2. Install the **llama.cpp** integration.
3. Restart Home Assistant.

### Manual

Copy the `custom_components/llamacpp` folder into your Home Assistant configuration's `custom_components/` directory, then restart Home Assistant.

## Configuration

Go to **Settings → Devices & Services → Add Integration → llama.cpp**.

| Field            | Description                                                        |
| ---------------- | ------------------------------------------------------------------ |
| Server URL       | Base URL of the llama.cpp server, e.g. `http://192.168.1.197:8001`  |
| GPU exporter URL | Optional. Base URL of the GPU metrics exporter. Leave empty to skip. |
| Polling interval | Seconds between polls (minimum 5, default 15).                      |

A bare `host:port` is accepted and defaults to `http://`. The GPU URL and polling interval can be changed later from the integration's options.

## Entities

**llama.cpp device** (from `/metrics`, one per series the server emits):

- Throughput: prompt and generation tokens/s
- Counters: prompt/generation tokens and time, decode calls, speculative-decode counters (draft, accepted, per-position)
- Prompt cache: admission/restore attempts, successes, failures, and serialized size
- Request gauges: processing, deferred, busy slots per decode
- KV exact-tail coverage: requested/exact tokens, complete/partial/none groups, degraded sequences

**Informational** (from `/props`): model, model type, slot count, and a "model sleeping" binary sensor.

**Slot subdevices** (one per active slot): context size, prompt tokens (total/processed/cached), task ID, plus "is processing" and "speculative" binary sensors.

**GPU device** (from the exporter): utilization, memory (used/total), temperature, power draw/limit, SM clock, fan speed, and the GPU/PCI identifiers.

All metric entities are categorized as **diagnostics** to keep the default dashboard uncluttered; promote any you like to the dashboard.

## Development

```bash
python -m venv .venv && source .venv/bin/activate
pip install homeassistant pytest-homeassistant-custom-component aioresponses ruff
pytest            # runs the test suite
ruff check custom_components tests
```

The test suite runs entirely against mocked endpoints (`aioresponses`); no network access is required.
