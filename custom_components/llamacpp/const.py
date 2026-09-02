"""Constants for the llama.cpp integration."""

from __future__ import annotations

from dataclasses import dataclass

from homeassistant.components.sensor import SensorDeviceClass, SensorStateClass
from homeassistant.const import (
    PERCENTAGE,
    UnitOfFrequency,
    UnitOfInformation,
    UnitOfPower,
    UnitOfTemperature,
    UnitOfTime,
)

DOMAIN = "llamacpp"

CONF_LLAMA_URL = "llama_url"
CONF_GPU_URL = "gpu_url"
CONF_SCAN_INTERVAL = "scan_interval"

DEFAULT_SCAN_INTERVAL = 15
MIN_SCAN_INTERVAL = 5

# llama.cpp server endpoints
PATH_METRICS = "/metrics"
PATH_SLOTS = "/slots"
PATH_PROPS = "/props"


@dataclass(frozen=True)
class MetricDef:
    """Static description of a known llama.cpp / GPU metric."""

    name: str | None = None
    unit: str | None = None
    state_class: SensorStateClass | None = None
    device_class: SensorDeviceClass | None = None
    # Front-and-center: surfaced (not diagnostic). Statistical metrics that are
    # not front-and-center are disabled by default to avoid state spam.
    front: bool = False


# Known llama.cpp Prometheus metrics, keyed by full metric name (with the
# `llamacpp:` namespace prefix). Series that carry labels (e.g. per draft
# position) resolve through their base name here and get a per-series entity.
LLAMACPP_METRICS: dict[str, MetricDef] = {
    # Throughput gauges
    "llamacpp:prompt_tokens_seconds": MetricDef(
        name="Prompt throughput",
        unit="tokens/s",
        state_class=SensorStateClass.MEASUREMENT,
        front=True,
    ),
    "llamacpp:predicted_tokens_seconds": MetricDef(
        name="Generation throughput",
        unit="tokens/s",
        state_class=SensorStateClass.MEASUREMENT,
        front=True,
    ),
    # Token / time counters
    "llamacpp:prompt_tokens_total": MetricDef(
        name="Prompt tokens",
        unit="tokens",
        state_class=SensorStateClass.TOTAL_INCREASING,
        front=True,
    ),
    "llamacpp:tokens_predicted_total": MetricDef(
        name="Generated tokens",
        unit="tokens",
        state_class=SensorStateClass.TOTAL_INCREASING,
        front=True,
    ),
    "llamacpp:prompt_seconds_total": MetricDef(
        name="Prompt time",
        unit=UnitOfTime.SECONDS,
        state_class=SensorStateClass.TOTAL_INCREASING,
        front=True,
    ),
    "llamacpp:tokens_predicted_seconds_total": MetricDef(
        name="Generation time",
        unit=UnitOfTime.SECONDS,
        state_class=SensorStateClass.TOTAL_INCREASING,
        front=True,
    ),
    # Decode counters
    "llamacpp:n_decode_total": MetricDef(
        name="Decode calls", state_class=SensorStateClass.TOTAL_INCREASING
    ),
    "llamacpp:n_tokens_max": MetricDef(
        name="Max tokens observed",
        unit="tokens",
        state_class=SensorStateClass.TOTAL_INCREASING,
    ),
    # Speculative decoding
    "llamacpp:spec_decode_num_draft_tokens_total": MetricDef(
        name="Draft tokens",
        unit="tokens",
        state_class=SensorStateClass.TOTAL_INCREASING,
    ),
    "llamacpp:spec_decode_num_accepted_tokens_total": MetricDef(
        name="Accepted draft tokens",
        unit="tokens",
        state_class=SensorStateClass.TOTAL_INCREASING,
    ),
    "llamacpp:spec_decode_num_drafts_total": MetricDef(
        name="Speculative drafts", state_class=SensorStateClass.TOTAL_INCREASING
    ),
    "llamacpp:spec_decode_num_accepted_tokens_per_pos_total": MetricDef(
        name="Accepted tokens (position)",
        unit="tokens",
        state_class=SensorStateClass.TOTAL_INCREASING,
    ),
    # Prompt cache counters
    "llamacpp:prompt_cache_admission_attempts_total": MetricDef(
        name="Prompt-cache admission attempts",
        state_class=SensorStateClass.TOTAL_INCREASING,
    ),
    "llamacpp:prompt_cache_admission_successes_total": MetricDef(
        name="Prompt-cache admission successes",
        state_class=SensorStateClass.TOTAL_INCREASING,
    ),
    "llamacpp:prompt_cache_admission_failures_total": MetricDef(
        name="Prompt-cache admission failures",
        state_class=SensorStateClass.TOTAL_INCREASING,
    ),
    "llamacpp:prompt_cache_restore_attempts_total": MetricDef(
        name="Prompt-cache restore attempts",
        state_class=SensorStateClass.TOTAL_INCREASING,
    ),
    "llamacpp:prompt_cache_restore_successes_total": MetricDef(
        name="Prompt-cache restore successes",
        state_class=SensorStateClass.TOTAL_INCREASING,
    ),
    "llamacpp:prompt_cache_restore_failures_total": MetricDef(
        name="Prompt-cache restore failures",
        state_class=SensorStateClass.TOTAL_INCREASING,
    ),
    # Request / slot gauges
    "llamacpp:requests_processing": MetricDef(
        name="Requests processing", state_class=SensorStateClass.MEASUREMENT
    ),
    "llamacpp:requests_deferred": MetricDef(
        name="Requests deferred", state_class=SensorStateClass.MEASUREMENT
    ),
    "llamacpp:n_busy_slots_per_decode": MetricDef(
        name="Busy slots per decode", state_class=SensorStateClass.MEASUREMENT
    ),
    # KV exact-tail coverage
    "llamacpp:kv_tail_requested_tokens": MetricDef(
        name="KV tail requested tokens",
        unit="tokens",
        state_class=SensorStateClass.MEASUREMENT,
    ),
    "llamacpp:kv_tail_exact_tokens": MetricDef(
        name="KV tail exact tokens",
        unit="tokens",
        state_class=SensorStateClass.MEASUREMENT,
    ),
    "llamacpp:kv_tail_complete_groups": MetricDef(
        name="KV tail complete groups", state_class=SensorStateClass.MEASUREMENT
    ),
    "llamacpp:kv_tail_partial_groups": MetricDef(
        name="KV tail partial groups", state_class=SensorStateClass.MEASUREMENT
    ),
    "llamacpp:kv_tail_none_groups": MetricDef(
        name="KV tail none groups", state_class=SensorStateClass.MEASUREMENT
    ),
    "llamacpp:kv_tail_degraded_sequences": MetricDef(
        name="KV tail degraded sequences",
        state_class=SensorStateClass.MEASUREMENT,
    ),
    # Prompt cache size
    "llamacpp:prompt_cache_accounted_bytes": MetricDef(
        name="Prompt-cache size",
        unit=UnitOfInformation.BYTES,
        device_class=SensorDeviceClass.DATA_SIZE,
        state_class=SensorStateClass.MEASUREMENT,
    ),
}

# GPU exporter fields (plain JSON, no prometheus framing).
GPU_METRICS: dict[str, MetricDef] = {
    "gpu_utilization": MetricDef(
        name="GPU utilization",
        unit=PERCENTAGE,
        state_class=SensorStateClass.MEASUREMENT,
        front=True,
    ),
    "memory_utilization": MetricDef(
        name="Memory utilization",
        unit=PERCENTAGE,
        state_class=SensorStateClass.MEASUREMENT,
    ),
    "memory_used_mb": MetricDef(
        name="Memory used",
        unit=UnitOfInformation.MEBIBYTES,
        device_class=SensorDeviceClass.DATA_SIZE,
        state_class=SensorStateClass.MEASUREMENT,
        front=True,
    ),
    "memory_total_mb": MetricDef(
        name="Memory total",
        unit=UnitOfInformation.MEBIBYTES,
        device_class=SensorDeviceClass.DATA_SIZE,
    ),
    "temperature_c": MetricDef(
        name="Temperature",
        unit=UnitOfTemperature.CELSIUS,
        device_class=SensorDeviceClass.TEMPERATURE,
        state_class=SensorStateClass.MEASUREMENT,
        front=True,
    ),
    "power_draw_w": MetricDef(
        name="Power",
        unit=UnitOfPower.WATT,
        device_class=SensorDeviceClass.POWER,
        state_class=SensorStateClass.MEASUREMENT,
        front=True,
    ),
    "power_limit_w": MetricDef(
        name="Power limit",
        unit=UnitOfPower.WATT,
        device_class=SensorDeviceClass.POWER,
    ),
    "sm_clock_mhz": MetricDef(
        name="SM clock",
        unit=UnitOfFrequency.MEGAHERTZ,
        state_class=SensorStateClass.MEASUREMENT,
    ),
    "sm_clock_max_mhz": MetricDef(name="SM clock max", unit=UnitOfFrequency.MEGAHERTZ),
    "fan_speed": MetricDef(
        name="Fan speed", unit=PERCENTAGE, state_class=SensorStateClass.MEASUREMENT
    ),
    "gpu_name": MetricDef(name="GPU name"),
    "pci_bus_id": MetricDef(name="PCI bus ID"),
}


def humanize_metric(name: str) -> str:
    """Turn a raw metric name into a readable label.

    `llamacpp:prompt_tokens_seconds` -> `Prompt tokens seconds`.
    Used as a fallback display name for metrics not in the registry.
    """
    local = name.split(":", 1)[1] if ":" in name else name
    return local.replace("_", " ").title()
