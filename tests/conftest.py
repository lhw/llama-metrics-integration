"""Shared fixtures and sample payloads for the llama.cpp integration tests."""

from __future__ import annotations

import pytest
from aioresponses import aioresponses
from homeassistant.helpers import entity_registry as er
from pytest_homeassistant_custom_component.common import MockConfigEntry

pytest_plugins = "pytest_homeassistant_custom_component"

from custom_components.llamacpp.const import (  # noqa: E402
    CONF_GPU_URL,
    CONF_LLAMA_URL,
    CONF_SCAN_INTERVAL,
    DOMAIN,
)

LLAMA_URL = "http://192.168.1.197:8001"
GPU_URL = "http://192.168.1.197:9091"

SAMPLE_METRICS = """\
# HELP llamacpp:prompt_tokens_seconds Average prompt throughput in tokens/s.
# TYPE llamacpp:prompt_tokens_seconds gauge
llamacpp:prompt_tokens_seconds 1249.65
# HELP llamacpp:predicted_tokens_seconds Average generation throughput in tokens/s.
# TYPE llamacpp:predicted_tokens_seconds gauge
llamacpp:predicted_tokens_seconds 76.2029
# HELP llamacpp:prompt_tokens_total Number of prompt tokens processed.
# TYPE llamacpp:prompt_tokens_total counter
llamacpp:prompt_tokens_total 67837
# HELP llamacpp:tokens_predicted_total Number of generation tokens processed.
# TYPE llamacpp:tokens_predicted_total counter
llamacpp:tokens_predicted_total 350
# HELP llamacpp:prompt_seconds_total Prompt process time
# TYPE llamacpp:prompt_seconds_total counter
llamacpp:prompt_seconds_total 54.285
# HELP llamacpp:tokens_predicted_seconds_total Predict process time
# TYPE llamacpp:tokens_predicted_seconds_total counter
llamacpp:tokens_predicted_seconds_total 4.593
# HELP llamacpp:n_decode_total Total number of llama_decode() calls
# TYPE llamacpp:n_decode_total counter
llamacpp:n_decode_total 332
# HELP llamacpp:n_tokens_max Largest observed n_tokens.
# TYPE llamacpp:n_tokens_max counter
llamacpp:n_tokens_max 33631
# HELP llamacpp:spec_decode_num_draft_tokens_total Total draft tokens generated
# TYPE llamacpp:spec_decode_num_draft_tokens_total counter
llamacpp:spec_decode_num_draft_tokens_total 339
# HELP llamacpp:spec_decode_num_accepted_tokens_total Total draft tokens accepted
# TYPE llamacpp:spec_decode_num_accepted_tokens_total counter
llamacpp:spec_decode_num_accepted_tokens_total 237
# HELP llamacpp:spec_decode_num_drafts_total Total speculative decoding steps
# TYPE llamacpp:spec_decode_num_drafts_total counter
llamacpp:spec_decode_num_drafts_total 113
# HELP llamacpp:spec_decode_num_accepted_tokens_per_pos_total Accepted per pos
# TYPE llamacpp:spec_decode_num_accepted_tokens_per_pos_total counter
llamacpp:spec_decode_num_accepted_tokens_per_pos_total{position="0"} 94
llamacpp:spec_decode_num_accepted_tokens_per_pos_total{position="1"} 78
# HELP llamacpp:prompt_cache_admission_attempts_total cache admissions
# TYPE llamacpp:prompt_cache_admission_attempts_total counter
llamacpp:prompt_cache_admission_attempts_total 3
# HELP llamacpp:prompt_cache_accounted_bytes Serialized prompt-cache bytes
# TYPE llamacpp:prompt_cache_accounted_bytes gauge
llamacpp:prompt_cache_accounted_bytes 3.789e+09
# HELP llamacpp:requests_processing Number of requests processing.
# TYPE llamacpp:requests_processing gauge
llamacpp:requests_processing 1
# HELP llamacpp:requests_deferred Number of requests deferred.
# TYPE llamacpp:requests_deferred gauge
llamacpp:requests_deferred 0
# HELP llamacpp:n_busy_slots_per_decode Average busy slots per decode
# TYPE llamacpp:n_busy_slots_per_decode gauge
llamacpp:n_busy_slots_per_decode 1
# HELP llamacpp:kv_tail_requested_tokens Requested exact-tail tokens
# TYPE llamacpp:kv_tail_requested_tokens gauge
llamacpp:kv_tail_requested_tokens 2048
# HELP llamacpp:custom_thing_total A metric not in the registry (fallback path)
# TYPE llamacpp:custom_thing_total counter
llamacpp:custom_thing_total 42
"""

SAMPLE_SLOTS = [
    {
        "id": 0,
        "n_ctx": 200192,
        "speculative": True,
        "is_processing": True,
        "id_task": 980,
        "n_prompt_tokens": 35671,
        "n_prompt_tokens_processed": 1956,
        "n_prompt_tokens_cache": 0,
    }
]

SAMPLE_PROPS = {
    "model_alias": "TestModel-8B",
    "model_ftype": "Q4_K_M",
    "model_path": "/models/snapshots/deadbeef/TestModel-8B.gguf",
    "total_slots": 1,
    "is_sleeping": False,
    "build_info": "b1234-dirty",
}

SAMPLE_HEALTH = {"status": "ok"}

SAMPLE_GPU = {
    "gpu_utilization": 93.0,
    "memory_utilization": 71.0,
    "memory_used_mb": 23382.0,
    "memory_total_mb": 24564.0,
    "temperature_c": 67.0,
    "power_draw_w": 385.17,
    "power_limit_w": 450.0,
    "sm_clock_mhz": 2730.0,
    "sm_clock_max_mhz": 3105.0,
    "fan_speed": "40",
    "gpu_name": "NVIDIA GeForce RTX 4090",
    "pci_bus_id": "00000000:01:00.0",
    "timestamp": 1788276820.2613342,
}


@pytest.fixture(autouse=True)
def auto_enable_custom_integrations(enable_custom_integrations):
    """Enable custom integrations for every test."""
    yield


@pytest.fixture
def mock_endpoints():
    """Mock every llama.cpp + GPU endpoint with the sample payloads."""
    with aioresponses() as mocked:
        mocked.get(f"{LLAMA_URL}/metrics", body=SAMPLE_METRICS, repeat=True)
        mocked.get(f"{LLAMA_URL}/slots", payload=SAMPLE_SLOTS, repeat=True)
        mocked.get(f"{LLAMA_URL}/props", payload=SAMPLE_PROPS, repeat=True)
        mocked.get(f"{LLAMA_URL}/health", payload=SAMPLE_HEALTH, repeat=True)
        mocked.get(GPU_URL, payload=SAMPLE_GPU, repeat=True)
        yield mocked


@pytest.fixture
def config_entry() -> MockConfigEntry:
    """A config entry with both llama and GPU URLs configured."""
    return MockConfigEntry(
        domain=DOMAIN,
        title="192.168.1.197:8001",
        unique_id=LLAMA_URL,
        data={
            CONF_LLAMA_URL: LLAMA_URL,
            CONF_GPU_URL: GPU_URL,
            CONF_SCAN_INTERVAL: 15,
        },
    )


@pytest.fixture
async def loaded_entry(hass, config_entry, mock_endpoints):
    """Set up the integration from ``config_entry`` and return it."""
    config_entry.add_to_hass(hass)
    assert await hass.config_entries.async_setup(config_entry.entry_id)
    await hass.async_block_till_done()
    return config_entry


def get_state(hass, entity_domain: str, unique_id: str):
    """Return the State for an entity by its (component, unique_id)."""
    entity_id = er.async_get(hass).async_get_entity_id(entity_domain, DOMAIN, unique_id)
    return hass.states.get(entity_id) if entity_id else None


def entity_exists(hass, entity_domain: str, unique_id: str) -> bool:
    """True if an entity with this (domain, unique_id) is registered."""
    entity_id = er.async_get(hass).async_get_entity_id(entity_domain, DOMAIN, unique_id)
    return entity_id is not None


def is_entity_enabled(hass, entity_domain: str, unique_id: str) -> bool:
    """True if the entity exists in the registry and is not disabled."""
    registry = er.async_get(hass)
    entity_id = registry.async_get_entity_id(entity_domain, DOMAIN, unique_id)
    if entity_id is None:
        return False
    return registry.async_get(entity_id).disabled_by is None


def entity_category(hass, entity_domain: str, unique_id: str):
    """Registry category: ``None`` when surfaced, else e.g. "diagnostic"."""
    registry = er.async_get(hass)
    entity_id = registry.async_get_entity_id(entity_domain, DOMAIN, unique_id)
    if entity_id is None:
        return None
    return registry.async_get(entity_id).entity_category
