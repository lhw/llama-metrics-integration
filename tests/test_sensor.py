"""Tests for the sensor platform entity mapping."""

from __future__ import annotations

from aioresponses import aioresponses
from tests.conftest import (
    GPU_URL,
    LLAMA_URL,
    SAMPLE_GPU,
    SAMPLE_HEALTH,
    SAMPLE_METRICS,
    SAMPLE_PROPS,
    SAMPLE_SLOTS,
    get_state,
)


async def test_known_metric_sensors(hass, loaded_entry) -> None:
    state = get_state(hass, "sensor", "llamacpp:prompt_tokens_total")
    assert state is not None
    assert float(state.state) == 67837.0

    state = get_state(hass, "sensor", "llamacpp:predicted_tokens_seconds")
    assert state.attributes.get("unit_of_measurement") == "tokens/s"
    assert state.attributes.get("state_class") == "measurement"

    state = get_state(hass, "sensor", "llamacpp:prompt_cache_accounted_bytes")
    assert state.attributes.get("device_class") == "data_size"


async def test_labeled_metric_series(hass, loaded_entry) -> None:
    s0 = get_state(
        hass,
        "sensor",
        'llamacpp:spec_decode_num_accepted_tokens_per_pos_total{position="0"}',
    )
    s1 = get_state(
        hass,
        "sensor",
        'llamacpp:spec_decode_num_accepted_tokens_per_pos_total{position="1"}',
    )
    assert s0 is not None and s1 is not None
    assert float(s0.state) == 94.0
    assert float(s1.state) == 78.0


async def test_unknown_metric_fallback(hass, loaded_entry) -> None:
    state = get_state(hass, "sensor", "llamacpp:custom_thing_total")
    assert state is not None
    assert float(state.state) == 42.0
    assert state.attributes.get("unit_of_measurement") is None
    assert state.attributes.get("state_class") is None


async def test_props_sensors(hass, loaded_entry) -> None:
    assert get_state(hass, "sensor", "model_alias").state == "TestModel-8B"
    assert get_state(hass, "sensor", "model_ftype").state == "Q4_K_M"
    assert float(get_state(hass, "sensor", "total_slots").state) == 1.0


async def test_slot_subdevice_sensors(hass, loaded_entry) -> None:
    assert float(get_state(hass, "sensor", "slot-0:n_ctx").state) == 200192.0
    assert float(get_state(hass, "sensor", "slot-0:n_prompt_tokens").state) == 35671.0
    assert float(get_state(hass, "sensor", "slot-0:id_task").state) == 980.0


async def test_gpu_sensors(hass, loaded_entry) -> None:
    assert float(get_state(hass, "sensor", "gpu:gpu_utilization").state) == 93.0
    state = get_state(hass, "sensor", "gpu:temperature_c")
    assert float(state.state) == 67.0
    assert state.attributes.get("device_class") == "temperature"
    assert get_state(hass, "sensor", "gpu:gpu_name").state == "NVIDIA GeForce RTX 4090"


async def test_entities_unavailable_when_down(hass, config_entry) -> None:
    config_entry.add_to_hass(hass)
    with aioresponses() as mocked:
        # FIFO: setup consumes the 200, the manual refresh consumes the 500
        mocked.get(f"{LLAMA_URL}/metrics", body=SAMPLE_METRICS)
        mocked.get(f"{LLAMA_URL}/metrics", status=500)
        mocked.get(f"{LLAMA_URL}/slots", payload=SAMPLE_SLOTS, repeat=True)
        mocked.get(f"{LLAMA_URL}/props", payload=SAMPLE_PROPS, repeat=True)
        mocked.get(f"{LLAMA_URL}/health", payload=SAMPLE_HEALTH, repeat=True)
        mocked.get(GPU_URL, payload=SAMPLE_GPU, repeat=True)
        assert await hass.config_entries.async_setup(config_entry.entry_id)
        await hass.async_block_till_done()
        assert (
            get_state(hass, "sensor", "llamacpp:prompt_tokens_seconds").state
            != "unavailable"
        )

        await config_entry.runtime_data["llama"].async_refresh()
        await hass.async_block_till_done()
        assert (
            get_state(hass, "sensor", "llamacpp:prompt_tokens_seconds").state
            == "unavailable"
        )
