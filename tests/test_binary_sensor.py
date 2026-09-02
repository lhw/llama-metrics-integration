"""Tests for the binary_sensor platform."""

from __future__ import annotations

import aiohttp
from aioresponses import aioresponses
from tests.conftest import (
    GPU_URL,
    LLAMA_URL,
    SAMPLE_GPU,
    SAMPLE_METRICS,
    SAMPLE_PROPS,
    SAMPLE_SLOTS,
    entity_category,
    get_state,
)


async def test_host_online_binary(hass, loaded_entry) -> None:
    state = get_state(hass, "binary_sensor", "online")
    assert state is not None
    assert state.state == "on"
    assert entity_category(hass, "binary_sensor", "online") == "diagnostic"


async def test_host_online_binary_stays_available_when_down(hass, config_entry) -> None:
    config_entry.add_to_hass(hass)
    with aioresponses() as mocked:
        # FIFO: setup consumes the 200, the manual refresh consumes the failure.
        mocked.get(f"{LLAMA_URL}/metrics", body=SAMPLE_METRICS)
        mocked.get(f"{LLAMA_URL}/metrics", exception=aiohttp.ClientConnectionError())
        mocked.get(f"{LLAMA_URL}/slots", payload=SAMPLE_SLOTS, repeat=True)
        mocked.get(f"{LLAMA_URL}/props", payload=SAMPLE_PROPS, repeat=True)
        mocked.get(GPU_URL, payload=SAMPLE_GPU, repeat=True)
        assert await hass.config_entries.async_setup(config_entry.entry_id)
        await hass.async_block_till_done()
        assert get_state(hass, "binary_sensor", "online").state == "on"

        await config_entry.runtime_data["llama"].async_refresh()
        await hass.async_block_till_done()
        # Down: still available (not "unavailable") and now reporting off.
        assert get_state(hass, "binary_sensor", "online").state == "off"


async def test_sleeping_binary(hass, loaded_entry) -> None:
    state = get_state(hass, "binary_sensor", "sleeping")
    assert state is not None
    assert state.state == "off"


async def test_slot_active_binary(hass, loaded_entry) -> None:
    state = get_state(hass, "binary_sensor", "slot-0:is_processing")
    assert state is not None
    assert state.state == "on"


async def test_slot_speculative_binary(hass, loaded_entry) -> None:
    state = get_state(hass, "binary_sensor", "slot-0:speculative")
    assert state is not None
    assert state.state == "on"
