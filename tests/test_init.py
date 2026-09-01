"""Tests for entry setup/unload and device registration."""

from __future__ import annotations

from aioresponses import aioresponses
from custom_components.llamacpp.const import CONF_LLAMA_URL, DOMAIN
from homeassistant.config_entries import ConfigEntryState
from homeassistant.helpers import device_registry as dr
from pytest_homeassistant_custom_component.common import MockConfigEntry
from tests.conftest import (
    LLAMA_URL,
    SAMPLE_HEALTH,
    SAMPLE_METRICS,
    SAMPLE_PROPS,
    SAMPLE_SLOTS,
    get_state,
)


async def test_setup_and_unload(hass, config_entry, mock_endpoints) -> None:
    config_entry.add_to_hass(hass)
    assert await hass.config_entries.async_setup(config_entry.entry_id)
    await hass.async_block_till_done()
    assert get_state(hass, "sensor", "model_alias") is not None

    assert await hass.config_entries.async_unload(config_entry.entry_id)
    await hass.async_block_till_done()
    assert config_entry.state is ConfigEntryState.NOT_LOADED


async def test_devices_registered(hass, loaded_entry) -> None:
    registry = dr.async_get(hass)
    devices = dr.async_entries_for_config_entry(registry, loaded_entry.entry_id)
    names = {d.name for d in devices}
    # main llama device + one slot subdevice + the GPU device
    assert names == {
        "llama.cpp 192.168.1.197:8001",
        "Slot 0",
        "NVIDIA GeForce RTX 4090",
    }
    main = next(d for d in devices if d.name.startswith("llama.cpp"))
    slot = next(d for d in devices if d.name == "Slot 0")
    assert slot.via_device_id == main.id
    assert main.model == "TestModel-8B"
    assert main.sw_version == "b1234-dirty"


async def test_setup_defers_when_server_down(hass, config_entry) -> None:
    """If the llama server is unreachable at setup, the entry is not set up."""
    config_entry.add_to_hass(hass)
    with aioresponses() as mocked:
        mocked.get(f"{LLAMA_URL}/metrics", status=500)
        assert not await hass.config_entries.async_setup(config_entry.entry_id)
        await hass.async_block_till_done()
    assert get_state(hass, "sensor", "model_alias") is None


async def test_no_gpu_device_when_unconfigured(hass) -> None:
    entry = MockConfigEntry(
        domain=DOMAIN,
        title="llama-only",
        unique_id="http://10.0.0.5:8001",
        data={CONF_LLAMA_URL: "http://10.0.0.5:8001"},
    )
    entry.add_to_hass(hass)
    with aioresponses() as mocked:
        mocked.get("http://10.0.0.5:8001/metrics", body=SAMPLE_METRICS)
        mocked.get("http://10.0.0.5:8001/slots", payload=SAMPLE_SLOTS)
        mocked.get("http://10.0.0.5:8001/props", payload=SAMPLE_PROPS)
        mocked.get("http://10.0.0.5:8001/health", payload=SAMPLE_HEALTH)
        assert await hass.config_entries.async_setup(entry.entry_id)
        await hass.async_block_till_done()
    registry = dr.async_get(hass)
    devices = dr.async_entries_for_config_entry(registry, entry.entry_id)
    names = {d.name for d in devices}
    assert not any(name.startswith("GPU") or "RTX" in name for name in names)
    assert "llama.cpp 10.0.0.5:8001" in names
