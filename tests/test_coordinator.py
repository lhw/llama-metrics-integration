"""Tests for the coordinators and prometheus parsing."""

from __future__ import annotations

import math

from aioresponses import aioresponses
from custom_components.llamacpp.const import (
    CONF_LLAMA_URL,
    DOMAIN,
)
from custom_components.llamacpp.coordinator import (
    create_coordinators,
    parse_prometheus,
    split_metric_key,
)
from pytest_homeassistant_custom_component.common import MockConfigEntry
from tests.conftest import (
    GPU_URL,
    LLAMA_URL,
)


def test_parse_prometheus_skips_comments_and_bad_lines() -> None:
    text = '# HELP x y\n# TYPE x counter\nx 1.5\nbad line\ny{a="1"} 2\nz NaN\n'
    metrics, names = parse_prometheus(text)
    assert metrics["x"] == 1.5
    assert metrics['y{a="1"}'] == 2.0
    assert math.isnan(metrics["z"])
    assert names == {"x", "y", "z"}


def test_split_metric_key() -> None:
    assert split_metric_key("m") == ("m", {})
    assert split_metric_key('m{a="1",b="2"}') == ("m", {"a": "1", "b": "2"})


async def test_llama_coordinator_update(hass, config_entry, mock_endpoints) -> None:
    config_entry.add_to_hass(hass)
    coords = create_coordinators(hass, config_entry)
    await coords["llama"].async_refresh()
    assert coords["llama"].last_update_success
    data = coords["llama"].data
    assert data["metrics"]["llamacpp:prompt_tokens_total"] == 67837.0
    assert data["metrics"]["llamacpp:prompt_cache_accounted_bytes"] == 3.789e9
    assert data["slots"][0]["id"] == 0
    assert data["props"]["model_alias"] == "TestModel-8B"
    assert data["health"] == "ok"
    assert "llamacpp:custom_thing_total" in data["metric_names"]


async def test_llama_coordinator_unavailable(hass, config_entry) -> None:
    config_entry.add_to_hass(hass)
    with aioresponses() as mocked:
        mocked.get(f"{LLAMA_URL}/metrics", status=500)
        coords = create_coordinators(hass, config_entry)
        await coords["llama"].async_refresh()
        assert not coords["llama"].last_update_success


async def test_gpu_coordinator_update(hass, config_entry, mock_endpoints) -> None:
    config_entry.add_to_hass(hass)
    coords = create_coordinators(hass, config_entry)
    await coords["gpu"].async_refresh()
    assert coords["gpu"].last_update_success
    assert coords["gpu"].data["gpu_utilization"] == 93.0


async def test_gpu_coordinator_unavailable(hass, config_entry) -> None:
    config_entry.add_to_hass(hass)
    with aioresponses() as mocked:
        mocked.get(GPU_URL, status=500)
        coords = create_coordinators(hass, config_entry)
        await coords["gpu"].async_refresh()
        assert not coords["gpu"].last_update_success


async def test_no_gpu_when_unconfigured(hass) -> None:
    entry = MockConfigEntry(
        domain=DOMAIN,
        title="x",
        unique_id="http://a:1",
        data={CONF_LLAMA_URL: LLAMA_URL},
    )
    entry.add_to_hass(hass)
    coords = create_coordinators(hass, entry)
    assert coords["gpu"] is None
