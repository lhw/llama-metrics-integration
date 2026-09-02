"""Tests for the config and options flows."""

from __future__ import annotations

import aiohttp
from aioresponses import aioresponses
from custom_components.llamacpp.const import (
    CONF_GPU_URL,
    CONF_LLAMA_URL,
    CONF_SCAN_INTERVAL,
    DOMAIN,
)
from tests.conftest import GPU_URL, LLAMA_URL


def _user_data(overrides: dict | None = None) -> dict:
    data = {CONF_LLAMA_URL: LLAMA_URL, CONF_GPU_URL: GPU_URL, CONF_SCAN_INTERVAL: 15}
    if overrides:
        data.update(overrides)
    return data


async def test_user_flow_success(hass, mock_endpoints) -> None:
    result = await hass.config_entries.flow.async_init(
        DOMAIN, context={"source": "user"}, data=_user_data()
    )
    assert result["type"] == "create_entry"
    assert result["title"] == "192.168.1.197:8001"
    assert result["data"][CONF_LLAMA_URL] == LLAMA_URL
    assert result["data"][CONF_GPU_URL] == GPU_URL


async def test_user_flow_bare_url_and_no_gpu(hass, mock_endpoints) -> None:
    result = await hass.config_entries.flow.async_init(
        DOMAIN,
        context={"source": "user"},
        data=_user_data({CONF_LLAMA_URL: "192.168.1.197:8001", CONF_GPU_URL: ""}),
    )
    assert result["type"] == "create_entry"
    assert result["data"][CONF_LLAMA_URL] == "http://192.168.1.197:8001"
    assert result["data"][CONF_GPU_URL] is None


async def test_user_flow_omits_gpu_url(hass, mock_endpoints) -> None:
    """GPU URL is optional: a submission without it still creates the entry."""
    result = await hass.config_entries.flow.async_init(
        DOMAIN,
        context={"source": "user"},
        data={CONF_LLAMA_URL: LLAMA_URL, CONF_SCAN_INTERVAL: 15},
    )
    assert result["type"] == "create_entry"
    assert result["data"][CONF_GPU_URL] is None


async def test_user_flow_cannot_connect(hass) -> None:
    with aioresponses() as mocked:
        mocked.get(f"{LLAMA_URL}/metrics", exception=aiohttp.ClientConnectionError())
        result = await hass.config_entries.flow.async_init(
            DOMAIN,
            context={"source": "user"},
            data=_user_data({CONF_GPU_URL: ""}),
        )
    assert result["type"] == "form"
    assert result["errors"] == {CONF_LLAMA_URL: "cannot_connect"}


async def test_user_flow_invalid_url(hass) -> None:
    result = await hass.config_entries.flow.async_init(
        DOMAIN,
        context={"source": "user"},
        data=_user_data({CONF_LLAMA_URL: "not a url", CONF_GPU_URL: ""}),
    )
    assert result["type"] == "form"
    assert result["errors"] == {CONF_LLAMA_URL: "invalid"}


async def test_user_flow_duplicate_aborts(hass, mock_endpoints, config_entry) -> None:
    config_entry.add_to_hass(hass)
    result = await hass.config_entries.flow.async_init(
        DOMAIN,
        context={"source": "user"},
        data=_user_data({CONF_GPU_URL: ""}),
    )
    assert result["type"] == "abort"
    assert result["reason"] == "unique_already_configured"


async def test_options_flow_updates_interval(
    hass, loaded_entry, mock_endpoints
) -> None:
    result = await hass.config_entries.options.async_init(loaded_entry.entry_id)
    assert result["type"] == "form"
    result = await hass.config_entries.options.async_configure(
        result["flow_id"],
        user_input={
            CONF_LLAMA_URL: LLAMA_URL,
            CONF_GPU_URL: GPU_URL,
            CONF_SCAN_INTERVAL: 30,
        },
    )
    assert result["type"] == "create_entry"
    await hass.async_block_till_done()
    assert loaded_entry.options[CONF_SCAN_INTERVAL] == 30
