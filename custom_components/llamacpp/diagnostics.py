"""Diagnostics support for the llama.cpp integration."""

from __future__ import annotations

from typing import Any

from homeassistant.config_entries import ConfigEntry
from homeassistant.core import HomeAssistant
from homeassistant.helpers import device_registry as dr
from homeassistant.helpers.update_coordinator import DataUpdateCoordinator


def _mask_path(value: str | None) -> str | None:
    """Reduce a filesystem path to its basename for diagnostics."""
    if not value:
        return value
    return value.rsplit("/", 1)[-1]


def _coordinator_state(coord: DataUpdateCoordinator | None) -> dict[str, Any] | None:
    if coord is None:
        return None
    data = coord.data
    if isinstance(data, dict):
        data = dict(data)
        props = data.get("props")
        if isinstance(props, dict) and "model_path" in props:
            props = {**props, "model_path": _mask_path(props.get("model_path"))}
            data["props"] = props
    return {
        "last_update_success": coord.last_update_success,
        "data": data,
    }


async def async_get_config_entry_diagnostics(
    hass: HomeAssistant, entry: ConfigEntry
) -> dict[str, Any]:
    """Return diagnostics for a config entry."""
    runtime = entry.runtime_data or {}
    device_registry = dr.async_get(hass)
    devices = [
        {
            "name": device.name,
            "model": device.model,
            "sw_version": device.sw_version,
            "identifiers": [f"{d[0]}:{d[1]}" for d in device.identifiers],
        }
        for device in dr.async_entries_for_config_entry(device_registry, entry.entry_id)
    ]
    return {
        "config": {
            "data": dict(entry.data),
            "options": dict(entry.options),
        },
        "devices": devices,
        "llama": _coordinator_state(runtime.get("llama")),
        "gpu": _coordinator_state(runtime.get("gpu")),
    }
