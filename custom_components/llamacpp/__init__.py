"""The llama.cpp integration."""

from __future__ import annotations

import logging

from homeassistant.config_entries import ConfigEntry
from homeassistant.core import HomeAssistant

from .coordinator import create_coordinators

PLATFORMS = ["sensor", "binary_sensor"]

LOGGER = logging.getLogger(__package__)


async def async_setup_entry(hass: HomeAssistant, entry: ConfigEntry) -> bool:
    """Set up llama.cpp from a config entry."""
    coordinators = create_coordinators(hass, entry)
    entry.runtime_data = coordinators

    # The llama.cpp server is the primary data source: if it is unreachable at
    # setup time, defer (HA retries with backoff) rather than adding entities
    # that are all unavailable.
    await coordinators["llama"].async_config_entry_first_refresh()

    # The GPU is optional and a separate device: prime it best-effort so its
    # entities are populated immediately, but never let it block the entry.
    if coordinators["gpu"] is not None:
        await coordinators["gpu"].async_refresh()

    await hass.config_entries.async_forward_entry_setups(entry, PLATFORMS)
    return True


async def async_unload_entry(hass: HomeAssistant, entry: ConfigEntry) -> bool:
    """Unload a config entry."""
    if unloaded := await hass.config_entries.async_unload_platforms(entry, PLATFORMS):
        entry.runtime_data = None
    return unloaded
