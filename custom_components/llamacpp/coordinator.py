"""Data coordinators for the llama.cpp integration."""

from __future__ import annotations

import asyncio
import logging
import re
from datetime import timedelta
from typing import Any

import aiohttp
from homeassistant.config_entries import ConfigEntry
from homeassistant.core import HomeAssistant
from homeassistant.helpers.aiohttp_client import async_get_clientsession
from homeassistant.helpers.update_coordinator import (
    DataUpdateCoordinator,
    UpdateFailed,
)

from .const import (
    CONF_GPU_URL,
    CONF_LLAMA_URL,
    CONF_SCAN_INTERVAL,
    DEFAULT_SCAN_INTERVAL,
    PATH_HEALTH,
    PATH_METRICS,
    PATH_PROPS,
    PATH_SLOTS,
)

LOGGER = logging.getLogger(__package__)

CLIENT_TIMEOUT = aiohttp.ClientTimeout(total=10)

# A single prometheus sample line: `name{labels} value [timestamp]`.
_PROM_LINE = re.compile(
    r"^(?P<name>[a-zA-Z_:][a-zA-Z0-9_:]*)(?P<labels>\{[^}]*\})?\s+(?P<value>\S+)"
)


def parse_prometheus(text: str) -> tuple[dict[str, float], set[str]]:
    """Parse prometheus text into ``{full_key: value}`` and the set of base names.

    ``full_key`` is the metric name, or ``name{labels}`` when the sample carries
    labels, so each labelled series is kept as its own key.
    """
    metrics: dict[str, float] = {}
    names: set[str] = set()
    for raw in text.splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        match = _PROM_LINE.match(line)
        if not match:
            continue
        name = match.group("name")
        try:
            value = float(match.group("value"))
        except ValueError:
            continue
        labels = match.group("labels")
        names.add(name)
        metrics[name if labels is None else f"{name}{labels}"] = value
    return metrics, names


def split_metric_key(key: str) -> tuple[str, dict[str, str]]:
    """Split a metric full_key into ``(name, {label: value})``."""
    if "{" not in key:
        return key, {}
    name, _, rest = key.partition("{")
    labels: dict[str, str] = {}
    for part in rest.rstrip("}").split(","):
        if "=" in part:
            k, v = part.split("=", 1)
            labels[k.strip()] = v.strip().strip('"')
    return name, labels


class _BaseCoordinator(DataUpdateCoordinator[dict[str, Any]]):
    """Shared aiohttp plumbing for the llama.cpp and GPU coordinators."""

    def __init__(
        self,
        hass: HomeAssistant,
        entry: ConfigEntry,
        name: str,
        scan_interval: int,
    ) -> None:
        super().__init__(
            hass,
            LOGGER,
            config_entry=entry,
            name=name,
            update_interval=timedelta(seconds=scan_interval),
        )
        self.session = async_get_clientsession(hass)

    async def _async_get_text(self, url: str) -> str:
        try:
            async with self.session.get(url, timeout=CLIENT_TIMEOUT) as resp:
                resp.raise_for_status()
                return await resp.text()
        except (TimeoutError, aiohttp.ClientError) as err:
            raise UpdateFailed(f"Error communicating with {url}: {err}") from err

    async def _async_get_json_optional(self, url: str) -> Any:
        """Fetch JSON, returning ``None`` instead of failing on any error.

        ``/slots`` is a list and the rest are dicts; callers filter by type.
        """
        try:
            async with self.session.get(url, timeout=CLIENT_TIMEOUT) as resp:
                resp.raise_for_status()
                return await resp.json()
        except (TimeoutError, aiohttp.ClientError):
            return None


class LlamaCPPCoordinator(_BaseCoordinator):
    """Poll a llama.cpp server: /metrics, /slots, /props, /health."""

    def __init__(self, hass: HomeAssistant, entry: ConfigEntry) -> None:
        self.base_url = entry.data[CONF_LLAMA_URL].rstrip("/")
        scan_interval = entry.options.get(
            CONF_SCAN_INTERVAL,
            entry.data.get(CONF_SCAN_INTERVAL, DEFAULT_SCAN_INTERVAL),
        )
        super().__init__(hass, entry, f"llamacpp {self.base_url}", int(scan_interval))

    async def _async_update_data(self) -> dict[str, Any]:
        base = self.base_url
        # /metrics is required: without it there is nothing to show, so a
        # failure here marks the whole device unavailable.
        metrics_text = await self._async_get_text(base + PATH_METRICS)
        metrics, metric_names = parse_prometheus(metrics_text)

        slots_raw, props, health = await asyncio.gather(
            self._async_get_json_optional(base + PATH_SLOTS),
            self._async_get_json_optional(base + PATH_PROPS),
            self._async_get_json_optional(base + PATH_HEALTH),
        )
        slots = slots_raw if isinstance(slots_raw, list) else []

        return {
            "metrics": metrics,
            "metric_names": metric_names,
            "slots": slots,
            "props": props or {},
            "health": (health or {}).get("status"),
        }


class GpuCoordinator(_BaseCoordinator):
    """Poll a GPU exporter (single JSON object at the base URL)."""

    def __init__(self, hass: HomeAssistant, entry: ConfigEntry) -> None:
        self.gpu_url = entry.data[CONF_GPU_URL].rstrip("/")
        scan_interval = entry.options.get(
            CONF_SCAN_INTERVAL,
            entry.data.get(CONF_SCAN_INTERVAL, DEFAULT_SCAN_INTERVAL),
        )
        super().__init__(hass, entry, f"gpu {self.gpu_url}", int(scan_interval))

    async def _async_update_data(self) -> dict[str, Any]:
        try:
            async with self.session.get(self.gpu_url, timeout=CLIENT_TIMEOUT) as resp:
                resp.raise_for_status()
                data = await resp.json()
        except (TimeoutError, aiohttp.ClientError) as err:
            raise UpdateFailed(f"Error fetching GPU metrics: {err}") from err
        if not isinstance(data, dict):
            raise UpdateFailed("Unexpected GPU metrics payload")
        return data


def create_coordinators(hass: HomeAssistant, entry: ConfigEntry) -> dict[str, Any]:
    """Build the coordinators for an entry (stored on ``entry.runtime_data``)."""
    data = entry.data
    return {
        "llama": LlamaCPPCoordinator(hass, entry),
        "gpu": GpuCoordinator(hass, entry) if data.get(CONF_GPU_URL) else None,
    }


def host_of(entry: ConfigEntry) -> str:
    """The llama.cpp host (without scheme) used for device identification."""
    return _host_from_url(entry.data[CONF_LLAMA_URL])


def gpu_host_of(entry: ConfigEntry) -> str:
    """The GPU host (without scheme) used for device identification."""
    return _host_from_url(entry.data.get(CONF_GPU_URL, ""))


def _host_from_url(url: str) -> str:
    """Reduce a base URL to ``host:port`` for stable device identifiers."""
    url = url.split("://", 1)[-1]
    return url.split("/", 1)[0]


__all__ = [
    "GpuCoordinator",
    "LlamaCPPCoordinator",
    "create_coordinators",
    "gpu_host_of",
    "host_of",
    "parse_prometheus",
    "split_metric_key",
]
