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
    PATH_METRICS,
    PATH_PROPS,
    PATH_SLOTS,
)

LOGGER = logging.getLogger(__package__)

# `sock_connect` is kept short so a suspended/offline host is detected quickly
# instead of waiting out the full response timeout on every probe.
CLIENT_TIMEOUT = aiohttp.ClientTimeout(total=10, sock_connect=3)

# A single prometheus sample line: `name{labels} value [timestamp]`.
_PROM_LINE = re.compile(
    r"^(?P<name>[a-zA-Z_:][a-zA-Z0-9_:]*)(?P<labels>\{[^}]*\})?\s+(?P<value>\S+)"
)


def parse_prometheus(text: str) -> dict[str, float]:
    """Parse prometheus text into ``{full_key: value}``.

    ``full_key`` is the metric name, or ``name{labels}`` when the sample carries
    labels, so each labelled series is kept as its own key.
    """
    metrics: dict[str, float] = {}
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
        metrics[name if labels is None else f"{name}{labels}"] = value
    return metrics


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
    """Shared aiohttp plumbing and offline backoff for both coordinators."""

    # A suspended/offline host is probed at most once per this many seconds.
    MAX_RETRY_AFTER = 300.0

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
        self._base_interval = float(scan_interval)
        self._fail_streak = 0

    def _next_retry_after(self) -> float:
        """Growing (capped) delay for the next probe while the host is unreachable."""
        self._fail_streak += 1
        delay = self._base_interval * 2 ** (self._fail_streak - 1)
        return min(self.MAX_RETRY_AFTER, delay)

    def _note_success(self) -> None:
        """Reset the backoff once the host is reachable again."""
        self._fail_streak = 0

    async def _async_get_text(self, url: str) -> str:
        try:
            async with self.session.get(url, timeout=CLIENT_TIMEOUT) as resp:
                resp.raise_for_status()
                text = await resp.text()
        except (TimeoutError, aiohttp.ClientError) as err:
            raise UpdateFailed(
                f"Error communicating with {url}: {err}",
                retry_after=self._next_retry_after(),
            ) from err
        self._note_success()
        return text

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
    """Poll a llama.cpp server: /metrics, /slots, /props."""

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
        metrics = parse_prometheus(metrics_text)

        slots_raw, props = await asyncio.gather(
            self._async_get_json_optional(base + PATH_SLOTS),
            self._async_get_json_optional(base + PATH_PROPS),
        )
        slots = slots_raw if isinstance(slots_raw, list) else []

        return {
            "metrics": metrics,
            "slots": slots,
            "props": props or {},
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
            raise UpdateFailed(
                f"Error fetching GPU metrics: {err}",
                retry_after=self._next_retry_after(),
            ) from err
        if not isinstance(data, dict):
            # Host is reachable but sent an unexpected payload; don't back off.
            raise UpdateFailed("Unexpected GPU metrics payload")
        self._note_success()
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
