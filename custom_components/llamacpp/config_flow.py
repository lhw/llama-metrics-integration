"""Config flow for the llama.cpp integration."""

from __future__ import annotations

import logging
from typing import Any
from urllib.parse import urlparse

import aiohttp
import voluptuous as vol
from homeassistant.config_entries import (
    ConfigEntry,
    ConfigFlow,
    OptionsFlowWithReload,
)
from homeassistant.data_entry_flow import FlowResult
from homeassistant.helpers.aiohttp_client import async_get_clientsession

from .const import (
    CONF_GPU_URL,
    CONF_LLAMA_URL,
    CONF_SCAN_INTERVAL,
    DEFAULT_SCAN_INTERVAL,
    DOMAIN,
    MIN_SCAN_INTERVAL,
    PATH_METRICS,
)
from .coordinator import _host_from_url

LOGGER = logging.getLogger(__package__)

PROBE_TIMEOUT = aiohttp.ClientTimeout(total=10)


def _normalize_url(value: str | None) -> str | None:
    """Strip whitespace and default a bare ``host:port`` to ``http://``."""
    value = (value or "").strip()
    if not value:
        return None
    if "://" not in value:
        value = "http://" + value
    return value


def _is_valid_url(url: str) -> bool:
    parsed = urlparse(url)
    return (
        parsed.scheme in ("http", "https") and bool(parsed.hostname) and " " not in url
    )


def _schema(defaults: dict[str, Any]) -> vol.Schema:
    return vol.Schema(
        {
            vol.Required(CONF_LLAMA_URL, default=defaults.get(CONF_LLAMA_URL, "")): str,
            vol.Required(CONF_GPU_URL, default=defaults.get(CONF_GPU_URL, "")): str,
            vol.Required(
                CONF_SCAN_INTERVAL,
                default=defaults.get(CONF_SCAN_INTERVAL, DEFAULT_SCAN_INTERVAL),
            ): vol.All(vol.Coerce(int), vol.Range(min=MIN_SCAN_INTERVAL)),
        }
    )


async def _validate_reachable(hass, data: dict[str, Any]) -> dict[str, str]:
    """Validate URL format and probe that each supplied endpoint is reachable."""
    errors: dict[str, str] = {}
    for key in (CONF_LLAMA_URL, CONF_GPU_URL):
        url = data.get(key)
        if not url:
            continue
        if not _is_valid_url(url):
            errors[key] = "invalid"
    if errors:
        return errors

    session = async_get_clientsession(hass)
    for key in (CONF_LLAMA_URL, CONF_GPU_URL):
        url = data.get(key)
        if not url:
            continue
        probe = url.rstrip("/") + (PATH_METRICS if key == CONF_LLAMA_URL else "")
        try:
            async with session.get(probe, timeout=PROBE_TIMEOUT) as resp:
                await resp.read()
        except (TimeoutError, aiohttp.ClientError):
            errors[key] = "cannot_connect"
        except Exception:
            LOGGER.exception("Unexpected error probing %s", url)
            errors[key] = "unknown"
    return errors


def _clean_input(user_input: dict[str, Any]) -> dict[str, Any]:
    data = dict(user_input)
    data[CONF_LLAMA_URL] = _normalize_url(data[CONF_LLAMA_URL])
    data[CONF_GPU_URL] = _normalize_url(data.get(CONF_GPU_URL))
    return data


class LlamaConfigFlow(ConfigFlow, domain=DOMAIN):
    """Handle the llama.cpp config flow."""

    async def async_step_user(
        self, user_input: dict[str, Any] | None = None
    ) -> FlowResult:
        errors: dict[str, str] = {}
        if user_input is not None:
            data = _clean_input(user_input)
            errors = await _validate_reachable(self.hass, data)
            if not errors:
                await self.async_set_unique_id(data[CONF_LLAMA_URL])
                self._abort_if_unique_id_configured(error="unique_already_configured")
                return self.async_create_entry(
                    title=_host_from_url(data[CONF_LLAMA_URL]), data=data
                )
        return self.async_show_form(
            step_id="user", data_schema=_schema({}), errors=errors
        )

    @staticmethod
    def async_get_options_flow(config_entry: ConfigEntry) -> LlamaOptionsFlow:
        return LlamaOptionsFlow()


class LlamaOptionsFlow(OptionsFlowWithReload):
    """Handle the llama.cpp options flow (GPU URL + scan interval)."""

    async def async_step_init(
        self, user_input: dict[str, Any] | None = None
    ) -> FlowResult:
        errors: dict[str, str] = {}
        if user_input is not None:
            data = _clean_input(user_input)
            errors = await _validate_reachable(self.hass, data)
            if not errors:
                return self.async_create_entry(data=data)
        entry = self.config_entry
        defaults = {
            CONF_LLAMA_URL: entry.data.get(CONF_LLAMA_URL, ""),
            CONF_GPU_URL: entry.data.get(CONF_GPU_URL) or "",
            CONF_SCAN_INTERVAL: (entry.options or entry.data).get(
                CONF_SCAN_INTERVAL, DEFAULT_SCAN_INTERVAL
            ),
        }
        return self.async_show_form(
            step_id="init", data_schema=_schema(defaults), errors=errors
        )
