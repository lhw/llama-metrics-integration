"""Sensor platform for the llama.cpp integration."""

from __future__ import annotations

from typing import Any

from homeassistant.components.sensor import SensorEntity
from homeassistant.config_entries import ConfigEntry
from homeassistant.core import HomeAssistant
from homeassistant.helpers.entity import EntityCategory
from homeassistant.helpers.update_coordinator import CoordinatorEntity

from .const import (
    DOMAIN,
    GPU_METRICS,
    LLAMACPP_METRICS,
    MetricDef,
    humanize_metric,
)
from .coordinator import (
    GpuCoordinator,
    LlamaCPPCoordinator,
    gpu_host_of,
    host_of,
    split_metric_key,
)

# Flat per-slot numeric fields: (attr, name, unit). Entities are only created
# for fields the server actually returns on the first scrape.
SLOT_SENSOR_DEFS: list[tuple[str, str, str | None]] = [
    ("n_ctx", "Context size", None),
    ("n_prompt_tokens", "Prompt tokens", None),
    ("n_prompt_tokens_processed", "Prompt tokens processed", None),
    ("n_prompt_tokens_cache", "Prompt tokens cached", None),
    ("n_tokens_predicted", "Tokens predicted", None),
    ("id_task", "Task ID", None),
]


def _presentation(definition: MetricDef | None) -> tuple[EntityCategory | None, bool]:
    """Return ``(entity_category, enabled_by_default)`` for a metric definition.

    Front-and-center metrics are surfaced; statistical metrics that are not are
    hidden (diagnostic) and disabled by default so they don't spam state. Static
    (non-statistical) metrics stay available.
    """
    if definition is None:
        return EntityCategory.DIAGNOSTIC, True
    front = definition.front
    category = None if front else EntityCategory.DIAGNOSTIC
    enabled = front or definition.state_class is None
    return category, enabled


def _main_device_info(
    host: str, props: dict[str, Any], base_url: str
) -> dict[str, Any]:
    return {
        "identifiers": {(DOMAIN, host)},
        "name": f"llama.cpp {host}",
        "manufacturer": "llama.cpp",
        "model": props.get("model_alias"),
        "sw_version": props.get("build_info"),
        "configuration_url": base_url or None,
    }


def _slot_device_info(host: str, slot_id: Any) -> dict[str, Any]:
    return {
        "identifiers": {(DOMAIN, f"{host}/slot-{slot_id}")},
        "name": f"Slot {slot_id}",
        "via_device": (DOMAIN, host),
    }


def _gpu_device_info(gpu_host: str, data: dict[str, Any]) -> dict[str, Any]:
    name = data.get("gpu_name")
    return {
        "identifiers": {(DOMAIN, f"{gpu_host}/gpu")},
        "name": name or f"GPU {gpu_host}",
        "model": data.get("pci_bus_id"),
    }


class LlamaMetricSensor(CoordinatorEntity[LlamaCPPCoordinator], SensorEntity):
    """A single prometheus metric series from the llama.cpp server."""

    _attr_has_entity_name = True

    def __init__(
        self,
        coordinator: LlamaCPPCoordinator,
        device_info: dict[str, Any],
        key: str,
    ) -> None:
        super().__init__(coordinator)
        name, labels = split_metric_key(key)
        definition = LLAMACPP_METRICS.get(name)
        base_name = (
            definition.name if definition and definition.name else humanize_metric(name)
        )
        if labels:
            suffix = ", ".join(f"{k} {v}" for k, v in labels.items())
            self._attr_name = f"{base_name} ({suffix})"
        else:
            self._attr_name = base_name
        self._attr_native_unit_of_measurement = definition.unit if definition else None
        self._attr_state_class = definition.state_class if definition else None
        self._attr_device_class = definition.device_class if definition else None
        category, enabled = _presentation(definition)
        self._attr_entity_category = category
        self.entity_registry_enabled_default = enabled
        self._key = key
        self._attr_unique_id = f"{key}"
        self._attr_device_info = device_info

    @property
    def native_value(self) -> float | None:
        return self.coordinator.data.get("metrics", {}).get(self._key)


class LlamaInfoSensor(CoordinatorEntity[LlamaCPPCoordinator], SensorEntity):
    """An informational field from llama.cpp /props."""

    _attr_has_entity_name = True
    _attr_entity_category = EntityCategory.DIAGNOSTIC

    def __init__(
        self,
        coordinator: LlamaCPPCoordinator,
        device_info: dict[str, Any],
        key: str,
        name: str,
    ) -> None:
        super().__init__(coordinator)
        self._key = key
        self._attr_name = name
        self._attr_unique_id = f"{key}"
        self._attr_device_info = device_info

    @property
    def native_value(self) -> Any:
        return self.coordinator.data.get("props", {}).get(self._key)


class SlotSensor(CoordinatorEntity[LlamaCPPCoordinator], SensorEntity):
    """A numeric field from a single llama.cpp slot (subdevice)."""

    _attr_has_entity_name = True
    _attr_entity_category = EntityCategory.DIAGNOSTIC

    def __init__(
        self,
        coordinator: LlamaCPPCoordinator,
        device_info: dict[str, Any],
        slot_id: Any,
        attr: str,
        name: str,
        unit: str | None,
    ) -> None:
        super().__init__(coordinator)
        self._slot_id = slot_id
        self._attr = attr
        self._attr_name = name
        self._attr_native_unit_of_measurement = unit
        self._attr_unique_id = f"slot-{slot_id}:{attr}"
        self._attr_device_info = device_info

    @property
    def native_value(self) -> Any:
        for slot in self.coordinator.data.get("slots", []):
            if slot.get("id") == self._slot_id:
                return slot.get(self._attr)
        return None


class GpuMetricSensor(CoordinatorEntity[GpuCoordinator], SensorEntity):
    """A single field from the GPU exporter."""

    _attr_has_entity_name = True

    def __init__(
        self,
        coordinator: GpuCoordinator,
        device_info: dict[str, Any],
        key: str,
    ) -> None:
        super().__init__(coordinator)
        definition = GPU_METRICS.get(key, MetricDef(name=humanize_metric(key)))
        self._attr_name = definition.name or humanize_metric(key)
        self._attr_native_unit_of_measurement = definition.unit
        self._attr_state_class = definition.state_class
        self._attr_device_class = definition.device_class
        category, enabled = _presentation(definition)
        self._attr_entity_category = category
        self.entity_registry_enabled_default = enabled
        self._key = key
        self._attr_unique_id = f"gpu:{key}"
        self._attr_device_info = device_info

    @property
    def native_value(self) -> Any:
        return self.coordinator.data.get(self._key)


async def async_setup_entry(
    hass: HomeAssistant, entry: ConfigEntry, async_add_entities
) -> None:
    """Create entities for every metric the server actually serves."""
    runtime = entry.runtime_data
    llama: LlamaCPPCoordinator = runtime["llama"]
    gpu: GpuCoordinator | None = runtime.get("gpu")

    data = llama.data
    props = data.get("props", {})
    host = host_of(entry)
    main_info = _main_device_info(host, props, llama.base_url)

    entities: list[SensorEntity] = []

    # 1. /metrics — one sensor per labelled series the server emitted.
    for key in sorted(data.get("metrics", {})):
        entities.append(LlamaMetricSensor(llama, main_info, key))

    # 2. /props — a few informational sensors.
    if props.get("model_alias") is not None:
        entities.append(LlamaInfoSensor(llama, main_info, "model_alias", "Model"))
    if props.get("model_ftype") is not None:
        entities.append(LlamaInfoSensor(llama, main_info, "model_ftype", "Model type"))
    if props.get("total_slots") is not None:
        entities.append(LlamaInfoSensor(llama, main_info, "total_slots", "Slots"))

    # 3. /slots — a subdevice per slot, with sensors for the fields it serves.
    for slot in data.get("slots", []):
        slot_id = slot.get("id")
        slot_info = _slot_device_info(host, slot_id)
        for attr, name, unit in SLOT_SENSOR_DEFS:
            if attr in slot:
                entities.append(SlotSensor(llama, slot_info, slot_id, attr, name, unit))

    # 4. GPU — separate device when configured.
    if gpu is not None:
        gpu_host = gpu_host_of(entry)
        gpu_info = _gpu_device_info(gpu_host, gpu.data)
        for key in GPU_METRICS:
            entities.append(GpuMetricSensor(gpu, gpu_info, key))

    async_add_entities(entities)
