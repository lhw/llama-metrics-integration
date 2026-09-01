"""Binary sensor platform for the llama.cpp integration."""

from __future__ import annotations

from typing import Any

from homeassistant.components.binary_sensor import (
    BinarySensorDeviceClass,
    BinarySensorEntity,
)
from homeassistant.config_entries import ConfigEntry
from homeassistant.core import HomeAssistant
from homeassistant.helpers.entity import EntityCategory
from homeassistant.helpers.update_coordinator import CoordinatorEntity

from .coordinator import LlamaCPPCoordinator, host_of
from .sensor import _main_device_info, _slot_device_info


class LlamaSleepingBinary(CoordinatorEntity[LlamaCPPCoordinator], BinarySensorEntity):
    """Whether the llama.cpp server is currently sleeping / idle."""

    _attr_has_entity_name = True
    _attr_name = "Sleeping"
    _attr_device_class = BinarySensorDeviceClass.RUNNING
    _attr_entity_category = EntityCategory.DIAGNOSTIC

    def __init__(
        self, coordinator: LlamaCPPCoordinator, device_info: dict[str, Any]
    ) -> None:
        super().__init__(coordinator)
        self._attr_unique_id = "sleeping"
        self._attr_device_info = device_info

    @property
    def is_on(self) -> bool:
        return bool(self.coordinator.data.get("props", {}).get("is_sleeping", False))


class SlotBinary(CoordinatorEntity[LlamaCPPCoordinator], BinarySensorEntity):
    """A boolean field from a single llama.cpp slot (subdevice)."""

    _attr_has_entity_name = True
    _attr_entity_category = EntityCategory.DIAGNOSTIC

    def __init__(
        self,
        coordinator: LlamaCPPCoordinator,
        device_info: dict[str, Any],
        slot_id: Any,
        attr: str,
        name: str,
        device_class: BinarySensorDeviceClass,
    ) -> None:
        super().__init__(coordinator)
        self._slot_id = slot_id
        self._attr = attr
        self._attr_name = name
        self._attr_device_class = device_class
        self._attr_unique_id = f"slot-{slot_id}:{attr}"
        self._attr_device_info = device_info

    @property
    def is_on(self) -> bool:
        for slot in self.coordinator.data.get("slots", []):
            if slot.get("id") == self._slot_id:
                return bool(slot.get(self._attr))
        return False


async def async_setup_entry(
    hass: HomeAssistant, entry: ConfigEntry, async_add_entities
) -> None:
    """Create the model-sleeping and per-slot binary sensors."""
    runtime = entry.runtime_data
    llama: LlamaCPPCoordinator = runtime["llama"]
    data = llama.data
    props = data.get("props", {})
    host = host_of(entry)
    main_info = _main_device_info(host, props, llama.base_url)

    entities: list[BinarySensorEntity] = [
        LlamaSleepingBinary(llama, main_info),
    ]

    for slot in data.get("slots", []):
        slot_id = slot.get("id")
        slot_info = _slot_device_info(host, slot_id)
        if "is_processing" in slot:
            entities.append(
                SlotBinary(
                    llama,
                    slot_info,
                    slot_id,
                    "is_processing",
                    "Active",
                    BinarySensorDeviceClass.RUNNING,
                )
            )
        if "speculative" in slot:
            entities.append(
                SlotBinary(
                    llama,
                    slot_info,
                    slot_id,
                    "speculative",
                    "Speculative",
                    BinarySensorDeviceClass.RUNNING,
                )
            )

    async_add_entities(entities)
