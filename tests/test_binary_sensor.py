"""Tests for the binary_sensor platform."""

from __future__ import annotations

from tests.conftest import get_state


async def test_sleeping_binary(hass, loaded_entry) -> None:
    state = get_state(hass, "binary_sensor", "sleeping")
    assert state is not None
    assert state.state == "off"


async def test_slot_active_binary(hass, loaded_entry) -> None:
    state = get_state(hass, "binary_sensor", "slot-0:is_processing")
    assert state is not None
    assert state.state == "on"


async def test_slot_speculative_binary(hass, loaded_entry) -> None:
    state = get_state(hass, "binary_sensor", "slot-0:speculative")
    assert state is not None
    assert state.state == "on"
