"""Tests for diagnostics redaction."""

from __future__ import annotations

from custom_components.llamacpp.diagnostics import (
    async_get_config_entry_diagnostics,
)


async def test_diagnostics_redacts_model_path(hass, loaded_entry) -> None:
    data = await async_get_config_entry_diagnostics(hass, loaded_entry)
    props = data["llama"]["data"]["props"]
    # Full filesystem path is masked to its basename.
    assert props["model_path"] == "TestModel-8B.gguf"
    assert "deadbeef" not in str(data)
    # Non-sensitive props survive.
    assert props["model_alias"] == "TestModel-8B"
    # Coordinator liveness is included.
    assert data["llama"]["last_update_success"] is True
    assert data["gpu"]["data"]["gpu_name"] == "NVIDIA GeForce RTX 4090"
