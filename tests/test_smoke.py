"""Smoke test: the integration loads and produces a metric sensor."""

from tests.conftest import get_state


async def test_loaded_creates_metric_sensor(hass, loaded_entry):
    state = get_state(hass, "sensor", "llamacpp:prompt_tokens_seconds")
    assert state is not None
    assert float(state.state) == 1249.65
    assert state.attributes.get("unit_of_measurement") == "tokens/s"
