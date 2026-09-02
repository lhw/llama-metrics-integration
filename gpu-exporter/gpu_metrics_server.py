"""Headless GPU metrics exporter.

Serves a snapshot of nvidia-smi statistics as a JSON document on :9091, the
payload the ``llamacpp`` Home Assistant integration polls. Standard library
only; expects ``nvidia-smi`` on the container PATH (the
nvidia-container-toolkit provides it at runtime).
"""

import json
import logging
import subprocess
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

HOST = "0.0.0.0"
PORT = 9091

# (json_key, nvidia-smi field). The --query-gpu list is built in this order,
# so nvidia-smi's CSV columns line up 1:1 with the keys below.
FIELDS: tuple[tuple[str, str], ...] = (
    ("gpu_utilization", "utilization.gpu"),
    ("memory_utilization", "utilization.memory"),
    ("memory_used_mb", "memory.used"),
    ("memory_total_mb", "memory.total"),
    ("temperature_c", "temperature.gpu"),
    ("power_draw_w", "power.draw"),
    ("power_limit_w", "power.limit"),
    ("sm_clock_mhz", "clocks.current.sm"),
    ("sm_clock_max_mhz", "clocks.max.sm"),
    ("fan_speed", "fan.speed"),
    ("gpu_name", "name"),
    ("pci_bus_id", "pci.bus_id"),
)

# nvidia-smi values that are not always numeric (fan speed can be "[N/A]"),
# passed through as strings.
TEXT_KEYS = frozenset({"fan_speed", "gpu_name", "pci_bus_id"})

log = logging.getLogger("gpu-metrics")


def _to_float(raw: str) -> float:
    try:
        return float(raw)
    except ValueError as exc:
        raise RuntimeError(f"non-numeric GPU value {raw!r}") from exc


def read_gpu() -> dict[str, object]:
    """Return one GPU's metrics as a ``{json_key: value}`` mapping."""
    query = ",".join(smi_field for _, smi_field in FIELDS)
    proc = subprocess.run(
        ["nvidia-smi", f"--query-gpu={query}", "--format=csv,noheader,nounits"],
        capture_output=True,
        text=True,
        timeout=5,
    )
    if proc.returncode != 0:
        raise RuntimeError(
            f"nvidia-smi exited {proc.returncode}: {proc.stderr.strip()}"
        )

    values = [part.strip() for part in proc.stdout.strip().split(",")]
    if len(values) != len(FIELDS):
        raise RuntimeError(
            f"expected {len(FIELDS)} GPU fields, got {len(values)} "
            "(multi-GPU not supported)"
        )

    return {
        key: (raw if key in TEXT_KEYS else _to_float(raw))
        for (key, _), raw in zip(FIELDS, values)
    }


class Handler(BaseHTTPRequestHandler):
    server_version = "GpuMetrics/1.0"

    def do_GET(self) -> None:
        try:
            payload: dict[str, object] = read_gpu()
            payload["timestamp"] = time.time()
            body = json.dumps(payload).encode()
        except Exception as exc:  # a bad scrape is still a JSON answer
            log.warning("GPU scrape failed: %s", exc)
            body = json.dumps({"error": str(exc)}).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, format: str, *args: object) -> None:
        log.debug(format, *args)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    server = ThreadingHTTPServer((HOST, PORT), Handler)

    log.info("serving GPU metrics on http://%s:%d/", HOST, PORT)
    try:
        server.serve_forever()
    finally:
        server.server_close()
        log.info("stopped")


if __name__ == "__main__":
    main()
