# llama.cpp GPU exporter (headless)

A tiny container that shells out to `nvidia-smi` and serves the GPU stats as
JSON on `:9091` — the exact payload the `llamacpp` Home Assistant integration
reads (see `custom_components/llamacpp`). No dashboard, no UI, no third-party
Python dependencies.

## Build

```sh
podman build -t llamacpp-gpu-exporter gpu-exporter/
```

## Run (podman + NVIDIA)

The `nvidia-container-toolkit` injects the host driver's `nvidia-smi` and
libraries into the container when you pass `--gpus all` — so the image stays a
plain `python:3.14-slim` (~50 MB) instead of a multi-GB CUDA image.

On SELinux-enforcing hosts (e.g. Bazzite/ubiBlue, Fedora, RHEL) you also need
`--security-opt label=disable`, otherwise `nvidia-smi` fails with
`Failed to initialize NVML: Insufficient Permissions`.

```sh
podman run -d \
  --name llamacpp-gpu-exporter \
  --gpus all \
  --security-opt label=disable \
  -p 9091:9091 \
  llamacpp-gpu-exporter
```

On a non-SELinux host (Ubuntu, most Docker setups) drop the
`--security-opt label=disable` flag.

## Verify

```sh
curl http://localhost:9091/
# {"gpu_utilization": 0.0, "memory_used_mb": 23512.0, "gpu_name": "NVIDIA GeForce RTX 4090", ...}
```

Then point the HA integration's "GPU exporter URL" at
`http://<host>:9091`.

## Requirements

- `podman` (5.3+ for `--gpus`) and the `nvidia-container-toolkit` installed on
  the host.
- An NVIDIA GPU with a working driver (`nvidia-smi` on the host).
