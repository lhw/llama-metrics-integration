# Headless GPU metrics exporter for the llama.cpp integration.
# Serves nvidia-smi stats as JSON on :9091 (the payload the HA `llamacpp`
# integration reads). No UI, no third-party Python deps.
#
# nvidia-smi is NOT in this image on purpose: the nvidia-container-toolkit
# injects the host driver's nvidia-smi + libs at runtime when you pass
# `--gpus all`. See README.md for the exact podman run command.
FROM python:3.14-slim

WORKDIR /app
COPY gpu_metrics_server.py .
EXPOSE 9091
CMD ["python3", "gpu_metrics_server.py"]
