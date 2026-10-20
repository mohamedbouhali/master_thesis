FROM python:3.10

ENV PYTHONUNBUFFERED=1

WORKDIR /app

RUN --mount=type=cache,target=/root/.cache/pip \
    python -m pip install --upgrade pip && \
    python -m pip install agentrl-worker && \
    # Pin aiodocker to a version whose stream timeout handling is compatible
    # with the installed aiohttp. aiodocker 0.27 passes a raw int where aiohttp
    # expects a ClientTimeout, raising "'int' object has no attribute 'connect'"
    # when executing commands in sandbox containers.
    python -m pip install "aiodocker==0.24.0"

COPY ./src/server/tasks/os_interaction /app/src/server/tasks/os_interaction
COPY ./configs/tasks/os.yaml /app/configs/tasks/os.yaml
COPY ./data/os_interaction /app/data/os_interaction

COPY --chmod=0755 ./extra/worker-entrypoint.sh /entrypoint.sh

ENTRYPOINT ["/entrypoint.sh", "-c", "configs/tasks/os.yaml"]
