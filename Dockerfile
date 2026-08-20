FROM pytorch/pytorch:2.8.0-cuda12.8-cudnn9-runtime@sha256:417bd75df6365104c283ea4c1651fb3530d9eb5a4c2fafa51943cff2a94e6385

LABEL org.opencontainers.image.source="https://github.com/anycloud-sh/anycloud-trainer"
LABEL org.opencontainers.image.description="Stateful LoRA training service powered by AnyCloud"
LABEL org.opencontainers.image.licenses="Apache-2.0"

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    HF_HOME=/cache/huggingface \
    HF_HUB_DISABLE_TELEMETRY=1 \
    TOKENIZERS_PARALLELISM=false \
    PORT=8088

WORKDIR /app

COPY pyproject.toml README.md LICENSE ./
COPY src ./src

RUN python -m pip install '.[gpu]' && \
    useradd --create-home --uid 10001 trainer && \
    mkdir -p /cache/huggingface && \
    chown -R trainer:trainer /cache /app

USER trainer

EXPOSE 8088

CMD ["python", "-m", "anycloud_trainer.main"]
