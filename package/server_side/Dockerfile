# syntax=docker/dockerfile:1

FROM python:3.12-slim-bookworm

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_CACHE_DIR=1 \
    HOME=/tmp

WORKDIR /app

RUN groupadd --gid 10001 runfold \
    && useradd --uid 10001 --gid runfold --no-create-home --shell /usr/sbin/nologin runfold \
    && install -d -o runfold -g runfold /var/lib/runfold

COPY requirements-dev.txt ./
RUN python -m pip install --no-cache-dir --requirement requirements-dev.txt

COPY --chown=runfold:runfold runfold_server ./runfold_server

USER 10001:10001

EXPOSE 8383

CMD ["python", "-m", "runfold_server", "serve", "--config", "/etc/runfold/config.yaml", "--workers", "1"]
