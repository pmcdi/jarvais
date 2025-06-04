ARG PYTHON_VERSION=3.12
FROM python:${PYTHON_VERSION}-slim

# Add uv binary
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /usr/local/bin/

# Set working directory inside container
WORKDIR /app

# Copy local source code into the image
COPY . /app

# Install local package in editable mode
ENV UV_SYSTEM_PYTHON=1
RUN uv pip install --system --no-cache --editable .

# Metadata
ARG VERSION
ARG GIT_COMMIT
ARG BUILD_DATE

LABEL maintainer="PM Cancer Digital Intelligence " \
    license="MIT" \
    usage="docker run -it --rm <image_name> jarvais --help" \
    org.opencontainers.image.title="JARVAIS" \
    org.opencontainers.image.description="Standardized ML pipelines for oncology" \
    org.opencontainers.image.url="https://github.com/pmcdi/jarvais" \
    org.opencontainers.image.source="https://github.com/pmcdi/jarvais" \
    org.opencontainers.image.version="${VERSION}" \
    org.opencontainers.image.revision="${GIT_COMMIT}" \
    org.opencontainers.image.created="${BUILD_DATE}" \
    org.opencontainers.image.authors="Joshua Siraj"

RUN rm -rf /var/lib/apt/lists/* && rm -rf /tmp/* && rm -rf /var/tmp/*

RUN jarvais --help
CMD ["/bin/bash"]
