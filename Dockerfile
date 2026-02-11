# ===== Builder Stage =====
FROM ubuntu:22.04 AS builder

ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=Etc/UTC
SHELL ["/bin/bash", "-o", "pipefail", "-c"]

ARG SNAP_VERSION=12.0.0
ARG SNAP_SKIP_UPDATES=1
ENV JAVA_HOME="/usr/lib/jvm/java-8-openjdk-amd64"
ENV SNAP_HOME="/usr/local/snap"
ENV PIP_DISABLE_PIP_VERSION_CHECK=1
ENV PIP_NO_CACHE_DIR=1

# Install build dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates \
    python3.11 \
    python3.11-venv \
    python3.11-dev \
    curl \
    wget \
    git \
    build-essential \
    openjdk-8-jdk \
    gdal-bin \
    libgdal-dev \
    && rm -rf /var/lib/apt/lists/*

# Install SNAP
COPY support/snap-install.sh /tmp/snap-install.sh
COPY support/snap.varfile /tmp/snap.varfile
RUN chmod +x /tmp/snap-install.sh && /tmp/snap-install.sh -v && rm -f /tmp/snap-install.sh /tmp/snap.varfile

# Install pip
RUN curl -fsSL https://bootstrap.pypa.io/get-pip.py -o /tmp/get-pip.py && \
    python3.11 /tmp/get-pip.py && \
    rm -f /tmp/get-pip.py

WORKDIR /workspace

# Copy and build sarpyx package
COPY pyproject.toml ./
COPY sarpyx ./sarpyx
RUN python3.11 -m pip install --no-cache-dir . && \
    python3.11 -c "import sarpyx; print('sarpyx installed successfully')"

# ===== Runtime Stage =====
FROM ubuntu:22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=Etc/UTC
SHELL ["/bin/bash", "-o", "pipefail", "-c"]

ARG SNAP_SKIP_UPDATES=1
ENV JAVA_HOME="/usr/lib/jvm/java-8-openjdk-amd64"
ENV SNAP_HOME="/usr/local/snap"
ENV SNAP_SKIP_UPDATES="${SNAP_SKIP_UPDATES}"
ENV PATH="${PATH}:${SNAP_HOME}/bin"

# Install only runtime dependencies (no build-essential, python3.11-dev, etc.)
RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates \
    python3.11 \
    openjdk-8-jdk \
    gdal-bin \
    libgdal32 \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean \
    && rm -rf /tmp/* /var/tmp/*

# Create symlinks
RUN ln -sf /usr/bin/python3.11 /usr/bin/python3 && \
    ln -sf /usr/bin/python3.11 /usr/bin/python

WORKDIR /workspace

# Copy SNAP installation from builder
COPY --from=builder /usr/local/snap /usr/local/snap

# Copy Python site-packages from builder
COPY --from=builder /usr/local/lib/python3.11/dist-packages /usr/local/lib/python3.11/dist-packages

# Copy entrypoint script
COPY entrypoint.sh /usr/local/bin/entrypoint.sh
RUN chmod +x /usr/local/bin/entrypoint.sh

ENTRYPOINT ["/usr/local/bin/entrypoint.sh"]
CMD ["python3.11", "-c", "import sarpyx; print('sarpyx version:', getattr(sarpyx, '__version__', 'unknown'))"]
