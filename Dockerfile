# ===== Builder Stage =====
FROM ubuntu:22.04 AS builder

ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=Etc/UTC
SHELL ["/bin/bash", "-o", "pipefail", "-c"]

ARG SNAP_VERSION=12.0.0
ARG SNAP_SKIP_UPDATES=1
ENV JAVA_HOME="/usr/lib/jvm/java-8-openjdk-amd64"
ENV SNAP_HOME="/snap12"
ENV SNAP_SKIP_UPDATES="${SNAP_SKIP_UPDATES}"
ENV PATH="${PATH}:${SNAP_HOME}/bin"
ENV PIP_DISABLE_PIP_VERSION_CHECK=1
ENV PIP_NO_CACHE_DIR=1

# Install build dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates \
    software-properties-common \
    gnupg \
    gpg-agent \
    lsb-release \
    && add-apt-repository -y ppa:deadsnakes/ppa \
    && add-apt-repository -y ppa:openjdk-r/ppa \
    && apt-get update && apt-get install -y --no-install-recommends \
    python3.11 \
    python3.11-venv \
    python3.11-dev \
    curl \
    wget \
    git \
    build-essential \
    openjdk-8-jdk \
    libhdf5-dev \
    libxml2-dev \
    libxslt1-dev \
    libproj-dev \
    libgeos-dev \
    && rm -rf /var/lib/apt/lists/*

# Install SNAP
COPY support/snap-install.sh /tmp/snap-install.sh
COPY support/snap.varfile /tmp/snap.varfile
RUN chmod +x /tmp/snap-install.sh \
    && /tmp/snap-install.sh -v \
    && rm -f /tmp/snap-install.sh /tmp/snap.varfile \
    && rm -rf /var/lib/apt/lists/*

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
ENV PIP_DISABLE_PIP_VERSION_CHECK=1
ENV PIP_NO_CACHE_DIR=1

# Install only runtime dependencies (no build-essential, python3.11-dev, etc.)
RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates \
    python3.11 \
    curl \
    openjdk-8-jdk \
    gdal-bin \
    libgdal30 \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean \
    && rm -rf /tmp/* /var/tmp/*

# Create symlinks
RUN ln -sf /usr/bin/python3.11 /usr/bin/python3 && \
    ln -sf /usr/bin/python3.11 /usr/bin/python

# Install pip for Python 3.11 in runtime image
RUN curl -fsSL https://bootstrap.pypa.io/get-pip.py -o /tmp/get-pip.py && \
    python3.11 /tmp/get-pip.py && \
    rm -f /tmp/get-pip.py

WORKDIR /workspace

# Copy only essential files
COPY pyproject.toml ./
COPY sarpyx ./sarpyx
COPY tests ./tests  

# Install sarpyx in development mode and verify import
RUN python3.11 -m pip install --upgrade pip setuptools wheel && \
    python3.11 -m pip install -e . && \
    python3.11 -c "import six; print('six installed successfully')" && \
    python3.11 -m pip install --ignore-installed --no-cache-dir six python-dateutil && \
    python3.11 -c "import six, dateutil; print('runtime deps ok')" && \
    rm -rf /var/lib/apt/lists/* /var/cache/apt/* /tmp/* /var/tmp/*

# Copy entrypoint script
COPY entrypoint.sh /usr/local/bin/entrypoint.sh
RUN chmod +x /usr/local/bin/entrypoint.sh

ENTRYPOINT ["/usr/local/bin/entrypoint.sh"]
CMD ["/bin/bash"]
