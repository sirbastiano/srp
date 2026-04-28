# ===== Base Stage =====
FROM ubuntu:22.04 AS base

ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=Etc/UTC
ENV LANG=en_US.UTF-8
ENV LANGUAGE=en_US:en
ENV LC_ALL=en_US.UTF-8
SHELL ["/bin/bash", "-o", "pipefail", "-c"]

ARG SNAP_SKIP_UPDATES=1
ENV JAVA_HOME="/usr/lib/jvm/java-8-openjdk-amd64"
ENV SNAP_HOME="/workspace/snap13"
ENV SNAP_SKIP_UPDATES="${SNAP_SKIP_UPDATES}"
ENV PATH="${PATH}:${SNAP_HOME}/bin"
ENV PIP_DISABLE_PIP_VERSION_CHECK=1
ENV PIP_NO_CACHE_DIR=1

# Install shared system dependencies once for SNAP + Python runtime.
RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates \
    software-properties-common \
    gnupg \
    gpg-agent \
    lsb-release \
    && add-apt-repository -y ppa:deadsnakes/ppa \
    && add-apt-repository -y ppa:openjdk-r/ppa \
    && apt-get update && apt-get install -y --no-install-recommends \
    locales \
    python3.11 \
    python3.11-venv \
    python3.11-dev \
    curl \
    wget \
    git \
    build-essential \
    gfortran \
    openjdk-8-jdk \
    gdal-bin \
    libgdal30 \
    libfftw3-dev \
    libtiff5-dev \
    libgfortran5 \
    jblas \
    libhdf5-dev \
    libxml2-dev \
    libxslt1-dev \
    libproj-dev \
    libgeos-dev \
    && locale-gen en_US.UTF-8 \
    && update-locale LANG=en_US.UTF-8 LC_ALL=en_US.UTF-8 \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean \
    && rm -rf /tmp/* /var/tmp/*

# Create Python aliases expected by scripts and users.
RUN ln -sf /usr/bin/python3.11 /usr/bin/python3 && \
    ln -sf /usr/bin/python3.11 /usr/bin/python

# Install pip for Python 3.11.
RUN curl -fsSL https://bootstrap.pypa.io/get-pip.py -o /tmp/get-pip.py && \
    python3.11 /tmp/get-pip.py && \
    rm -f /tmp/get-pip.py

WORKDIR /workspace

# ===== SNAP Stage =====
FROM base AS snap

ARG SNAP_VERSION=13.0.0
ENV SNAP_HOME="/snap13"
ENV PATH="${PATH}:${SNAP_HOME}/bin"
ENV SNAP_SKIP_SYSTEM_PACKAGES=1

COPY support/snap-install.sh /tmp/snap-install.sh
COPY support/snap.varfile /tmp/snap.varfile
RUN SNAP_MAJOR="${SNAP_VERSION%%.*}" \
    && sed -i "s/^VERSION=.*/VERSION=${SNAP_MAJOR}/" /tmp/snap-install.sh \
    && sed -i "s|^sys.installationDir=.*|sys.installationDir=${SNAP_HOME}|" /tmp/snap.varfile \
    && chmod +x /tmp/snap-install.sh \
    && /tmp/snap-install.sh -v \
    && rm -f /tmp/snap-install.sh /tmp/snap.varfile

# ===== Runtime Stage =====
FROM base

# Bring SNAP installation from the dedicated SNAP stage into the final image.
COPY --from=snap /snap13 /workspace/snap13
RUN ln -sf /workspace/snap13/bin/snap /usr/local/bin/snap && \
    ln -sf /workspace/snap13/bin/gpt /usr/local/bin/gpt

COPY README.md pyproject.toml ./
COPY sarpyx ./sarpyx

# Install sarpyx once in the final image.
RUN python3.11 -m pip install --upgrade pip setuptools wheel && \
    python3.11 -m pip install --no-cache-dir . && \
    python3.11 -c "import sarpyx; print('sarpyx installed successfully')" && \
    rm -rf /tmp/* /var/tmp/*

COPY entrypoint.sh /usr/local/bin/entrypoint.sh
COPY start-jupyter.sh /usr/local/bin/start-jupyter.sh
RUN chmod +x /usr/local/bin/entrypoint.sh /usr/local/bin/start-jupyter.sh
ENTRYPOINT ["/usr/local/bin/entrypoint.sh"]
CMD ["/bin/bash"]
