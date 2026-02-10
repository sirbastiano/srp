FROM ubuntu:24.04

ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=Etc/UTC
SHELL ["/bin/bash", "-o", "pipefail", "-c"]

ARG SNAP_VERSION=12.0.0
ARG SNAP_SKIP_UPDATES=1
ENV JAVA_HOME="/usr/lib/jvm/java-8-openjdk-amd64"
ENV SNAP_HOME="/usr/local/snap"
ENV SNAP_SKIP_UPDATES="${SNAP_SKIP_UPDATES}"
ENV PATH="${PATH}:${SNAP_HOME}/bin"
ENV PIP_DISABLE_PIP_VERSION_CHECK=1
ENV PIP_NO_CACHE_DIR=1

# Install only essential packages
RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates \
    software-properties-common \
    && add-apt-repository ppa:ubuntugis/ubuntugis-unstable \
    && apt-get update && apt-get install -y --no-install-recommends \
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
    libhdf5-dev \
    libxml2-dev \
    libxslt1-dev \
    libproj-dev \
    libgeos-dev \
    && rm -rf /var/lib/apt/lists/*

# Set GDAL environment variables
ENV GDAL_CONFIG=/usr/bin/gdal-config

# Install SNAP
COPY support/snap-install.sh /tmp/snap-install.sh
COPY support/snap.varfile /tmp/snap.varfile
RUN chmod +x /tmp/snap-install.sh && /tmp/snap-install.sh -v && rm -f /tmp/snap-install.sh /tmp/snap.varfile

# RUN wget -q "https://download.esa.int/step/snap/12.0/installers/esa-snap_all_linux-${SNAP_VERSION}.sh" -O /tmp/snap_installer.sh && \
#     chmod +x /tmp/snap_installer.sh && \
#     /tmp/snap_installer.sh -q -dir "${SNAP_HOME}" && \
#     rm -f /tmp/snap_installer.sh

# Install pip
RUN curl -fsSL https://bootstrap.pypa.io/get-pip.py -o /tmp/get-pip.py && \
    python3.11 /tmp/get-pip.py && \
    rm -f /tmp/get-pip.py

# Create symlinks
RUN ln -sf /usr/bin/python3.11 /usr/bin/python3 && \
    ln -sf /usr/bin/python3.11 /usr/bin/python

WORKDIR /workspace

# Copy only essential files
COPY pyproject.toml ./
COPY sarpyx ./sarpyx
COPY tests ./tests

# Install sarpyx in development mode and verify import
RUN python3.11 -m pip install --upgrade pip setuptools wheel && \
    python3.11 -m pip install numpy && \
    GDAL_VERSION=$(gdal-config --version) && \
    python3.11 -m pip install GDAL==${GDAL_VERSION} && \
    python3.11 -m pip install -e . && \
    python3.11 -c "import sarpyx; print('sarpyx installed successfully')"

# Copy and set up entrypoint script
COPY entrypoint.sh /usr/local/bin/entrypoint.sh
RUN chmod +x /usr/local/bin/entrypoint.sh

ENTRYPOINT ["/usr/local/bin/entrypoint.sh"]
CMD ["python3.11", "-c", "import sarpyx; print('sarpyx version:', getattr(sarpyx, '__version__', 'unknown'))"]
