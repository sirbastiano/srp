FROM ubuntu:22.04 AS base

ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=Etc/UTC

ENV JAVA_HOME="/usr/lib/jvm/java-8-openjdk-amd64"
ENV LD_LIBRARY_PATH=".:/usr/lib/jvm/java-8-openjdk-amd64/jre/lib/amd64/server/"

RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates \
    software-properties-common \
    && \
    add-apt-repository -y ppa:deadsnakes/ppa && \
    apt-get update && apt-get install -y --no-install-recommends \
    openjdk-8-jdk \
    python3.11 \
    python3.11-venv \
    python3.11-dev \
    python3.11-distutils \
    && rm -rf /var/lib/apt/lists/*

FROM base AS build

LABEL authors="Roberto Del Prete"
LABEL maintainer="roberto.delprete@esa.int"

USER root

RUN apt-get update && apt-get install -y --no-install-recommends \
    wget \
    unzip \
    git \
    vim \
    fontconfig \
    fonts-dejavu \
    && rm -rf /var/lib/apt/lists/*

ENV LC_ALL="en_US.UTF-8"
ARG SNAP_VERSION=12.0.0

# Download and install SNAP directly
RUN wget -q "https://download.esa.int/step/snap/12.0/installers/esa-snap_all_linux-${SNAP_VERSION}.sh" -O /tmp/snap_installer.sh && \
    chmod +x /tmp/snap_installer.sh && \
    /tmp/snap_installer.sh -q -dir /usr/local/snap && \
    rm /tmp/snap_installer.sh

FROM base AS jupyter-ready

RUN apt-get update && apt-get install -y --no-install-recommends \
    fonts-dejavu \
    git \
    build-essential \
    pkg-config \
    libzstd-dev \
    gcc \
    g++ \
    clang \
    make \
    cmake \
    libc6-dev \
    curl \
    && rm -rf /var/lib/apt/lists/*

ENV PATH="${PATH}:/usr/local/snap/bin"
ENV PIP_DISABLE_PIP_VERSION_CHECK=1
ENV PIP_NO_CACHE_DIR=1

COPY --from=build /usr/local/snap /usr/local/snap

# Install pip for Python 3.11
RUN curl -sS https://bootstrap.pypa.io/get-pip.py -o /tmp/get-pip.py && \
    python3.11 /tmp/get-pip.py && \
    rm /tmp/get-pip.py

# Create symlink for python command
RUN ln -sf /usr/bin/python3.11 /usr/bin/python3 && \
    ln -sf /usr/bin/python3.11 /usr/bin/python

# Copy project files
COPY Makefile /workspace/
COPY pdm.lock /workspace/
COPY pyproject.toml /workspace/
COPY sarpyx /workspace/sarpyx/

WORKDIR /workspace

# Install the package and Jupyter
RUN python3.11 -m pip install --no-cache-dir pdm jupyter jupyterlab ipykernel
RUN pdm install && pdm add lxml

# Install sarpyx as a Jupyter kernel
RUN python -m ipykernel install --user --name=sarpyx --display-name="SAR Python (sarpyx-12.0)"

# Create a startup script
RUN cat <<'EOF' > /usr/local/bin/start-jupyter.sh
#!/bin/bash
export JAVA_HOME="/usr/lib/jvm/java-8-openjdk-amd64"
export LD_LIBRARY_PATH=".:$LD_LIBRARY_PATH"
export PATH="${PATH}:/usr/local/snap/bin"
cd /workspace
jupyter lab --ip=0.0.0.0 --port=8888 --no-browser --allow-root --NotebookApp.token="" --NotebookApp.password=""
EOF

RUN chmod +x /usr/local/bin/start-jupyter.sh

EXPOSE 8888

CMD ["/usr/local/bin/start-jupyter.sh"]
