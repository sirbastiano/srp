FROM ubuntu:22.04 as base

RUN apt-get update && apt-get install -y openjdk-8-jdk && rm -rf /var/lib/apt/lists/*

FROM base as build

LABEL authors="Roberto Del Prete"
LABEL maintainer="roberto.delprete@esa.int"

USER root

RUN apt-get update && apt-get install -y \
    wget \
    unzip \
    python3 \
    python3-pip \
    git \
    vim \
    fontconfig \
    fonts-dejavu \
    && rm -rf /var/lib/apt/lists/*

ENV LC_ALL "en_US.UTF-8"
ENV LD_LIBRARY_PATH ".:/usr/lib/jvm/java-8-openjdk-amd64/jre/lib/amd64/server/:$LD_LIBRARY_PATH"
ENV JAVA_HOME "/usr/lib/jvm/java-8-openjdk-amd64"

# Download and install SNAP directly
RUN wget -q https://download.esa.int/step/snap/12.0/installers/esa-snap_all_linux-12.0.0.sh -O /tmp/snap_installer.sh && \
    chmod +x /tmp/snap_installer.sh && \
    /tmp/snap_installer.sh -q -dir /usr/local/snap && \
    rm /tmp/snap_installer.sh

FROM base as snappy

RUN apt-get update && apt-get install -y \
    python3 \
    python3-pip \
    python3-dev \
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
    && rm -rf /var/lib/apt/lists/*

ENV LD_LIBRARY_PATH ".:$LD_LIBRARY_PATH"
ENV JAVA_HOME "/usr/lib/jvm/java-8-openjdk-amd64"
COPY --from=build /usr/local/snap /usr/local/snap

# Copy project files
COPY Makefile /workspace/
COPY pdm.lock /workspace/
COPY pyproject.toml /workspace/
COPY sarpyx /workspace/sarpyx/

WORKDIR /workspace

# Install the package in editable mode
RUN pip install pdm 
RUN cd sarpyx && pdm install 

# add gpt to PATH
ENV PATH="${PATH}:/usr/local/snap/bin"

# Create symlink for python command
RUN ln -s /usr/bin/python3 /usr/bin/python