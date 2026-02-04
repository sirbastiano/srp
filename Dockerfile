FROM ubuntu:22.04 AS base

ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=Etc/UTC

RUN apt-get update && apt-get install -y software-properties-common && \
    add-apt-repository ppa:deadsnakes/ppa && \
    apt-get update && apt-get install -y \
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

RUN apt-get update && apt-get install -y \
    wget \
    unzip \
    git \
    vim \
    fontconfig \
    fonts-dejavu \
    && rm -rf /var/lib/apt/lists/*

ENV LC_ALL="en_US.UTF-8"
ENV LD_LIBRARY_PATH=".:/usr/lib/jvm/java-8-openjdk-amd64/jre/lib/amd64/server/"
ENV JAVA_HOME="/usr/lib/jvm/java-8-openjdk-amd64"

# Download and install SNAP directly
RUN wget -q https://download.esa.int/step/snap/12.0/installers/esa-snap_all_linux-12.0.0.sh -O /tmp/snap_installer.sh && \
    chmod +x /tmp/snap_installer.sh && \
    /tmp/snap_installer.sh -q -dir /usr/local/snap && \
    rm /tmp/snap_installer.sh

FROM base AS jupyter-ready

RUN apt-get update && apt-get install -y \
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

ENV LD_LIBRARY_PATH=".:/usr/lib/jvm/java-8-openjdk-amd64/jre/lib/amd64/server/"
ENV JAVA_HOME="/usr/lib/jvm/java-8-openjdk-amd64"
ENV PATH="${PATH}:/usr/local/snap/bin"

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
RUN python3.11 -m pip install pdm jupyter jupyterlab ipykernel
RUN pdm install && pdm add lxml

# Install sarpyx as a Jupyter kernel
RUN python -m ipykernel install --user --name=sarpyx --display-name="SAR Python (sarpyx-12.0)"

# Create a startup script
RUN echo '#!/bin/bash\n\
export JAVA_HOME="/usr/lib/jvm/java-8-openjdk-amd64"\n\
export LD_LIBRARY_PATH=".:$LD_LIBRARY_PATH"\n\
export PATH="${PATH}:/usr/local/snap/bin"\n\
cd /workspace\n\
jupyter lab --ip=0.0.0.0 --port=8888 --no-browser --allow-root --NotebookApp.token="" --NotebookApp.password=""' > /usr/local/bin/start-jupyter.sh

RUN chmod +x /usr/local/bin/start-jupyter.sh

EXPOSE 8888

CMD ["/usr/local/bin/start-jupyter.sh"]
