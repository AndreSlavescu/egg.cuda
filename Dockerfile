FROM nvidia/cuda:13.0.2-cudnn-devel-ubuntu24.04

ENV DEBIAN_FRONTEND=noninteractive
ENV NVCC=nvcc

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

RUN echo 'PS1="\[\033[1;36m\]egg-cuda\[\033[0m\]:\[\033[1;34m\]\w\[\033[0m\]\$ "' >> /root/.bashrc && \
    echo 'alias ls="ls --color=auto"' >> /root/.bashrc && \
    echo 'alias ll="ls -la"' >> /root/.bashrc && \
    echo 'alias grep="grep --color=auto"' >> /root/.bashrc && \
    echo 'export TERM=xterm-256color' >> /root/.bashrc

WORKDIR /app

COPY full_trained_egg.cu .
COPY int8_tc.cuh .
COPY Makefile .
COPY scripts/ ./scripts/
COPY train_gpu.sh .
COPY input.txt .

RUN chmod +x train_gpu.sh

CMD ["/bin/bash"]

