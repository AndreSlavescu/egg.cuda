FROM nvidia/cuda:13.0.2-cudnn-devel-ubuntu24.04

ENV DEBIAN_FRONTEND=noninteractive
ENV NVCC=nvcc

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    openssh-server \
    curl \
    git \
    python3 \
    python3-pip \
    && rm -rf /var/lib/apt/lists/* \
    && pip3 install --break-system-packages datasets

RUN mkdir -p /var/run/sshd /root/.ssh && \
    chmod 700 /root/.ssh

RUN sed -i 's/#PermitRootLogin prohibit-password/PermitRootLogin prohibit-password/' /etc/ssh/sshd_config && \
    sed -i 's/#PasswordAuthentication yes/PasswordAuthentication no/' /etc/ssh/sshd_config && \
    sed -i 's/#PubkeyAuthentication yes/PubkeyAuthentication yes/' /etc/ssh/sshd_config

COPY authorized_keys /root/.ssh/authorized_keys
RUN chmod 600 /root/.ssh/authorized_keys

EXPOSE 22

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
COPY wikigen/convert_wikitext.py ./wikigen/

RUN chmod +x train_gpu.sh

CMD ["/usr/sbin/sshd", "-D"]
