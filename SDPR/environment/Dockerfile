# hash:sha256:5db416afcb8e994ffc8366afac20ffb06e3443f08a8857d5e19802634307bb7c
FROM registry.codeocean.com/codeocean/miniconda3:4.9.2-cuda11.7.0-cudnn8-ubuntu20.04

ARG DEBIAN_FRONTEND=noninteractive

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        python3-pip=20.0.2-5ubuntu1.10 \
        python3-setuptools=45.2.0-1ubuntu0.1 \
        python3-wheel=0.34.2-1ubuntu0.1 \
    && rm -rf /var/lib/apt/lists/*

RUN pip install -U --no-cache-dir \
    numpy==1.21.6 \
    scikit-learn==1.0.2 \
    scipy==1.4.1 \
    torch==1.10.0

RUN pip3 install -U --no-cache-dir \
    tensorboardx==2.0

COPY postInstall /
RUN /postInstall
