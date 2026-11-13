FROM python:3.11.12-slim-bookworm@sha256:dbf1de478a55d6763afaa39c2f3d7b54b25230614980276de5cacdde79529d0c

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    TOKENIZERS_PARALLELISM=false \
    OMP_NUM_THREADS=4 \
    MPLCONFIGDIR=/tmp/matplotlib \
    HF_HOME=/cache/huggingface \
    NVIDIA_VISIBLE_DEVICES=all \
    NVIDIA_DRIVER_CAPABILITIES=compute,utility

RUN apt-get update \
    && apt-get install -y --no-install-recommends ca-certificates libgomp1 \
    && rm -rf /var/lib/apt/lists/*
COPY requirements.txt /opt/encoder-ranking/requirements.txt
RUN python -m pip install --no-cache-dir pip==25.1.1 \
    && python -m pip install --no-cache-dir --no-deps torch==2.7.1+cu126 --index-url https://download.pytorch.org/whl/cu126 \
    && python -m pip install --no-cache-dir -r /opt/encoder-ranking/requirements.txt \
    && python -m pip check \
    && python -m pip freeze > /opt/encoder-ranking/python-requirements.lock.txt \
    && diff -u /opt/encoder-ranking/requirements.txt /opt/encoder-ranking/python-requirements.lock.txt

WORKDIR /work/code
ENTRYPOINT ["python", "-u"]
