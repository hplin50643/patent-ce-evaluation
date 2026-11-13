ARG BASE_IMAGE=patent-encoder-ranking:patent_ce_modernbert_20261107
FROM ${BASE_IMAGE}

# ModernBERT requires the existing Transformers 4.50.3 environment. Do not inherit
# the older 4.46.2 SDK used for the preceding Longformer reproduction container.
COPY requirements.lock.txt /opt/paper-modernbert/expected-requirements.lock.txt
RUN python -m pip check \
    && mkdir -p /opt/paper-modernbert \
    && python -m pip freeze > /opt/paper-modernbert/requirements.lock.txt \
    && diff -u /opt/paper-modernbert/expected-requirements.lock.txt /opt/paper-modernbert/requirements.lock.txt
