ARG BASE_IMAGE=patent-encoder-ranking:patent_ce_modernbert_20261107
FROM ${BASE_IMAGE}

# These SDK versions follow the released checkpoint's saved metadata; they do not
# establish the versions used during the authors' original training.
COPY requirements.lock.txt /opt/paper-reproduction/expected-requirements.lock.txt
RUN python -m pip install --no-cache-dir --no-deps transformers==4.46.2 tokenizers==0.20.3 \
    && python -m pip check \
    && mkdir -p /opt/paper-reproduction \
    && python -m pip freeze > /opt/paper-reproduction/requirements.lock.txt \
    && diff -u /opt/paper-reproduction/expected-requirements.lock.txt /opt/paper-reproduction/requirements.lock.txt
