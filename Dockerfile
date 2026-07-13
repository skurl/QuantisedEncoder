# Build for HPC x86_64, even from Apple Silicon:
# docker buildx build --platform linux/amd64 -t ghcr.io/skurl/quantised-encoder:0.0.1 --push .

FROM mambaorg/micromamba:0.15.3

ARG MAMBA_DOCKERFILE_ACTIVATE=1
SHELL ["/bin/bash", "-lc"]

ENV MAMBA_ROOT_PREFIX=/opt/conda
ENV CONDA_SUBDIR=linux-64
ENV PATH=/opt/conda/bin:$PATH

COPY environment.yml /tmp/environment.yml

RUN micromamba install -y -p $MAMBA_ROOT_PREFIX -c conda-forge python=3.10 pip && \
    micromamba install -y -p $MAMBA_ROOT_PREFIX -f /tmp/environment.yml --override-channels && \
    $MAMBA_ROOT_PREFIX/bin/python -m pip install --no-cache-dir --upgrade pip && \
    $MAMBA_ROOT_PREFIX/bin/python -m pip install --no-cache-dir \
    --index-url https://download.pytorch.org/whl/cu124 \
    torch && \
    micromamba clean --all -y

WORKDIR /app

COPY . /app

CMD ["bash"]