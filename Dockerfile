# FROM swr.cn-north-4.myhuaweicloud.com/ddn-k8s/docker.io/nvidia/cuda:13.0.2-cudnn-devel-ubuntu24.04
FROM swr.cn-north-4.myhuaweicloud.com/ddn-k8s/docker.io/nvidia/cuda:13.0.0-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV PATH=/opt/conda/bin:$PATH
ENV CUDA_HOME=/usr/local/cuda
ENV LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/cuda/lib64

RUN apt-get update && apt-get install -y \
    curl \
    git \
    ca-certificates \
    build-essential \
    ffmpeg \
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender1 \
    pkg-config \
    && rm -rf /var/lib/apt/lists/*

RUN curl -fsSL https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-x86_64.sh \
    -o /tmp/miniforge.sh \
    && bash /tmp/miniforge.sh -b -p /opt/conda \
    && rm -f /tmp/miniforge.sh

RUN conda update -n base -c defaults conda

RUN conda config --set channel_priority strict \
    && conda config --add channels conda-forge \
    && conda config --add channels pytorch

RUN conda install -y python=3.12 

RUN pip config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple \
    && pip config set install.trusted-host pypi.tuna.tsinghua.edu.cn \
    && pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir pytorch-lightning pyrender opencv-python yacs scikit-image einops timm dill pandas rich \
    hydra-core hydra-submitit-launcher hydra-colorlog pyrootutils webdataset chump networkx==3.2.1 roma \
    joblib seaborn wandb appdirs appnope cython jsonlines pytest loguru optree fvcore black pycocotools \
    tensorboard huggingface_hub mhr fastapi uvicorn requests \
    && pip install --no-cache-dir 'git+https://github.com/facebookresearch/detectron2.git@a1ce2f9' --no-build-isolation --no-deps \
    && pip install --no-cache-dir git+https://github.com/microsoft/MoGe.git

RUN pip uninstall torch torchvision nvidia-nccl-cu12 -y

RUN conda install pymomentum=0.1.102 torchvision -c conda-forge -y \
    && conda clean -afy

WORKDIR /app
COPY . /app

EXPOSE 8000
CMD ["python", "server.py"]
