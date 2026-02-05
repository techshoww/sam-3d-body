# FROM swr.cn-north-4.myhuaweicloud.com/ddn-k8s/docker.io/nvidia/cuda:13.0.2-cudnn-devel-ubuntu24.04
FROM swr.cn-north-4.myhuaweicloud.com/ddn-k8s/docker.io/nvidia/cuda:13.0.0-devel-ubuntu22.04
# FROM swr.cn-north-4.myhuaweicloud.com/ddn-k8s/docker.io/pytorch/pytorch:2.8.0-cuda12.9-cudnn9-devel

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV PATH=/opt/myconda/bin:$PATH
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
    && bash /tmp/miniforge.sh -b -p /opt/myconda \
    && rm -f /tmp/miniforge.sh

RUN /bin/bash -lc "source /opt/myconda/etc/profile.d/conda.sh && conda activate base \
    && conda config --set show_channel_urls yes \
    && conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/main \
    && conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/r \
    && conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud/conda-forge"

RUN /bin/bash -lc "source /opt/myconda/etc/profile.d/conda.sh && conda activate base && conda update -n base -c defaults conda"

# RUN /bin/bash -lc "source /opt/myconda/etc/profile.d/conda.sh && conda activate base \
#     && conda config --set channel_priority flexible  \
#     && conda config --prepend channels nvidia \
#     && conda config --prepend channels pytorch  \
#     && conda config --append channels conda-forge"

RUN /bin/bash -lc "source /opt/myconda/etc/profile.d/conda.sh && conda activate base && conda install -y python=3.12"

RUN /bin/bash -lc "source /opt/myconda/etc/profile.d/conda.sh && conda activate base && pip config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple \
    && pip config set install.trusted-host pypi.tuna.tsinghua.edu.cn \
    && pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir pytorch-lightning pyrender opencv-python yacs scikit-image einops timm dill pandas rich \
    hydra-core hydra-submitit-launcher hydra-colorlog pyrootutils webdataset chump networkx==3.2.1 roma \
    joblib seaborn wandb appdirs appnope cython jsonlines pytest loguru optree fvcore black pycocotools \
    tensorboard huggingface_hub mhr fastapi uvicorn requests \
    && pip install --no-cache-dir 'git+https://github.com/facebookresearch/detectron2.git@a1ce2f9' --no-build-isolation --no-deps \
    && pip install --no-cache-dir git+https://github.com/microsoft/MoGe.git "
    # && pip uninstall torch torchvision nvidia-nccl-cu12 -y "
    # && conda install -y pymomentum pytorch=2.10.0 torchvision=0.25.0 cuda-version=13.0 -c conda-forge \
    # && conda clean -afy"


RUN /bin/bash -lc "source /opt/myconda/etc/profile.d/conda.sh && conda activate base && pip uninstall torch torchvision nvidia-nccl-cu12 -y"

# RUN /bin/bash -lc "source /opt/myconda/etc/profile.d/conda.sh && conda activate base && conda install pytorch==2.10 torchvision==0.25.0  -c conda-forge -y"

# RUN /bin/bash -lc "source /opt/myconda/etc/profile.d/conda.sh && conda activate base && conda install pymomentum torchvision pytorch=2.10.* cuda-version=13.0  -c conda-forge -y \
#     && conda clean -afy"


WORKDIR /app
COPY . /app

COPY entrypoint.sh /usr/local/bin/entrypoint.sh
RUN /bin/bash -lc "source /opt/myconda/etc/profile.d/conda.sh && conda activate base && chmod +x /usr/local/bin/entrypoint.sh"

EXPOSE 8000
ENTRYPOINT ["/usr/local/bin/entrypoint.sh"]
CMD ["uvicorn", "server:app", "--host", "0.0.0.0", "--port", "8000"]
