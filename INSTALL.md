# Installation Guide for SAM 3D Body

## Setup Python Environment

### 1. Create and Activate Environment

```bash
conda create -n sam_3d_body python=3.12 -y
conda activate sam_3d_body
```

### 2. Install PyTorch

<!-- Please install PyTorch following the [official instructions](https://pytorch.org/get-started/locally/). -->

### 3. Install Python Dependencies

```bash
pip install pytorch-lightning pyrender opencv-python yacs scikit-image einops timm dill pandas rich hydra-core hydra-submitit-launcher hydra-colorlog pyrootutils webdataset chump networkx==3.2.1 roma joblib seaborn wandb appdirs appnope ffmpeg cython jsonlines pytest  loguru optree fvcore black pycocotools tensorboard huggingface_hub
```
<!-- xtcocotools -->
### 4. Install Detectron2

```bash
pip install 'git+https://github.com/facebookresearch/detectron2.git@a1ce2f9' --no-build-isolation --no-deps
```

### 5. Install MoGe (Optional)

```bash
pip install git+https://github.com/microsoft/MoGe.git
```

### 6. Install SAM3 (Optional)
```bash
# this is a minimal installation of sam3 only to support its inference 
git clone https://github.com/facebookresearch/sam3.git
cd sam3
pip install -e .
pip install decord psutil
```
<!-- pip uninstall torch torchvision
conda install -c conda-forge pymomentum=0.1.102="cuda129*" -y
pip install mhr
conda install torchvision==0.23.0 -->

conda install pymomentum torchvision -y
pip install mhr

libtorch                  2.10.0          cuda130_mkl_hfedd1fc_301    conda-forge
pytorch                   2.10.0          cuda130_mkl_py312_h9405518_301    conda-forge
pytorch-lightning         2.6.1                    pypi_0    pypi
torchmetrics              1.8.2                    pypi_0    pypi
torchvision               0.25.0          cuda130_py312_h2a13e54_2    conda-forge
torchvision-extra-decoders 0.0.2           py312h1b2fc9e_6    conda-forge

## Getting Model Checkpoints

We host model checkpoints on Hugging Face. **Available models:**
- [`facebook/sam-3d-body-dinov3`](https://huggingface.co/facebook/sam-3d-body-dinov3)
- [`facebook/sam-3d-body-vith`](https://huggingface.co/facebook/sam-3d-body-vith)


⚠️ Please note that you need to **request access** on the SAM 3D Body Hugging Face repos above. Once accepted, you need to be authenticated to download the checkpoints.

⚠️ SAM 3D Body is available via HuggingFace globally, **except** in comprehensively sanctioned jurisdictions. Sanctioned jurisdiction will result in requests being **rejected**.

