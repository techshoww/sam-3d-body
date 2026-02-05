#!/usr/bin/env bash
set -e

source /opt/myconda/etc/profile.d/conda.sh
conda activate base
conda install pymomentum torchvision pytorch=2.10.* cuda-version=13.0 -y
exec "$@"
