!!Update in progress!!

The official implementation of paper "SGSR: Style-subnets-assisted Generative latent bank for large-factor Super-Resolution with registered medical image dataset".

## Installation
The code is based on mmediting, which is now mmagic (https://github.com/open-mmlab/mmagic).
Please refer to [install.md](docs/en/install.md) for installation.

## Dataset
OASIS-1 (https://www.oasis-brains.org/)
Please use the pre-processed image.

## Getting Started

# Quick inference

CUDA_VISIBLE_DEVICES=2 bash tools/dist_test.sh \
  configs/restorers/sgsr/sgsr_oasis_8x_test.py \
  /path_to_trainedmodel \
  1

# Train

CUDA_VISIBLE_DEVICES=0 bash tools/dist_train.sh \
  configs/restorers/glean/glean_oasis_8x.py 1 \
  --work-dir /youworkdir
