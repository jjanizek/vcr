#!/bin/bash
#SBATCH --job-name=vcr
#SBATCH --partition=roxanad
#SBATCH --gres=gpu:1
#SBATCH --time=12:00:00
#SBATCH --mem=16G
#SBATCH --cpus-per-task=1
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

# This source file is part of the VCR project
#
# SPDX-FileCopyrightText: 2025 Stanford University and the project authors (see AUTHORS.md)
#
# SPDX-License-Identifier: MIT

ml python/3.9.0
ml gcc/14.2.0
ml cuda/11.7.1

# Activate the virtual environment
python3 -m venv /home/groups/roxanad/sonnet/vcr/flam_env
source /home/groups/roxanad/sonnet/vcr/flam_env/bin/activate

export PYTHONPATH="/home/groups/roxanad/sonnet/vcr:$PYTHONPATH"
export TMPDIR=/scratch/users/$USER/tmp
export HF_HOME=/scratch/users/$USER/huggingface
export HF_DATASETS_CACHE=/scratch/users/$USER/huggingface/datasets
export TORCH_HOME=/scratch/users/$USER/torch

# Create all directories
mkdir -p $TMPDIR $HF_HOME $HF_DATASETS_CACHE $TORCH_HOME 

which python

# Install dependencies
pip3 install --no-cache-dir --upgrade pip
pip3 install --no-cache-dir torch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2 --index-url https://download.pytorch.org/whl/cu118
pip3 install --no-cache-dir pandas
pip3 install --no-cache-dir "transformers @ git+https://github.com/huggingface/transformers@01734dba842c29408c96caa5c345c9e415c7569b"
pip3 install --no-cache-dir open-flamingo==2.0.1
pip3 install --no-cache-dir open-clip-torch==2.23.0
pip3 install --no-cache-dir einops
pip3 install --no-cache-dir "accelerate @ git+https://github.com/huggingface/accelerate@4d583ad6a1f13d1d7617e6a37f791ec01a68413a"
pip3 install --no-cache-dir numpy==1.24.2
pip3 install --no-cache-dir scikit-learn==1.2.2
pip3 install --no-cache-dir protobuf==3.20.3
pip3 install --no-cache-dir timm==0.6.12
pip3 install --no-cache-dir matplotlib

python /home/groups/roxanad/sonnet/vcr/src/experiments/bootstrap_resample_for_pvalues.py